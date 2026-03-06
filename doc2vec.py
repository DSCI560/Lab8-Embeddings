#!/usr/bin/env python3
"""
DSCI-560 Lab 8 — Doc2Vec Embedding & Clustering Pipeline
=========================================================

SCRAPING STRATEGY: Reddit's undocumented public JSON API (zero credentials)
─────────────────────────────────────────────────────────────────────────────
Reddit exposes every public page as raw JSON by appending ".json" to any URL:
    https://www.reddit.com/r/technology/hot.json?limit=100&after=<token>

This is the SAME data PRAW fetches under the hood, but requires NO API key,
NO account, NO OAuth — just a proper User-Agent header.

It gives us: full selftext, score, upvote_ratio, num_comments, flair, domain,
             post URL, author, UTC timestamp, preview image URLs, all in one
             structured JSON response.

We also pull the comments JSON for each post permalink:
    https://www.reddit.com/r/technology/comments/<id>.json?limit=20&depth=1

MULTI-MIRROR FALLBACK ROTATION
  Primary: www.reddit.com  (fastest, richest data)
  Fallbacks: old.reddit.com, Redlib public instances (for listing pages only)
  → If www.reddit.com returns 429 or 5xx, the next mirror is tried automatically.
  → Redlib instances serve the same data from Reddit's internal API, proxied
    through their servers (OAuth token spoofing), so they work when Reddit itself
    throttles direct access.

ARTICLE ENRICHMENT: async aiohttp + trafilatura
  • Up to 15 concurrent HTTP fetches → 3 000 articles in ~3-5 min, not 2+ hrs
  • BLOCKED_DOMAINS list: instant skip of Bloomberg / WSJ / paywalls / social
    media with ZERO retries (was the main cause of the 2-3 hr runtime)
  • Per-request 8 s timeout; no exponential backoff spirals
  • Browser-like headers to bypass lightweight bot checks

OUTPUTS:  ./doc2vec_outputs/
    model_<config>.d2v            — 6 trained Doc2Vec models
    doc2vec_cluster_comparison.png — 6-panel PCA grid
    doc2vec_metrics_bar.png       — silhouette + DB bar charts
    metrics_summary.json          — all numeric results

DATABASE: lab8  |  table: doc2vec_posts
"""

import argparse
import asyncio
import aiohttp
import os, re, pickle, logging, time, json, random
import psycopg2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import requests
import nltk

try:
    import trafilatura
except ImportError:
    raise SystemExit("pip install trafilatura")

for _r, _p in [('punkt', 'tokenizers/punkt'),
               ('punkt_tab', 'tokenizers/punkt_tab'),
               ('stopwords', 'corpora/stopwords')]:
    try:    nltk.data.find(_p)
    except: nltk.download(_r, quiet=True)

STOPWORDS   = set(stopwords.words('english'))
OUTPUT_DIR  = "doc2vec_outputs"
CONCURRENCY = 15      # async article fetch workers

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# ─────────────────────────────────────────────────────────────────────────────
# REDDIT MIRROR POOL
#   Primary = reddit.com JSON API (no key, ~60 req/min per IP is safe)
#   If 429/5xx → rotate through old.reddit, then Redlib instances
#   Redlib instances are community-hosted; they use OAuth token spoofing so
#   they bypass Reddit's direct throttling, but may have their own uptime issues.
# ─────────────────────────────────────────────────────────────────────────────
REDDIT_MIRRORS = [
    "https://www.reddit.com",
    "https://old.reddit.com",
    # Redlib public instances — listing pages only (no .json suffix needed,
    # Redlib instances that expose /r/<sub>/hot.json are rare; we use the main
    # reddit.com JSON endpoints and fall back to old.reddit.com HTML+JSON)
]

# Separate user-agents to rotate — looks less bot-like
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

IMAGE_EXTS = re.compile(r"\.(jpg|jpeg|png|gif|webp)(\?.*)?$", re.I)

# ─────────────────────────────────────────────────────────────────────────────
# ARTICLE PAYWALL / BOT-WALL BLOCKLIST  →  instant zero-retry skip
# ─────────────────────────────────────────────────────────────────────────────
BLOCKED_DOMAINS = {
    # Hard paywalls
    "bloomberg.com","wsj.com","nytimes.com","ft.com","thetimes.co.uk",
    "economist.com","barrons.com","seekingalpha.com","hbr.org",
    # Soft paywalls / login walls
    "washingtonpost.com","businessinsider.com","theatlantic.com",
    "newyorker.com","medium.com",
    # Aggressive bot-detection / Cloudflare
    "wired.com","forbes.com","reuters.com","techcrunch.com",
    # Video / social — no article text
    "youtube.com","youtu.be","twitch.tv","twitter.com","x.com",
    "instagram.com","tiktok.com","facebook.com",
    # Reddit itself — text already captured from JSON
    "reddit.com","redd.it","i.redd.it","v.redd.it","gallery.reddit.com",
}

def _host(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return re.sub(r"^www\.", "", urlparse(url).hostname or "")
    except Exception:
        return ""

def is_blocked(url: str) -> bool:
    if not url:
        return True
    h = _host(url)
    return any(h == d or h.endswith("." + d) for d in BLOCKED_DOMAINS)


# ─────────────────────────────────────────────────────────────────────────────
# REDDIT JSON SCRAPER  (no API key)
# ─────────────────────────────────────────────────────────────────────────────

def _reddit_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
    }

def _reddit_get(url: str, params: dict = None,
                retries: int = 4) -> dict | None:
    """
    GET a reddit .json endpoint, rotating User-Agent and adding small jitter.
    Returns parsed JSON dict or None on permanent failure.
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_reddit_headers(),
                             params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 2 ** attempt + random.uniform(1, 3)
                logging.warning(f"429 on {url} — sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            logging.warning(f"HTTP {r.status_code} for {url}")
            return None
        except Exception as e:
            logging.warning(f"Request error ({e}) — retry {attempt+1}")
            time.sleep(2 ** attempt)
    return None


def _parse_post(child: dict, subreddit_name: str) -> dict | None:
    """Extract a clean post dict from a Reddit JSON 'child' node."""
    d = child.get("data", {})
    if not d or d.get("stickied"):
        return None

    reddit_id = d.get("id", "")
    if not reddit_id:
        return None

    url       = d.get("url", "")
    image_url = None
    if IMAGE_EXTS.search(url):
        image_url = url
    else:
        try:
            # Reddit provides preview images in JSON
            image_url = (d["preview"]["images"][0]
                          ["source"]["url"].replace("&amp;", "&"))
        except Exception:
            pass

    return {
        "reddit_id":    reddit_id,
        "subreddit":    subreddit_name,
        "title":        d.get("title", ""),
        "selftext":     (d.get("selftext", "") or "").strip(),
        "image_url":    image_url,
        "author":       d.get("author", ""),
        "created_utc":  datetime.utcfromtimestamp(d.get("created_utc", 0)),
        "score":        d.get("score", 0),
        "upvote_ratio": d.get("upvote_ratio", 0.0),
        "num_comments": d.get("num_comments", 0),
        "flair":        d.get("link_flair_text", "") or "",
        "post_url":     url,
        "domain":       d.get("domain", ""),
        "permalink":    d.get("permalink", ""),
        "top_comments": "",   # filled in next step
    }


def _fetch_comments(permalink: str, limit: int = 20) -> str:
    """Fetch top-level comment texts for a post via JSON endpoint."""
    url  = f"https://www.reddit.com{permalink}.json"
    data = _reddit_get(url, params={"limit": limit, "depth": 1})
    if not data or len(data) < 2:
        return ""
    texts = []
    try:
        for child in data[1]["data"]["children"]:
            body = child.get("data", {}).get("body", "")
            if body and 10 < len(body) < 600:
                texts.append(body.strip())
    except Exception:
        pass
    return " ".join(texts[:12])


def scrape_subreddit_json(subreddit: str, max_posts: int) -> list:
    """
    Scrape `max_posts` from a subreddit using reddit.com's public JSON API.
    Mixes hot + new + top(month) for thematic variety.
    """
    posts = []
    seen  = set()

    feeds = [
        ("hot",  {}),
        ("new",  {}),
        ("top",  {"t": "month"}),
    ]
    limits_per_feed = [max_posts, max_posts // 3, max_posts // 3]

    for (sort, extra_params), feed_limit in zip(feeds, limits_per_feed):
        after   = None
        fetched = 0

        while fetched < feed_limit:
            batch = min(100, feed_limit - fetched)
            params = {"limit": batch, **extra_params}
            if after:
                params["after"] = after

            url  = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            data = _reddit_get(url, params=params)
            if not data:
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                post = _parse_post(child, subreddit)
                if post and post["reddit_id"] not in seen:
                    seen.add(post["reddit_id"])
                    posts.append(post)
                    fetched += 1

            after = data.get("data", {}).get("after")
            if not after:
                break

            # Polite delay — ~30 req/min well within Reddit's 60/min limit
            time.sleep(2 + random.uniform(0, 1))

    logging.info(f"r/{subreddit}: {len(posts)} posts via JSON API")
    return posts


def enrich_with_comments(posts: list) -> None:
    """In-place: fetch top-level comments for each post (synchronous, polite)."""
    for post in tqdm(posts, desc="Fetching comments"):
        if post.get("permalink"):
            post["top_comments"] = _fetch_comments(post["permalink"])
            time.sleep(1.5 + random.uniform(0, 0.5))


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC ARTICLE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


async def _fetch_article(session: aiohttp.ClientSession,
                          url: str,
                          sem: asyncio.Semaphore) -> str:
    """Fetch one URL and extract main article text. Never raises."""
    if is_blocked(url):
        return ""
    async with sem:
        try:
            async with session.get(
                url, headers=BROWSER_HEADERS,
                timeout=aiohttp.ClientTimeout(total=8),
                ssl=False, allow_redirects=True, max_redirects=4,
            ) as resp:
                if resp.status != 200:
                    return ""
                html = await resp.text(errors="replace")
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
                no_fallback=False,
            )
            return (text or "")[:3000]
        except Exception:
            return ""


async def fetch_articles_async(urls: list) -> list:
    """Fetch all URLs concurrently. Returns list in same order as input."""
    sem       = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 5,
                                     ssl=False, ttl_dns_cache=300)
    results   = [""] * len(urls)

    async with aiohttp.ClientSession(connector=connector) as session:
        futures = {
            asyncio.ensure_future(_fetch_article(session, u, sem)): i
            for i, u in enumerate(urls)
        }
        pbar = tqdm(total=len(futures), desc="Articles (async)", unit="url")
        for fut in asyncio.as_completed(futures):
            results[futures[fut]] = await fut
            pbar.update(1)
        pbar.close()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE  —  lab8  |  doc2vec_posts
# ─────────────────────────────────────────────────────────────────────────────

def get_db_conn(host, user, password, database="lab8"):
    conn = psycopg2.connect(host=host, database=database,
                            user=user, password=password)
    conn.autocommit = True
    logging.info(f"Connected to '{database}' on {host}")
    return conn


def ensure_table(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS doc2vec_posts (
        id                   SERIAL PRIMARY KEY,
        reddit_id            VARCHAR(50)  UNIQUE,
        subreddit            VARCHAR(100),
        title                TEXT,
        selftext             TEXT,
        image_url            TEXT,
        image_path           TEXT,
        image_ocr_text       TEXT,
        author_masked        VARCHAR(128),
        created_utc          TIMESTAMP,
        score                INTEGER,
        upvote_ratio         FLOAT,
        num_comments         INTEGER,
        flair                TEXT,
        post_url             TEXT,
        domain               TEXT,
        article_text         TEXT,
        top_comments         TEXT,
        cleaned_text         TEXT,
        embedding            BYTEA,
        cluster_id           INTEGER,
        keywords             TEXT,
        distance_to_centroid DOUBLE PRECISION,
        fetched_at           TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_d2v_cluster ON doc2vec_posts(cluster_id);
    CREATE INDEX IF NOT EXISTS idx_d2v_dist    ON doc2vec_posts(distance_to_centroid);
    """)
    cur.close()
    logging.info("Table doc2vec_posts ready.")


def insert_post(conn, r):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO doc2vec_posts
      (reddit_id, subreddit, title, selftext, image_url, image_path,
       image_ocr_text, author_masked, created_utc,
       score, upvote_ratio, num_comments, flair, post_url, domain,
       article_text, top_comments, cleaned_text,
       embedding, cluster_id, keywords, distance_to_centroid, fetched_at)
    VALUES (%s,%s,%s,%s,%s,%s, %s,%s,%s, %s,%s,%s,%s,%s,%s,
            %s,%s,%s, %s,%s,%s,%s,%s)
    ON CONFLICT (reddit_id) DO UPDATE SET
        cleaned_text         = EXCLUDED.cleaned_text,
        embedding            = EXCLUDED.embedding,
        cluster_id           = EXCLUDED.cluster_id,
        keywords             = EXCLUDED.keywords,
        distance_to_centroid = EXCLUDED.distance_to_centroid,
        article_text         = EXCLUDED.article_text,
        top_comments         = EXCLUDED.top_comments,
        fetched_at           = EXCLUDED.fetched_at
    """, (
        r['reddit_id'],    r['subreddit'],    r['title'],
        r['selftext'],     r['image_url'],    r.get('image_path'),
        r['image_ocr_text'], r['author_masked'], r['created_utc'],
        r['score'],        r['upvote_ratio'], r['num_comments'],
        r['flair'],        r['post_url'],     r['domain'],
        r['article_text'], r['top_comments'], r['cleaned_text'],
        r['embedding'],    r['cluster_id'],   r['keywords'],
        r['distance_to_centroid'], r['fetched_at'],
    ))
    cur.close()


# ─────────────────────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"http\S+",          " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]",   " ", text)
    text = re.sub(r"\b\d+\b",          " ", text)
    text = re.sub(r"\s+",              " ", text).strip()
    return text.lower()

def tokenize(text: str) -> list:
    return [w for w in word_tokenize(text.lower())
            if w.isalpha() and w not in STOPWORDS and len(w) > 2]

def mask_author(a: str) -> str:
    return "user_unknown" if not a else "user_" + str(abs(hash(a)) % 10**8)

def ocr_image(url: str) -> str:
    if not url:
        return ""
    try:
        import pytesseract
        r   = requests.get(url, headers=BROWSER_HEADERS, timeout=10)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if min(img.size) < 300:
            s = 300 / min(img.size)
            img = img.resize((int(img.width*s), int(img.height*s)),
                             Image.LANCZOS)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# DOC2VEC — SIX CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────
DOC2VEC_CONFIGS = [
    # (name,             vec_size, min_cnt, epochs, dm,  window)
    ("dbow_50_fast",          50,       2,     20,   0,    5),
    ("dbow_100_std",         100,       2,     30,   0,    5),
    ("dbow_200_deep",        200,       2,     50,   0,    8),
    ("dm_100_std",           100,       2,     30,   1,    5),
    ("dm_200_deep",          200,       2,     50,   1,    8),
    ("dm_300_large",         300,       1,     60,   1,   10),
]

def embed_and_cluster(records: list) -> list:
    texts = [
        " ".join(filter(None, [
            r["cleaned_text"], r.get("article_text",""),
            r.get("top_comments",""), r.get("image_ocr_text",""),
        ]))
        for r in records
    ]
    k = min(10, max(2, len(records) // 50))
    logging.info(f"k={k} clusters for {len(records)} records")

    results = []
    for (name, vsz, mc, ep, dm, win) in DOC2VEC_CONFIGS:
        logging.info(f"Training Doc2Vec: {name}")
        tagged = [TaggedDocument(words=word_tokenize(t), tags=[str(i)])
                  for i, t in enumerate(texts)]
        model = Doc2Vec(vector_size=vsz, min_count=mc, epochs=ep,
                        dm=dm, window=win, workers=4, seed=42,
                        dbow_words=(1 if dm==0 else 0))
        model.build_vocab(tagged)
        model.train(tagged, total_examples=model.corpus_count,
                    epochs=model.epochs)
        model.save(os.path.join(OUTPUT_DIR, f"model_{name}.d2v"))

        emb    = np.array([model.infer_vector(word_tokenize(t), epochs=30)
                           for t in texts])
        normed = normalize(emb)
        km     = KMeans(n_clusters=k, n_init=15,
                        random_state=42, max_iter=500)
        lbls   = km.fit_predict(normed)

        sil = silhouette_score(normed, lbls, metric="cosine")
        db  = davies_bouldin_score(normed, lbls)
        ch  = calinski_harabasz_score(normed, lbls)
        print(f"\n  [{name}] vec={vsz} dm={dm} win={win} ep={ep}")
        print(f"    Silhouette (↑): {sil:.4f}  DB (↓): {db:.4f}  CH (↑): {ch:.2f}")
        results.append(dict(name=name, embeddings=emb, labels=lbls,
                            centroids=km.cluster_centers_,
                            silhouette=sil, db=db, ch=ch))

    results.sort(key=lambda x: x["silhouette"], reverse=True)
    best = results[0]
    logging.info(f"Best: {best['name']}  Sil={best['silhouette']:.4f}")

    # TF-IDF keywords
    vect       = TfidfVectorizer(max_features=2000, stop_words="english",
                                  ngram_range=(1,2))
    tmat       = vect.fit_transform(texts)
    fnames     = vect.get_feature_names_out()
    cluster_kw = {}
    for cid in range(k):
        idx = np.where(best["labels"]==cid)[0]
        if not len(idx): cluster_kw[cid]=[]; continue
        mt  = np.asarray(tmat[idx].mean(axis=0)).flatten()
        cluster_kw[cid] = [fnames[i] for i in mt.argsort()[-8:][::-1]]

    # 6-panel PCA
    fig, axes = plt.subplots(2, 3, figsize=(18,11))
    for ax, res in zip(axes.flatten(), results):
        red = PCA(2, random_state=42).fit_transform(normalize(res["embeddings"]))
        for cid in range(k):
            pts = red[res["labels"]==cid]
            ax.scatter(pts[:,0], pts[:,1], s=8, alpha=0.55)
        ax.set_title(
            f"{res['name']}\nSil={res['silhouette']:.3f} "
            f"DB={res['db']:.3f} CH={res['ch']:.0f}", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("Doc2Vec — 6-Config Cluster Comparison (PCA)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"doc2vec_cluster_comparison.png"), dpi=150)
    plt.close()

    # Metrics bars
    names = [r["name"] for r in results]; x = np.arange(len(names))
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
    ax1.bar(x,[r["silhouette"] for r in results], color="steelblue")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_title("Silhouette (↑ better)")
    ax2.bar(x,[r["db"] for r in results], color="salmon")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_title("Davies-Bouldin (↓ better)")
    fig.suptitle("Doc2Vec Metrics Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"doc2vec_metrics_bar.png"), dpi=130)
    plt.close()

    # Summary table
    print("\n" + "="*70)
    print(f"{'Config':<22} {'Silhouette':>11} {'DB':>14} {'CH':>10}")
    print("-"*70)
    for res in results:
        tag = " ← BEST" if res["name"]==best["name"] else ""
        print(f"{res['name']:<22} {res['silhouette']:>11.4f} "
              f"{res['db']:>14.4f} {res['ch']:>10.2f}{tag}")
    print("="*70)

    with open(os.path.join(OUTPUT_DIR,"metrics_summary.json"),"w") as f:
        json.dump([{k:v for k,v in r.items()
                    if k not in ("embeddings","labels","centroids")}
                   for r in results], f, indent=2)

    # Write back
    be=best["embeddings"]; bl=best["labels"]; bc=best["centroids"]
    nb=normalize(be)
    for i,r in enumerate(records):
        cid=int(bl[i])
        r["embedding"]            = pickle.dumps(be[i])
        r["cluster_id"]           = cid
        r["keywords"]             = ", ".join(cluster_kw.get(cid,[]))
        r["distance_to_centroid"] = float(np.linalg.norm(nb[i]-bc[cid]))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = get_db_conn(args.db_host, args.db_user, args.db_pass)
    ensure_table(conn)

    # 1. Scrape via public JSON API
    all_posts = []
    for sub in args.subs:
        all_posts.extend(scrape_subreddit_json(sub, args.num))
    logging.info(f"Total posts: {len(all_posts)}")

    # 2. Fetch comments (synchronous, polite pacing)
    if args.fetch_comments:
        enrich_with_comments(all_posts)

    # 3. Async article enrichment (fast)
    urls          = [p["post_url"] for p in all_posts]
    article_texts = asyncio.run(fetch_articles_async(urls))

    # 4. Build records
    records = []
    for p, art in tqdm(zip(all_posts, article_texts),
                        total=len(all_posts), desc="Building records"):
        image_ocr = ocr_image(p["image_url"]) if args.images else ""
        raw = " ".join(filter(None,[
            p["title"], p["selftext"], art,
            p["top_comments"], image_ocr, p.get("flair",""),
        ]))
        cleaned = clean_text(raw)
        if len(tokenize(cleaned)) < 5:
            continue
        records.append({
            "reddit_id":       p["reddit_id"],
            "subreddit":       p["subreddit"],
            "title":           p["title"],
            "selftext":        p["selftext"],
            "image_url":       p.get("image_url"),
            "image_path":      None,
            "image_ocr_text":  image_ocr,
            "author_masked":   mask_author(p["author"]),
            "created_utc":     p["created_utc"],
            "score":           p.get("score",0),
            "upvote_ratio":    p.get("upvote_ratio",0.0),
            "num_comments":    p.get("num_comments",0),
            "flair":           p.get("flair",""),
            "post_url":        p.get("post_url",""),
            "domain":          p.get("domain",""),
            "article_text":    art,
            "top_comments":    p.get("top_comments",""),
            "cleaned_text":    cleaned,
            "embedding":       None,
            "cluster_id":      None,
            "keywords":        None,
            "distance_to_centroid": None,
            "fetched_at":      datetime.utcnow(),
        })

    logging.info(f"Records after filter: {len(records)}")
    if records:
        embed_and_cluster(records)
        for r in tqdm(records, desc="Inserting to DB"):
            insert_post(conn, r)
    logging.info(f"Done. Outputs → ./{OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE QUERY
# ─────────────────────────────────────────────────────────────────────────────

def interactive_query(conn):
    print("\nInteractive query — 'exit' to quit.\n")
    cur = conn.cursor()
    cur.execute("SELECT embedding, cluster_id, title, keywords "
                "FROM doc2vec_posts WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    if not rows:
        print("No embedded posts found."); return

    embs, cids, kws = [], [], []
    for row in rows:
        embs.append(pickle.loads(row[0])); cids.append(row[1])
        kws.append(row[3] or "")

    embs = np.array(embs); cids = np.array(cids)
    centroids = {c: embs[cids==c].mean(axis=0) for c in sorted(set(cids))}

    while True:
        q = input("Keywords: ").strip()
        if q.lower()=="exit": break
        tokens = tokenize(clean_text(q))
        scores = np.array([sum(1 for t in tokens if t in kw) for kw in kws],
                          dtype=float)
        vec    = (embs[scores.argsort()[-20:]].mean(axis=0)
                  if scores.max()>0 else embs.mean(axis=0))
        best   = min(centroids, key=lambda c: np.linalg.norm(vec-centroids[c]))
        print(f"\n→ Cluster {best}")
        cur.execute("SELECT title, keywords FROM doc2vec_posts "
                    "WHERE cluster_id=%s LIMIT 8", (best,))
        for t,kw in cur.fetchall():
            print(f"  • {t}\n    Keywords: {kw}")
        print()
    cur.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Lab 8 Doc2Vec — Reddit JSON API + async article enrichment")
    p.add_argument("--subs", nargs="+",
                   default=["technology","technews","tech",
                            "cybersecurity","netsec","windowssecurity"])
    p.add_argument("--num",             type=int, default=500)
    p.add_argument("--db-host",         required=True)
    p.add_argument("--db-user",         required=True)
    p.add_argument("--db-pass",         required=True)
    p.add_argument("--interval",        type=int, default=0)
    p.add_argument("--images",          action="store_true",
                   help="Enable OCR on post images")
    p.add_argument("--fetch-comments",  action="store_true", default=True,
                   help="Fetch top-level comments per post (default: on)")
    p.add_argument("--no-fetch-comments", dest="fetch_comments",
                   action="store_false")
    p.add_argument("--query",           action="store_true")
    args = p.parse_args()

    if args.interval > 0:
        while True:
            run_pipeline(args); time.sleep(args.interval * 60)
    else:
        run_pipeline(args)
        if args.query:
            interactive_query(get_db_conn(args.db_host, args.db_user, args.db_pass))

if __name__ == "__main__":
    main()