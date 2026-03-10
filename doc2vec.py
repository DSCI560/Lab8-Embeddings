#!/usr/bin/env python3
# DSCI-560 Lab 8 - Doc2Vec Embedding and Clustering Pipeline (OPTIMIZED)
#
# KEY CHANGES vs original:
#   1. --skip-enrich flag: uses only scraped title/selftext/flair - no comment fetches
#      (title text alone is sufficient for meaningful NLP clustering)
#   2. --from-db flag: reads already-scraped posts from DB instead of re-scraping
#   3. Batch DB inserts (executemany) instead of one-by-one
#   4. infer_vector epochs reduced 30→10 (stable enough for 3k docs)
#   5. Fixed variable name inconsistencies (brHead, Swords, etc.)
#   6. Checkpoint: saves/loads scraped posts as JSON so crashes don't restart from zero
#   7. Article enrichment still async but --skip-articles skips external fetch entirely
#   8. 429 backoff on comment fetch removed by default (comments skipped)

import argparse
import asyncio
import aiohttp
import os
import re
import pickle
import logging
import time
import json
import random
import psycopg2
import psycopg2.extras
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import nltk

from datetime import datetime
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from urllib.parse import urlparse
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise SystemExit("pip install beautifulsoup4")

try:
    import trafilatura
except ImportError:
    raise SystemExit("pip install trafilatura")

for _r, _p in [
    ("punkt",     "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
]:
    try:
        nltk.data.find(_p)
    except LookupError:
        nltk.download(_r, quiet=True)

STOPWORDS = set(stopwords.words("english"))
OUT_DIR   = "doc2vec_outputs"
CONCURRENCY = 25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

BROWSER_HEADERS = {
    "User-Agent": USER_AGENTS[0],
    "Accept":          "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

IMAGE_EXTS = re.compile(r"\.(jpg|jpeg|png|gif|webp)(\?.*)?$", re.I)

BLOCKED_DOMAINS = {
    "bloomberg.com", "wsj.com", "nytimes.com", "ft.com", "thetimes.co.uk",
    "economist.com", "barrons.com", "seekingalpha.com", "hbr.org",
    "washingtonpost.com", "businessinsider.com", "theatlantic.com",
    "newyorker.com", "medium.com", "wired.com", "forbes.com",
    "reuters.com", "techcrunch.com",
    "youtube.com", "youtu.be", "twitch.tv", "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "facebook.com",
    "reddit.com", "redd.it", "i.redd.it", "v.redd.it", "gallery.reddit.com",
}

CHECKPOINT_FILE = "doc2vec_posts_checkpoint.json"


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _host(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", urlparse(url).hostname or "")
    except Exception:
        return ""


def is_blocked(url: str) -> bool:
    if not url:
        return True
    h = _host(url)
    return any(h == d or h.endswith("." + d) for d in BLOCKED_DOMAINS)


# ─── SCRAPER ────────────────────────────────────────────────────────────────

def _reddit_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    s.cookies.set("over18", "1", domain=".old.reddit.com")
    return s


def _get_page(session, url, retries=5):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20, allow_redirects=True)
            if r.status_code == 200:
                return BeautifulSoup(r.text, "html.parser")
            if r.status_code == 429:
                wait = 10 * (attempt + 1) + random.uniform(2, 5)
                logging.warning(f"429 from {url} - waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(4 ** attempt)
                continue
            logging.warning(f"HTTP {r.status_code} for {url}")
            return None
        except Exception as e:
            logging.warning(f"Request error: {e} - retry {attempt + 1}")
            time.sleep(4 ** attempt)
    return None


def _parse_old_reddit_page(soup, subreddit):
    posts = []
    for thing in soup.select("div.thing"):
        try:
            if "stickied" in thing.get("class", []):
                continue
            if thing.get("data-promoted") == "true":
                continue
            reddit_id = thing.get("data-fullname", "").replace("t3_", "")
            if not reddit_id:
                continue
            title_tag = thing.select_one("a.title")
            title     = title_tag.get_text(strip=True) if title_tag else ""
            post_url  = thing.get("data-url", "")
            permalink = thing.get("data-permalink", "")
            author    = thing.get("data-author", "")
            try:
                score = int(thing.get("data-score", "0") or "0")
            except ValueError:
                score = 0
            try:
                num_comments = int(thing.get("data-comments-count", "0") or "0")
            except ValueError:
                num_comments = 0
            timestamp_tag = thing.select_one("time")
            created_utc   = datetime.utcnow().isoformat()
            if timestamp_tag and timestamp_tag.get("datetime"):
                created_utc = timestamp_tag["datetime"]
            domain_tag = thing.select_one("span.domain a")
            domain     = domain_tag.get_text(strip=True) if domain_tag else ""
            flair_tag  = thing.select_one("span.linkflairlabel")
            flair      = flair_tag.get_text(strip=True) if flair_tag else ""
            image_url  = post_url if IMAGE_EXTS.search(post_url) else None
            try:
                upvote_ratio = float(thing.get("data-upvote-ratio", "0") or "0")
            except ValueError:
                upvote_ratio = 0.0
            posts.append({
                "reddit_id": reddit_id, "subreddit": subreddit,
                "title": title, "selftext": "", "image_url": image_url,
                "author": author, "created_utc": created_utc,
                "score": score, "upvote_ratio": upvote_ratio,
                "num_comments": num_comments, "flair": flair,
                "post_url": post_url, "domain": domain,
                "permalink": permalink, "top_comments": "",
            })
        except Exception:
            continue
    return posts


def scrape_subreddit(subreddit, max_posts):
    session = _reddit_session()
    posts, seen = [], set()
    feeds  = [
        f"https://old.reddit.com/r/{subreddit}/hot/",
        f"https://old.reddit.com/r/{subreddit}/new/",
        f"https://old.reddit.com/r/{subreddit}/top/?t=month",
    ]
    quotas = [max_posts, max_posts // 3, max_posts // 3]
    for feed_url, quota in zip(feeds, quotas):
        collected, url, page_num = 0, feed_url, 0
        while collected < quota and len(posts) < max_posts:
            page_num += 1
            feed_name = feed_url.rstrip("/").split("/")[-1].split("?")[0]
            logging.info(f"r/{subreddit} | feed={feed_name} | page={page_num} | collected={len(posts)}")
            soup = _get_page(session, url)
            if not soup:
                break
            page_posts = _parse_old_reddit_page(soup, subreddit)
            if not page_posts:
                break
            for p in page_posts:
                if p["reddit_id"] not in seen and len(posts) < max_posts:
                    seen.add(p["reddit_id"])
                    posts.append(p)
                    collected += 1
            next_btn = soup.select_one("span.next-button a")
            if not next_btn:
                break
            url = next_btn["href"]
            time.sleep(3 + random.uniform(1, 2))
        if len(posts) >= max_posts:
            break
    logging.info(f"r/{subreddit}: scraped {len(posts)} posts total")
    return posts[:max_posts]


# ─── OPTIONAL ENRICHMENT (COMMENTS) ─────────────────────────────────────────
# OPTIMIZATION: This was the main bottleneck (~4s × 3000 posts = 3+ hours).
# Skip with --skip-enrich (default True). Title + flair gives good clustering.

def enrich_with_comments(posts, limit=8):
    """
    Fetch top comments only. Selftext skipped (not worth the extra request).
    Only called when --enrich flag is explicitly set.
    Uses a single request per post (combined comment page).
    """
    session = _reddit_session()
    for i, post in enumerate(tqdm(posts, desc="Enriching posts")):
        permalink = post.get("permalink", "")
        if not permalink:
            continue
        url  = f"https://old.reddit.com{permalink}?limit=25&depth=1"
        soup = _get_page(session, url)
        if soup:
            texts = []
            for div in soup.select("div.commentarea div.entry div.usertext-body div.md"):
                t = div.get_text(separator=" ", strip=True)
                if 10 < len(t) < 500:
                    texts.append(t)
                if len(texts) >= limit:
                    break
            post["top_comments"] = " ".join(texts)
            # selftext from same page
            body = soup.select_one("div.usertext-body div.md")
            if body and not post.get("selftext"):
                post["selftext"] = body.get_text(separator=" ", strip=True)
        # REDUCED sleep: 1s base instead of 2-5s
        time.sleep(1 + random.uniform(0.3, 0.7))
        if (i + 1) % 20 == 0:
            time.sleep(3)


# ─── ASYNC ARTICLE FETCH ────────────────────────────────────────────────────

async def _fetch_article(session, url, sem):
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
                html, include_comments=False, include_tables=False,
                favor_precision=True, no_fallback=False,
            )
            return (text or "")[:2000]
        except Exception:
            return ""


async def fetch_articles_async(urls):
    sem       = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 5, ssl=False, ttl_dns_cache=300)
    pbar      = tqdm(total=len(urls), desc="Fetching articles", unit="url")

    async def _tracked(session, url):
        result = await _fetch_article(session, url, sem)
        pbar.update(1)
        return result

    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(*[_tracked(session, u) for u in urls])
    pbar.close()
    return list(results)


# ─── DATABASE ───────────────────────────────────────────────────────────────

def get_db_conn(host, user, password, database="lab8"):
    conn = psycopg2.connect(host=host, port=5433, database=database, user=user, password=password)
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


def load_posts_from_db(conn):
    """Read already-stored posts from DB to skip re-scraping."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT reddit_id, subreddit, title, selftext, flair,
               post_url, domain, article_text, top_comments,
               image_url, author_masked, created_utc,
               score, upvote_ratio, num_comments
        FROM doc2vec_posts
    """)
    rows = cur.fetchall()
    cur.close()
    logging.info(f"Loaded {len(rows)} posts from DB.")
    return [dict(r) for r in rows]


def batch_insert_posts(conn, records):
    """Insert all records in one executemany call - much faster than one-by-one."""
    cur = conn.cursor()
    rows = []
    for r in records:
        rows.append((
            r["reddit_id"], r["subreddit"], r["title"],
            r.get("selftext", ""), r.get("image_url"), None,
            r.get("image_ocr_text", ""), r.get("author_masked", ""),
            r.get("created_utc"), r.get("score", 0),
            r.get("upvote_ratio", 0.0), r.get("num_comments", 0),
            r.get("flair", ""), r.get("post_url", ""), r.get("domain", ""),
            r.get("article_text", ""), r.get("top_comments", ""),
            r.get("cleaned_text", ""),
            r.get("embedding"), r.get("cluster_id"), r.get("keywords"),
            r.get("distance_to_centroid"), r.get("fetched_at", datetime.utcnow()),
        ))
    psycopg2.extras.execute_batch(cur, """
        INSERT INTO doc2vec_posts
          (reddit_id, subreddit, title, selftext, image_url, image_path,
           image_ocr_text, author_masked, created_utc,
           score, upvote_ratio, num_comments, flair, post_url, domain,
           article_text, top_comments, cleaned_text,
           embedding, cluster_id, keywords, distance_to_centroid, fetched_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (reddit_id) DO UPDATE SET
            cleaned_text         = EXCLUDED.cleaned_text,
            embedding            = EXCLUDED.embedding,
            cluster_id           = EXCLUDED.cluster_id,
            keywords             = EXCLUDED.keywords,
            distance_to_centroid = EXCLUDED.distance_to_centroid,
            article_text         = EXCLUDED.article_text,
            top_comments         = EXCLUDED.top_comments,
            fetched_at           = EXCLUDED.fetched_at
    """, rows, page_size=200)
    conn.commit()
    cur.close()
    logging.info(f"Batch-inserted {len(rows)} records.")


# ─── TEXT UTILITIES ──────────────────────────────────────────────────────────

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"http\S+",        " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\b\d+\b",        " ", text)
    text = re.sub(r"\s+",            " ", text).strip()
    return text.lower()


def tokenize(text):
    return [
        w for w in word_tokenize(text.lower())
        if w.isalpha() and w not in STOPWORDS and len(w) > 2
    ]


def mask_author(a):
    return "user_unknown" if not a else "user_" + str(abs(hash(a)) % 10**8)


def ocr_image(url):
    if not url:
        return ""
    try:
        import pytesseract
        r   = requests.get(url, headers=BROWSER_HEADERS, timeout=10)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if min(img.size) < 300:
            s = 300 / min(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


# ─── DOC2VEC - SIX CONFIGURATIONS ───────────────────────────────────────────
# Assignment requires 3 different vector sizes minimum.
# We use 3 DBOW + 3 DM for a thorough comparison.

DOC2VEC_CONFIGS = [
    # (name,            vec_size, min_count, epochs, dm, window)
    ("dbow_50_fast",         50,         2,     20,   0,    5),
    ("dbow_100_std",        100,         2,     30,   0,    5),
    ("dbow_200_deep",       200,         2,     40,   0,    8),
    ("dm_100_std",          100,         2,     30,   1,    5),
    ("dm_200_deep",         200,         2,     40,   1,    8),
    ("dm_300_large",        300,         1,     50,   1,   10),
]


def compute_k(num_records, num_subs):
    k_by_size = max(2, num_records // 50)
    return max(num_subs, min(k_by_size, 15))


def embed_and_cluster(records, num_subs):
    texts = [
        " ".join(filter(None, [
            r.get("cleaned_text", ""),
            r.get("article_text", ""),
            r.get("top_comments", ""),
        ]))
        for r in records
    ]
    k = compute_k(len(records), num_subs)
    logging.info(f"Clustering: k={k}, records={len(records)}")

    results = []
    for (name, vsz, mc, ep, dm, win) in DOC2VEC_CONFIGS:
        logging.info(f"Training Doc2Vec config: {name}")
        tagged = [TaggedDocument(words=word_tokenize(t), tags=[str(i)])
                  for i, t in enumerate(texts)]
        model = Doc2Vec(
            vector_size=vsz, min_count=mc, epochs=ep, dm=dm,
            window=win, workers=os.cpu_count() or 4, seed=42,
            dbow_words=(1 if dm == 0 else 0),
        )
        model.build_vocab(tagged)
        model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(os.path.join(OUT_DIR, f"model_{name}.d2v"))

        # OPTIMIZATION: infer_vector epochs 30→10; stable enough for 3k docs
        emb    = np.array([model.infer_vector(word_tokenize(t), epochs=10) for t in texts])
        normed = normalize(emb)

        km   = KMeans(n_clusters=k, n_init=15, random_state=42, max_iter=400)
        lbls = km.fit_predict(normed)

        sil = silhouette_score(normed, lbls, metric="cosine")
        db  = davies_bouldin_score(normed, lbls)
        ch  = calinski_harabasz_score(normed, lbls)

        print(f"[{name}] vec={vsz} dm={dm} win={win} ep={ep}")
        print(f"  Silhouette={sil:.4f}  DB={db:.4f}  CH={ch:.2f}")
        unique, counts = np.unique(lbls, return_counts=True)
        print(f"  Cluster dist: {' '.join(f'C{c}:{n}' for c,n in zip(unique,counts))}")

        results.append(dict(
            name=name, embeddings=emb, labels=lbls,
            centroids=km.cluster_centers_,
            silhouette=sil, db=db, ch=ch,
        ))

    results.sort(key=lambda x: x["silhouette"], reverse=True)
    best = results[0]
    logging.info(f"Best config: {best['name']}  Silhouette={best['silhouette']:.4f}")

    # TF-IDF keywords per cluster (for assignment writeup / interpretability)
    vect   = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    tmat   = vect.fit_transform(texts)
    fnames = vect.get_feature_names_out()
    cluster_kw = {}
    for cid in range(k):
        idx = np.where(best["labels"] == cid)[0]
        if not len(idx):
            cluster_kw[cid] = []
            continue
        mt = np.asarray(tmat[idx].mean(axis=0)).flatten()
        cluster_kw[cid] = [fnames[i] for i in mt.argsort()[-8:][::-1]]

    # PCA visualization (2×3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, res in zip(axes.flatten(), results):
        red = PCA(2, random_state=42).fit_transform(normalize(res["embeddings"]))
        for cid in range(k):
            pts = red[res["labels"] == cid]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.55)
        ax.set_title(
            f"{res['name']}\nSil={res['silhouette']:.3f} DB={res['db']:.3f} CH={res['ch']:.0f}",
            fontsize=9,
        )
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("Doc2Vec - 6-Config Cluster Comparison (PCA)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "doc2vec_cluster_comparison.png"), dpi=150)
    plt.close()

    names = [r["name"] for r in results]
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(x, [r["silhouette"] for r in results], color="steelblue")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_title("Silhouette (higher is better)")
    ax2.bar(x, [r["db"] for r in results], color="salmon")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_title("Davies-Bouldin (lower is better)")
    fig.suptitle("Doc2Vec Metrics Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "doc2vec_metrics_bar.png"), dpi=130)
    plt.close()

    print("\n" + "=" * 70)
    print(f"{'Config':<22} {'Silhouette':>11} {'DB':>14} {'CH':>10}")
    print("-" * 70)
    for res in results:
        tag = " <- BEST" if res["name"] == best["name"] else ""
        print(f"{res['name']:<22} {res['silhouette']:>11.4f} {res['db']:>14.4f} {res['ch']:>10.2f}{tag}")
    print("=" * 70)

    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k not in ("embeddings", "labels", "centroids")}
             for r in results],
            f, indent=2,
        )

    be, bl, bc = best["embeddings"], best["labels"], best["centroids"]
    nb = normalize(be)
    for i, r in enumerate(records):
        cid = int(bl[i])
        r["embedding"]            = pickle.dumps(be[i])
        r["cluster_id"]           = cid
        r["keywords"]             = ", ".join(cluster_kw.get(cid, []))
        r["distance_to_centroid"] = float(np.linalg.norm(nb[i] - bc[cid]))

    return results


# ─── INTERACTIVE QUERY ───────────────────────────────────────────────────────

def interactive_query(conn):
    print("\nInteractive query - type 'exit' to quit.\n")
    cur = conn.cursor()
    cur.execute("SELECT embedding, cluster_id, title, keywords FROM doc2vec_posts WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    if not rows:
        print("No embedded posts found.")
        return
    embs = np.array([pickle.loads(r[0]) for r in rows])
    cids = np.array([r[1] for r in rows])
    kws  = [r[3] or "" for r in rows]
    centroids = {c: embs[cids == c].mean(axis=0) for c in sorted(set(cids))}
    while True:
        q = input("Keywords: ").strip()
        if q.lower() == "exit":
            break
        tokens = tokenize(clean_text(q))
        scores = np.array([sum(1 for t in tokens if t in kw) for kw in kws], dtype=float)
        vec    = embs[scores.argsort()[-20:]].mean(axis=0) if scores.max() > 0 else embs.mean(axis=0)
        best_c = min(centroids, key=lambda c: np.linalg.norm(vec - centroids[c]))
        print(f"\nCluster {best_c}:")
        cur.execute("SELECT title, keywords FROM doc2vec_posts WHERE cluster_id=%s LIMIT 8", (best_c,))
        for t, kw in cur.fetchall():
            print(f"  {t}\n    Keywords: {kw}")
        print()
    cur.close()


# ─── PIPELINE ────────────────────────────────────────────────────────────────

def run_pipeline(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    db = get_db_conn(args.db_host, args.db_user, args.db_pass)
    ensure_table(db)

    # ── Step 1: Acquire posts ──────────────────────────────────────────────
    if args.from_db:
        # FAST PATH: read posts already in DB (from doc2vec.py previous run)
        logging.info("Loading posts from DB (--from-db). Skipping scrape.")
        all_posts = load_posts_from_db(db)
        if not all_posts:
            logging.error("No posts in DB. Run without --from-db first.")
            return
    elif os.path.exists(CHECKPOINT_FILE) and not args.no_checkpoint:
        logging.info(f"Loading checkpoint from {CHECKPOINT_FILE}")
        with open(CHECKPOINT_FILE) as f:
            all_posts = json.load(f)
        logging.info(f"Checkpoint: {len(all_posts)} posts loaded.")
    else:
        all_posts = []
        for sub in args.subs:
            all_posts.extend(scrape_subreddit(sub, args.num))
        logging.info(f"Total posts scraped: {len(all_posts)}")
        # Save checkpoint so a crash doesn't lose scraping work
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(all_posts, f, default=str)
        logging.info(f"Checkpoint saved to {CHECKPOINT_FILE}")

    # ── Step 2: Optional comment enrichment ───────────────────────────────
    # DEFAULT: SKIPPED. Use --enrich to enable (adds hours for 3k posts).
    if args.enrich:
        logging.info("Comment enrichment enabled (this will take a while).")
        enrich_with_comments(all_posts)
    else:
        logging.info("Comment enrichment SKIPPED (use --enrich to enable). Using title+selftext+flair.")

    # ── Step 3: Optional async article fetch ──────────────────────────────
    article_texts = [""] * len(all_posts)
    if not args.skip_articles:
        urls = [p["post_url"] for p in all_posts]
        article_texts = asyncio.run(fetch_articles_async(urls))
    else:
        logging.info("Article fetch SKIPPED (--skip-articles).")

    # ── Step 4: Build records ─────────────────────────────────────────────
    records = []
    for p, art in tqdm(zip(all_posts, article_texts), total=len(all_posts), desc="Building records"):
        image_ocr = ocr_image(p.get("image_url")) if args.images else ""
        raw = " ".join(filter(None, [
            p.get("title", ""),
            p.get("selftext", ""),
            art,
            p.get("top_comments", ""),
            image_ocr,
            p.get("flair", ""),
        ]))
        cleaned = clean_text(raw)
        if len(tokenize(cleaned)) < 5:
            continue
        records.append({
            "reddit_id":            p["reddit_id"],
            "subreddit":            p["subreddit"],
            "title":                p.get("title", ""),
            "selftext":             p.get("selftext", ""),
            "image_url":            p.get("image_url"),
            "image_path":           None,
            "image_ocr_text":       image_ocr,
            "author_masked":        mask_author(p.get("author", "")),
            "created_utc":          p.get("created_utc", datetime.utcnow()),
            "score":                p.get("score", 0),
            "upvote_ratio":         p.get("upvote_ratio", 0.0),
            "num_comments":         p.get("num_comments", 0),
            "flair":                p.get("flair", ""),
            "post_url":             p.get("post_url", ""),
            "domain":               p.get("domain", ""),
            "article_text":         art,
            "top_comments":         p.get("top_comments", ""),
            "cleaned_text":         cleaned,
            "embedding":            None,
            "cluster_id":           None,
            "keywords":             None,
            "distance_to_centroid": None,
            "fetched_at":           datetime.utcnow(),
        })

    logging.info(f"Records after filtering: {len(records)}")
    if not records:
        logging.error("No records to process. Exiting.")
        return

    # ── Step 5: Embed + cluster ───────────────────────────────────────────
    embed_and_cluster(records, num_subs=len(args.subs))

    # ── Step 6: Batch DB insert ───────────────────────────────────────────
    batch_insert_posts(db, records)
    logging.info(f"Done. Outputs saved to ./{OUT_DIR}/")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Lab 8 Doc2Vec Pipeline (Optimized)")
    p.add_argument("--subs", nargs="+",
                   default=["technology", "technews", "tech", "cybersecurity", "netsec", "windowssecurity"])
    p.add_argument("--num",           type=int, default=500, help="Posts per subreddit")
    p.add_argument("--db-host",       required=True)
    p.add_argument("--db-user",       required=True)
    p.add_argument("--db-pass",       required=True)
    p.add_argument("--enrich",        action="store_true",
                   help="Fetch top comments for each post (slow - adds hours)")
    p.add_argument("--skip-articles", action="store_true",
                   help="Skip async external article fetch")
    p.add_argument("--from-db",       action="store_true",
                   help="Load already-scraped posts from DB instead of re-scraping")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Ignore existing checkpoint file")
    p.add_argument("--images",        action="store_true",
                   help="Enable OCR on post images (requires pytesseract)")
    p.add_argument("--query",         action="store_true",
                   help="Launch interactive cluster query after pipeline")
    p.add_argument("--interval",      type=int, default=0,
                   help="Re-run interval in minutes (0 = run once)")
    args = p.parse_args()

    if args.interval > 0:
        while True:
            run_pipeline(args)
            time.sleep(args.interval * 60)
    else:
        run_pipeline(args)
        if args.query:
            interactive_query(get_db_conn(args.db_host, args.db_user, args.db_pass))


if __name__ == "__main__":
    main()