#!/usr/bin/env python3
# DSCI-560 Lab 8 - Word2Vec Bag-of-Words Embedding and Clustering Pipeline
# Scrapes old.reddit.com by following the "next" button.
# Enriches posts with comments and article text.
# Trains Word2Vec, clusters words into K bins, builds BoW document vectors,
# then clusters documents. Runs 6 configurations as required by assignment.

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
from urllib.parse import urljoin, urlparse
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

Swords   = set(stopwords.words("english"))
outDir  = "doc2vec_outputs"
conn = 25   # async article fetch workers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


usrAgents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

brHead = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

imgEx = re.compile(r"\.(jpg|jpeg|png|gif|webp)(\?.*)?$", re.I)

# domains we skip immediately during article enrichment - paywalls, social, etc.
blockedDom = {
    "bloomberg.com", "wsj.com", "nytimes.com", "ft.com", "thetimes.co.uk",
    "economist.com", "barrons.com", "seekingalpha.com", "hbr.org",
    "washingtonpost.com", "businessinsider.com", "theatlantic.com",
    "newyorker.com", "medium.com", "wired.com", "forbes.com",
    "reuters.com", "techcrunch.com",
    "youtube.com", "youtu.be", "twitch.tv", "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "facebook.com",
    "reddit.com", "redd.it", "i.redd.it", "v.redd.it", "gallery.reddit.com",
}


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



# OLD.REDDIT.COM HTML SCRAPER
# Loads listing pages and follows "next" button until max_posts collected.


def _reddit_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    s.cookies.set("over18", "1", domain=".old.reddit.com")
    return s


def _get_page(session: requests.Session, url: str, retries: int = 5) -> BeautifulSoup | None:
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


def _parse_old_reddit_page(soup: BeautifulSoup, subreddit: str) -> list:
    """Extract post dicts from one old.reddit listing page."""
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
            title = title_tag.get_text(strip=True) if title_tag else ""

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
            created_utc = datetime.utcnow()
            if timestamp_tag and timestamp_tag.get("datetime"):
                try:
                    created_utc = datetime.strptime(
                        timestamp_tag["datetime"], "%Y-%m-%dT%H:%M:%S+00:00"
                    )
                except Exception:
                    pass

            domain_tag = thing.select_one("span.domain a")
            domain = domain_tag.get_text(strip=True) if domain_tag else ""

            flair_tag = thing.select_one("span.linkflairlabel")
            flair = flair_tag.get_text(strip=True) if flair_tag else ""

            image_url = None
            if IMAGE_EXTS.search(post_url):
                image_url = post_url

            try:
                upvote_ratio = float(thing.get("data-upvote-ratio", "0") or "0")
            except ValueError:
                upvote_ratio = 0.0

            posts.append({
                "reddit_id":    reddit_id,
                "subreddit":    subreddit,
                "title":        title,
                "selftext":     "",
                "image_url":    image_url,
                "author":       author,
                "created_utc":  created_utc,
                "score":        score,
                "upvote_ratio": upvote_ratio,
                "num_comments": num_comments,
                "flair":        flair,
                "post_url":     post_url,
                "domain":       domain,
                "permalink":    permalink,
                "top_comments": "",
            })
        except Exception as e:
            logging.debug(f"Skipping post: {e}")
            continue
    return posts


def _fetch_selftext(session: requests.Session, permalink: str) -> str:
    """Fetch selftext for self posts from old.reddit post page."""
    if not permalink:
        return ""
    url = f"https://old.reddit.com{permalink}"
    soup = _get_page(session, url)
    if not soup:
        return ""
    try:
        body_div = soup.select_one("div.usertext-body div.md")
        if body_div:
            return body_div.get_text(separator=" ", strip=True)
    except Exception:
        pass
    return ""


def _fetch_comments_html(session: requests.Session, permalink: str, limit: int = 12) -> str:
    """Fetch top-level comments from old.reddit post page."""
    if not permalink:
        return ""
    url = f"https://old.reddit.com{permalink}?limit=25&depth=1"
    soup = _get_page(session, url)
    if not soup:
        return ""
    texts = []
    try:
        for comment_div in soup.select("div.commentarea div.entry div.usertext-body div.md"):
            text = comment_div.get_text(separator=" ", strip=True)
            if 10 < len(text) < 600:
                texts.append(text)
            if len(texts) >= limit:
                break
    except Exception:
        pass
    return " ".join(texts)


def scrape_subreddit_old_reddit(subreddit: str, max_posts: int) -> list:
    """
    Scrape exactly max_posts from old.reddit.com/r/<subreddit>.
    Follows the next button. Uses hot, new, and top(month) feeds for variety.
    """
    session = _reddit_session()
    posts   = []
    seen    = set()

    feeds  = [
        f"https://old.reddit.com/r/{subreddit}/hot/",
        f"https://old.reddit.com/r/{subreddit}/new/",
        f"https://old.reddit.com/r/{subreddit}/top/?t=month",
    ]
    quotas = [max_posts, max_posts // 3, max_posts // 3]

    for feed_url, quota in zip(feeds, quotas):
        collected = 0
        url       = feed_url
        page_num  = 0

        while collected < quota and len(posts) < max_posts:
            page_num += 1
            feed_name = feed_url.rstrip("/").split("/")[-1].split("?")[0]
            logging.info(
                f"r/{subreddit} | feed={feed_name} | page={page_num} | total={len(posts)}"
            )
            soup = _get_page(session, url)
            if not soup:
                logging.warning(f"No page returned for {url}, stopping feed.")
                break

            page_posts = _parse_old_reddit_page(soup, subreddit)
            if not page_posts:
                logging.info(f"No posts on page {page_num}, feed exhausted.")
                break

            for p in page_posts:
                if p["reddit_id"] not in seen and len(posts) < max_posts:
                    seen.add(p["reddit_id"])
                    posts.append(p)
                    collected += 1

            next_btn = soup.select_one("span.next-button a")
            if not next_btn:
                logging.info(f"No next button at page {page_num}, feed done.")
                break
            url = next_btn["href"]

            time.sleep(3 + random.uniform(1, 2))

        if len(posts) >= max_posts:
            break

    logging.info(f"r/{subreddit}: collected {len(posts)} posts")
    return posts[:max_posts]


def enrich_with_selftext_and_comments(posts: list) -> None:
    """
    For self posts, fetch the selftext body.
    For all posts, fetch top comments.
    Polite delays are used to avoid 429 responses.
    """
    session = _reddit_session()
    for i, post in enumerate(tqdm(posts, desc="Enriching posts")):
        permalink = post.get("permalink", "")
        if not permalink:
            continue

        # fetch selftext for text-only posts
        domain = post.get("domain", "")
        if domain.startswith("self.") and not post.get("selftext"):
            post["selftext"] = _fetch_selftext(session, permalink)

        # fetch top comments
        post["top_comments"] = _fetch_comments_html(session, permalink)

        # longer pause every 10 posts
        if (i + 1) % 10 == 0:
            time.sleep(5 + random.uniform(1, 3))
        else:
            time.sleep(2 + random.uniform(0.5, 1.5))



# ASYNC ARTICLE ENRICHMENT


async def _fetch_article(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> str:
    if is_blocked(url):
        return ""
    async with sem:
        try:
            async with session.get(
                url,
                headers=BROWSER_HEADERS,
                timeout=aiohttp.ClientTimeout(total=10),
                ssl=False,
                allow_redirects=True,
                max_redirects=4,
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
    sem       = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 5, ssl=False, ttl_dns_cache=300)
    results   = [""] * len(urls)
    async with aiohttp.ClientSession(connector=connector) as session:
        futures = {
            asyncio.ensure_future(_fetch_article(session, u, sem)): i
            for i, u in enumerate(urls)
        }
        pbar = tqdm(total=len(futures), desc="Fetching articles", unit="url")
        for fut in asyncio.as_completed(futures):
            results[futures[fut]] = await fut
            pbar.update(1)
        pbar.close()
    return results



# DATABASE - lab8 | word2vec_posts


def get_db_conn(host, user, password, database="lab8"):
    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    conn.autocommit = True
    logging.info(f"Connected to '{database}' on {host}")
    return conn


def ensure_table(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS word2vec_posts (
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
    CREATE INDEX IF NOT EXISTS idx_w2v_cluster ON word2vec_posts(cluster_id);
    CREATE INDEX IF NOT EXISTS idx_w2v_dist    ON word2vec_posts(distance_to_centroid);
    """)
    cur.close()
    logging.info("Table word2vec_posts ready.")


def insert_post(conn, r):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO word2vec_posts
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
        r["reddit_id"],    r["subreddit"],    r["title"],
        r["selftext"],     r["image_url"],    r.get("image_path"),
        r["image_ocr_text"], r["author_masked"], r["created_utc"],
        r["score"],        r["upvote_ratio"], r["num_comments"],
        r["flair"],        r["post_url"],     r["domain"],
        r["article_text"], r["top_comments"], r["cleaned_text"],
        r["embedding"],    r["cluster_id"],   r["keywords"],
        r["distance_to_centroid"], r["fetched_at"],
    ))
    cur.close()


def create_distance_index(conn):
    cur = conn.cursor()
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_w2v_dist2 ON word2vec_posts(distance_to_centroid);"
    )
    conn.commit()
    cur.close()



# TEXT UTILITIES


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"http\S+",         " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]",  " ", text)
    text = re.sub(r"\b\d+\b",         " ", text)
    text = re.sub(r"\s+",             " ", text).strip()
    return text.lower()


def tokenize(text: str) -> list:
    return [
        w for w in word_tokenize(text.lower())
        if w.isalpha() and w not in STOPWORDS and len(w) > 2
    ]


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
            img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""



# WORD2VEC BOW - SIX CONFIGURATIONS
#
# Assignment (Section 2) requires:
#   1. Train Word2Vec on all corpus tokens
#   2. Cluster word vectors into K semantic bins using KMeans
#   3. Represent each document as a K-dim normalized word-bin frequency vector
#   4. Cluster document vectors
#
# We run 6 configurations varying vector_size and num_bins.
# The same 3 vector dimensions used in Doc2Vec are mirrored here.

W2V_CONFIGS = [
    # (name,               vec_size, num_bins, window, min_count, epochs)
    ("w2v_sz50_bins50",        50,       50,       5,         2,      10),
    ("w2v_sz100_bins100",     100,      100,       5,         2,      10),
    ("w2v_sz100_bins200",     100,      200,       5,         2,      10),
    ("w2v_sz200_bins100",     200,      100,       8,         2,      12),
    ("w2v_sz200_bins200",     200,      200,       8,         2,      12),
    ("w2v_sz300_bins300",     300,      300,      10,         1,      15),
]


def build_bow_embeddings(tokenized_docs: list, w2v: Word2Vec, num_bins: int) -> np.ndarray:
    """
    Step 2-3 of the assignment Word2Vec BoW method:
    - Cluster all word vectors into num_bins semantic bins.
    - For each document, count how many words fall in each bin,
      then normalize by document length.
    """
    vocab        = w2v.wv.index_to_key
    word_vectors = np.array([w2v.wv[w] for w in vocab])

    # cluster word vectors into bins
    word_km     = KMeans(n_clusters=num_bins, n_init=10, random_state=42, max_iter=300)
    word_labels = word_km.fit_predict(word_vectors)
    word2bin    = {vocab[i]: int(word_labels[i]) for i in range(len(vocab))}

    doc_vecs = []
    for doc in tokenized_docs:
        vec   = np.zeros(num_bins, dtype=np.float32)
        valid = [w for w in doc if w in word2bin]
        if valid:
            for w in valid:
                vec[word2bin[w]] += 1
            # normalize by document word count
            vec /= len(valid)
        doc_vecs.append(vec)
    return np.array(doc_vecs)


def compute_k(num_records: int, num_subs: int) -> int:
    """
    k >= number of subreddits (each subreddit is a distinct topic domain).
    k is capped at 15 to avoid over-fragmentation.
    """
    k_by_size = max(2, num_records // 50)
    k = max(num_subs, min(k_by_size, 15))
    return k


def embed_and_cluster_w2v(records: list, num_subs: int) -> list:
    texts = [
        " ".join(filter(None, [
            r["cleaned_text"],
            r.get("article_text", ""),
            r.get("top_comments", ""),
            r.get("image_ocr_text", ""),
        ]))
        for r in records
    ]
    tokenized_docs = [tokenize(t) for t in texts]
    k = compute_k(len(records), num_subs)
    logging.info(f"Clustering: k={k}, records={len(records)}")

    results = []
    for (name, vsz, num_bins, win, mc, ep) in W2V_CONFIGS:
        logging.info(f"Training Word2Vec config: {name}")

        # step 1: train Word2Vec on all tokens
        w2v = Word2Vec(
            sentences=tokenized_docs,
            vector_size=vsz,
            window=win,
            min_count=mc,
            workers=4,
            seed=42,
            epochs=ep,
        )
        w2v.save(os.path.join(OUTPUT_DIR, f"word2vec_model_{name}.model"))

        # steps 2+3: cluster words into bins, then build doc BoW vectors
        emb    = build_bow_embeddings(tokenized_docs, w2v, num_bins)
        normed = normalize(emb)

        # step 4: cluster document vectors
        km   = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
        lbls = km.fit_predict(normed)

        sil = silhouette_score(normed, lbls, metric="cosine")
        db  = davies_bouldin_score(normed, lbls)
        ch  = calinski_harabasz_score(normed, lbls)

        print(f"[{name}] vec={vsz} bins={num_bins} win={win} ep={ep}")
        print(f"  Silhouette={sil:.4f}  DB={db:.4f}  CH={ch:.2f}")

        unique, counts = np.unique(lbls, return_counts=True)
        dist_str = " ".join(f"C{c}:{n}" for c, n in zip(unique, counts))
        print(f"  Cluster distribution: {dist_str}")

        results.append(dict(
            name=name, embeddings=emb, labels=lbls,
            centroids=km.cluster_centers_, w2v=w2v,
            num_bins=num_bins, silhouette=sil, db=db, ch=ch,
        ))

    results.sort(key=lambda x: x["silhouette"], reverse=True)
    best = results[0]
    best["w2v"].save(os.path.join(OUTPUT_DIR, "word2vec_model_BEST.model"))
    logging.info(f"Best config: {best['name']}  Silhouette={best['silhouette']:.4f}")

    # TF-IDF keywords per cluster
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

    # 6-panel PCA visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, res in zip(axes.flatten(), results):
        red = PCA(2, random_state=42).fit_transform(normalize(res["embeddings"]))
        for cid in range(k):
            pts = red[res["labels"] == cid]
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.55)
        ax.set_title(
            f"{res['name']}\nSil={res['silhouette']:.3f} DB={res['db']:.3f} CH={res['ch']:.0f}",
            fontsize=9,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle("Word2Vec BoW - 6-Config Cluster Comparison (PCA)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "word2vec_cluster_comparison.png"), dpi=150)
    plt.close()

    # metrics bar charts
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(x, [r["silhouette"] for r in results], color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_title("Silhouette (higher is better)")
    ax2.bar(x, [r["db"] for r in results], color="salmon")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_title("Davies-Bouldin (lower is better)")
    fig.suptitle("Word2Vec BoW - Metrics Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "word2vec_metrics_bar.png"), dpi=130)
    plt.close()

    # print summary table
    print("\n" + "=" * 70)
    print(f"{'Config':<28} {'Silhouette':>10} {'DB':>14} {'CH':>10}")
    print("-" * 70)
    for res in results:
        tag = " <- BEST" if res["name"] == best["name"] else ""
        print(
            f"{res['name']:<28} {res['silhouette']:>10.4f} "
            f"{res['db']:>14.4f} {res['ch']:>10.2f}{tag}"
        )
    print("=" * 70)

    # save metrics to JSON
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(
            [{k: v for k, v in r.items()
              if k not in ("embeddings", "labels", "centroids", "w2v")}
             for r in results],
            f, indent=2,
        )

    # write best embeddings back into records
    be = best["embeddings"]
    bl = best["labels"]
    bc = best["centroids"]
    nb = normalize(be)
    for i, r in enumerate(records):
        cid = int(bl[i])
        r["embedding"]            = pickle.dumps(be[i])
        r["cluster_id"]           = cid
        r["keywords"]             = ", ".join(cluster_kw.get(cid, []))
        r["distance_to_centroid"] = float(np.linalg.norm(nb[i] - bc[cid]))

    return results



# INTERACTIVE QUERY


def interactive_query(conn):
    print("\nInteractive query - type 'exit' to quit.\n")
    cur = conn.cursor()
    cur.execute(
        "SELECT embedding, cluster_id, title, keywords "
        "FROM word2vec_posts WHERE embedding IS NOT NULL"
    )
    rows = cur.fetchall()
    if not rows:
        print("No embedded posts found.")
        return

    embs, cids, kws = [], [], []
    for row in rows:
        embs.append(pickle.loads(row[0]))
        cids.append(row[1])
        kws.append(row[3] or "")

    embs = np.array(embs)
    cids = np.array(cids)
    centroids = {c: embs[cids == c].mean(axis=0) for c in sorted(set(cids))}

    while True:
        q = input("Keywords: ").strip()
        if q.lower() == "exit":
            break
        tokens = tokenize(clean_text(q))
        scores = np.array([sum(1 for t in tokens if t in kw) for kw in kws], dtype=float)
        vec    = (
            embs[scores.argsort()[-20:]].mean(axis=0)
            if scores.max() > 0
            else embs.mean(axis=0)
        )
        best_c = min(centroids, key=lambda c: np.linalg.norm(vec - centroids[c]))
        print(f"\nCluster {best_c}:")
        cur.execute(
            "SELECT title, keywords FROM word2vec_posts WHERE cluster_id=%s LIMIT 8",
            (best_c,),
        )
        for t, kw in cur.fetchall():
            print(f"  {t}")
            print(f"    Keywords: {kw}")
        print()
    cur.close()



#Pipeline


def run_pipeline(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = get_db_conn(args.db_host, args.db_user, args.db_pass)
    ensure_table(conn)

    #scrape old.reddit by following next button
    all_posts = []
    for sub in args.subs:
        sub_posts = scrape_subreddit_old_reddit(sub, args.num)
        all_posts.extend(sub_posts)
    logging.info(f"Total posts scraped: {len(all_posts)}")

    #enrich with selftext and comments
    enrich_with_selftext_and_comments(all_posts)

    #async article enrichment for external links
    urls          = [p["post_url"] for p in all_posts]
    article_texts = asyncio.run(fetch_articles_async(urls))

    #build records
    records = []
    for p, art in tqdm(zip(all_posts, article_texts), total=len(all_posts), desc="Building records"):
        image_ocr = ocr_image(p["image_url"]) if args.images else ""
        raw = " ".join(filter(None, [
            p["title"],
            p["selftext"],
            art,
            p["top_comments"],
            image_ocr,
            p.get("flair", ""),
        ]))
        cleaned = clean_text(raw)
        if len(tokenize(cleaned)) < 5:
            continue
        records.append({
            "reddit_id":            p["reddit_id"],
            "subreddit":            p["subreddit"],
            "title":                p["title"],
            "selftext":             p["selftext"],
            "image_url":            p.get("image_url"),
            "image_path":           None,
            "image_ocr_text":       image_ocr,
            "author_masked":        mask_author(p["author"]),
            "created_utc":          p["created_utc"],
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

    logging.info(f"Records after filtering short docs: {len(records)}")

    if not records:
        logging.error("No records to process. Exiting.")
        return

    #Word2Vec BoW embedding and clustering
    embed_and_cluster_w2v(records, num_subs=len(args.subs))

    #insert to DB
    for r in tqdm(records, desc="Inserting to DB"):
        insert_post(conn, r)
    create_distance_index(conn)

    logging.info(f"Done. Outputs saved to ./{OUTPUT_DIR}/")



#cli


def main():
    p = argparse.ArgumentParser(
        description="Lab 8 Word2Vec BoW - old.reddit scraper + Word2Vec clustering"
    )
    p.add_argument(
        "--subs", nargs="+",
        default=["technology", "technews", "tech", "netsec", "windowssecurity", "cybersecurity"],
    )
    p.add_argument("--num",      type=int, default=500,
                   help="Number of posts to collect per subreddit")
    p.add_argument("--db-host",  required=True)
    p.add_argument("--db-user",  required=True)
    p.add_argument("--db-pass",  required=True)
    p.add_argument("--interval", type=int, default=0,
                   help="Re-run interval in minutes (0 = run once)")
    p.add_argument("--images",   action="store_true",
                   help="Enable OCR on post images (requires pytesseract)")
    p.add_argument("--query",    action="store_true",
                   help="Launch interactive cluster query after pipeline")
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