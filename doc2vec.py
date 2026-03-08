#!/usr/bin/env python3
# DSCI-560 Lab 8 - Doc2Vec Embedding and Clustering Pipeline
# Scrapes old.reddit.com by following the "next" button.
# Enriches posts with comments and article text.
# Trains 6 Doc2Vec configs, clusters with cosine KMeans, saves outputs.

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
    return any(h == d or h.endswith("." + d) for d in blockedDom)



# OLD.REDDIT.COM HTML SCRAPER
# Loads old.reddit.com/r/<sub> pages and follows the "next" button until
# we have tried and collected at least max_posts unique posts.


def _reddit_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(usrAgents),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    # tell old.reddit to always serve the classic layout
    s.cookies.set("reddit_session", "", domain=".reddit.com")
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
    posts = []
    # each post is in a div.thing with data- attributes
    for thing in soup.select("div.thing"):
        try:
            # skip stickied posts
            if "stickied" in thing.get("class", []):
                continue
            if thing.get("data-promoted") == "true":
                continue

            reddit_id = thing.get("data-fullname", "")
            if not reddit_id:
                continue
            # strip t3_ prefix
            reddit_id = reddit_id.replace("t3_", "")

            title_tag = thing.select_one("a.title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            # the link the post points to
            post_url = thing.get("data-url", "")
            permalink = thing.get("data-permalink", "")

            author = thing.get("data-author", "")
            score_str = thing.get("data-score", "0") or "0"
            try:
                score = int(score_str)
            except ValueError:
                score = 0

            num_comments_str = thing.get("data-comments-count", "0") or "0"
            try:
                num_comments = int(num_comments_str)
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

            # check for image url
            image_url = None
            if imgEx.search(post_url):
                image_url = post_url

            # upvote ratio from data attribute (old reddit exposes it)
            ratio_str = thing.get("data-upvote-ratio", "0") or "0"
            try:
                upvote_ratio = float(ratio_str)
            except ValueError:
                upvote_ratio = 0.0

            posts.append({
                "reddit_id":    reddit_id,
                "subreddit":    subreddit,
                "title":        title,
                "selftext":     "",  # fetched separately for text posts
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
            logging.debug(f"Skipping thing: {e}")
            continue
    return posts


def _fetch_selftext(session: requests.Session, permalink: str) -> str:
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
    session = _reddit_session()
    posts = []
    seen = set()

    # use three feeds for variety as recommended by the assignment
    feeds = [
        f"https://old.reddit.com/r/{subreddit}/hot/",
        f"https://old.reddit.com/r/{subreddit}/new/",
        f"https://old.reddit.com/r/{subreddit}/top/?t=month",
    ]
    # allocate quota: hot gets the majority, new and top fill the rest
    quotas = [max_posts, max_posts // 3, max_posts // 3]

    for feed_url, quota in zip(feeds, quotas):
        collected = 0
        url = feed_url
        page_num = 0

        while collected < quota and len(posts) < max_posts:
            page_num += 1
            logging.info(f"r/{subreddit} | feed={feed_url.split('/')[-2] or 'top'} | page={page_num} | collected={len(posts)}")
            soup = _get_page(session, url)
            if not soup:
                logging.warning(f"No page returned for {url}, stopping feed.")
                break

            page_posts = _parse_old_reddit_page(soup, subreddit)
            if not page_posts:
                logging.info(f"No posts found on page {page_num}, feed exhausted.")
                break

            for p in page_posts:
                if p["reddit_id"] not in seen and len(posts) < max_posts:
                    seen.add(p["reddit_id"])
                    posts.append(p)
                    collected += 1

            # find the "next" button to get the next page
            next_btn = soup.select_one("span.next-button a")
            if not next_btn:
                logging.info(f"No next button found, feed exhausted at page {page_num}.")
                break
            url = next_btn["href"]

            # polite delay between pages - stay well under reddit's rate limit
            time.sleep(3 + random.uniform(1, 2))

        if len(posts) >= max_posts:
            break

    logging.info(f"r/{subreddit}: scraped {len(posts)} posts total")
    return posts[:max_posts]


def enrich_with_selftext_and_comments(posts: list) -> None:
    session = _reddit_session()
    for i, post in enumerate(tqdm(posts, desc="Enriching posts")):
        permalink = post.get("permalink", "")
        if not permalink:
            continue

        # fetch selftext for self posts
        domain = post.get("domain", "")
        if domain.startswith("self.") and not post.get("selftext"):
            post["selftext"] = _fetch_selftext(session, permalink)

        # fetch top comments from the post page
        post["top_comments"] = _fetch_comments_html(session, permalink)

        # sleep longer every 10 posts to avoid 429
        if (i + 1) % 10 == 0:
            time.sleep(5 + random.uniform(1, 3))
        else:
            time.sleep(2 + random.uniform(0.5, 1.5))


# Fetches external article text from post URLs concurrently.

async def _fetch_article(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> str:
    if is_blocked(url):
        return ""
    async with sem:
        try:
            async with session.get(
                url,
                headers=brHead,
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
    sem       = asyncio.Semaphore(conn)
    connector = aiohttp.TCPConnector(limit=conn + 5, ssl=False, ttl_dns_cache=300)
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



# Db - lab8; doc2vec_posts


def get_db_conn(host, user, password, database="lab8"):
    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
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
    logging.info("Tb doc2vec_posts")


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



# Text Utilities


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
        if w.isalpha() and w not in Swords and len(w) > 2
    ]


def mask_author(a: str) -> str:
    return "user_unknown" if not a else "user_" + str(abs(hash(a)) % 10**8)


def ocr_image(url: str) -> str:
    if not url:
        return ""
    try:
        import pytesseract
        r   = requests.get(url, headers=brHead, timeout=10)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if min(img.size) < 300:
            s = 300 / min(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""



# Doc2Vec - 6 configs
# Trying with (3 DBOW + 3 DM) to have thorough idea and comparison.

doc2vecConf = [
    # (name,            vec_size, min_count, epochs, dm,  window)
    ("dbow_50_fast",         50,         2,     20,   0,    5),
    ("dbow_100_std",        100,         2,     30,   0,    5),
    ("dbow_200_deep",       200,         2,     50,   0,    8),
    ("dm_100_std",          100,         2,     30,   1,    5),
    ("dm_200_deep",         200,         2,     50,   1,    8),
    ("dm_300_large",        300,         1,     60,   1,   10),
]


def compute_k(num_records: int, num_subs: int) -> int:
    k_by_size = max(2, num_records // 50)
    k = max(num_subs, min(k_by_size, 15))
    return k


def embed_and_cluster(records: list, num_subs: int) -> list:
    texts = [
        " ".join(filter(None, [
            r["cleaned_text"],
            r.get("article_text", ""),
            r.get("top_comments", ""),
            r.get("image_ocr_text", ""),
        ]))
        for r in records
    ]
    k = compute_k(len(records), num_subs)
    logging.info(f"Clustering: k={k}, records={len(records)}")

    results = []
    for (name, vsz, mc, ep, dm, win) in doc2vecConf:
        logging.info(f"Training Doc2Vec config: {name}")
        tagged = [
            TaggedDocument(words=word_tokenize(t), tags=[str(i)])
            for i, t in enumerate(texts)
        ]
        model = Doc2Vec(
            vector_size=vsz,
            min_count=mc,
            epochs=ep,
            dm=dm,
            window=win,
            workers=4,
            seed=42,
            dbow_words=(1 if dm == 0 else 0),
        )
        model.build_vocab(tagged)
        model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(os.path.join(outDir, f"model_{name}.d2v"))

        # infer vectors with more epochs for stability
        emb    = np.array([model.infer_vector(word_tokenize(t), epochs=30) for t in texts])
        normed = normalize(emb)

        # KMeans with cosine distance and normalized distnces
        km   = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
        lbls = km.fit_predict(normed)

        sil = silhouette_score(normed, lbls, metric="cosine")
        db  = davies_bouldin_score(normed, lbls)
        ch  = calinski_harabasz_score(normed, lbls)

        print(f"[{name}] vec={vsz} dm={dm} win={win} ep={ep}")
        print(f"  Silhouette={sil:.4f}  DB={db:.4f}  CH={ch:.2f}")

        # per-cluster post counts for interpretability
        unique, counts = np.unique(lbls, return_counts=True)
        dist_str = " ".join(f"C{c}:{n}" for c, n in zip(unique, counts))
        print(f"  Cluster distribution: {dist_str}")

        results.append(dict(
            name=name, embeddings=emb, labels=lbls,
            centroids=km.cluster_centers_,
            silhouette=sil, db=db, ch=ch,
        ))

    # sort by silhouette descending
    results.sort(key=lambda x: x["silhouette"], reverse=True)
    best = results[0]
    logging.info(f"Best config: {best['name']}  Silhouette={best['silhouette']:.4f}")

    #TF-IDF keywords 
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

    #PCA visualization
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
    fig.suptitle("Doc2Vec - 6-Config Cluster Comparison (PCA)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outDir, "doc2vec_cluster_comparison.png"), dpi=150)
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
    fig.suptitle("Doc2Vec Metrics Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(outDir, "doc2vec_metrics_bar.png"), dpi=130)
    plt.close()

    # print summary table
    print(f"{'Config':<22} {'Silhouette':>11} {'DB':>14} {'CH':>10}")
    print("-" * 70)
    for res in results:
        tag = " <- BEST" if res["name"] == best["name"] else ""
        print(
            f"{res['name']:<22} {res['silhouette']:>11.4f} "
            f"{res['db']:>14.4f} {res['ch']:>10.2f}{tag}"
        )
    print("=" * 70)

    # save metrics to JSON
    with open(os.path.join(outDir, "metrics_summary.json"), "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k not in ("embeddings", "labels", "centroids")}
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



# Interactive Query - sae from prev assignment


def interactive_query(conn):
    print("\nInteractive query - type 'exit' to quit.\n")
    cur = conn.cursor()
    cur.execute(
        "SELECT embedding, cluster_id, title, keywords "
        "FROM doc2vec_posts WHERE embedding IS NOT NULL"
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
            "SELECT title, keywords FROM doc2vec_posts WHERE cluster_id=%s LIMIT 8",
            (best_c,),
        )
        for t, kw in cur.fetchall():
            print(f"  {t}")
            print(f"    Keywords: {kw}")
        print()
    cur.close()



#Pipeline


def run_pipeline(args):
    os.makedirs(outDir, exist_ok=True)
    conn = get_db_conn(args.db_host, args.db_user, args.db_pass)
    ensure_table(conn)

    #scrape old.reddit by following next button
    all_posts = []
    for sub in args.subs:
        sub_posts = scrape_subreddit_old_reddit(sub, args.num)
        all_posts.extend(sub_posts)
    logging.info(f"Total posts scraped: {len(all_posts)}")

    #enrich with selftext (for self posts) and comments
    enrich_with_selftext_and_comments(all_posts)

    # step 3: async article enrichment for external link posts
    urls          = [p["post_url"] for p in all_posts]
    article_texts = asyncio.run(fetch_articles_async(urls))

    #build records, clean text, filter short docs
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

    #embed and cluster with 6 Doc2Vec configs
    embed_and_cluster(records, num_subs=len(args.subs))

    #insert to DB
    for r in tqdm(records, desc="Inserting to DB"):
        insert_post(conn, r)

    logging.info(f"Done. Outputs saved to ./{outDir}/")



#cli


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subs", nargs="+",
        default=["technology", "technews", "tech", "cybersecurity", "netsec", "windowssecurity"],
    )
    p.add_argument("--num",                type=int, default=500,
                   help="Number of posts to collect per subreddit")
    p.add_argument("--db-host",            required=True)
    p.add_argument("--db-user",            required=True)
    p.add_argument("--db-pass",            required=True)
    p.add_argument("--interval",           type=int, default=0,
                   help="Re-run interval in minutes (0 = run once)")
    p.add_argument("--images",             action="store_true",
                   help="Enable OCR on post images (requires pytesseract)")
    p.add_argument("--query",              action="store_true",
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