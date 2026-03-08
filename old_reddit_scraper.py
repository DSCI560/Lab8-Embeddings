#!/usr/bin/env python3

import argparse
from tokenize import tokenize
import requests
from bs4 import BeautifulSoup
import time
import re
import os
import psycopg2
import pickle
import logging
import pytesseract
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import math
import logging
from collections import defaultdict
from PIL import Image
from io import BytesIO
from datetime import datetime
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk

# One-time downloads (safe if already downloaded)
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


STOPWORDS = set(stopwords.words('english'))
HEADERS = {'User-Agent': 'DSCI560_Lab5_Scraper (Educational Project)'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# DATABASE

def get_db_conn(host, user, password, database='lab5_reddit'):
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    conn.autocommit = True
    return conn


def insert_post(conn, record):
    cursor = conn.cursor()

    sql = """
    INSERT INTO posts
    (reddit_id, subreddit, title, body, image_url, image_path,
     image_ocr_text, author_masked, created_utc, raw_html,
     cleaned_text, embedding, cluster_id)
    VALUES (%s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s)
    ON CONFLICT (reddit_id) DO NOTHING
    """

    cursor.execute(sql, (
        record['reddit_id'],
        record['subreddit'],
        record['title'],
        record['body'],
        record['image_url'],
        record['image_path'],
        record['image_ocr_text'],
        record['author_masked'],
        record['created_utc'],
        record['raw_html'],
        record['cleaned_text'],
        record['embedding'],
        record['cluster_id']
    ))

    cursor.close()



#Scraping old reddit

def safe_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r
    except Exception as e:
        logging.warning(f"Request failed: {e}")
        return None


def is_promoted(div):
    if div.get("data-promoted") == "true":
        return True
    classes = div.get("class") or []
    if "promoted" in " ".join(classes).lower():
        return True
    return False


def clean_text(text):
    if not text:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def mask_author(author):
    if not author:
        return "user_unknown"
    return "user_" + str(abs(hash(author)) % (10**8))


# Reusable tokenizer function 
def tokenize_text(text):
    return tokenize(text)  # replace with your existing tokenize function


def scrape_subreddit(subreddit, max_posts):
    url = f"https://old.reddit.com/r/{subreddit}/"
    posts = []

    while url and len(posts) < max_posts:
        logging.info(f"Scraping {url}")
        response = safe_get(url)
        if response is None:
            break

        soup = BeautifulSoup(response.text, "html.parser")
        things = soup.find_all("div", class_=lambda x: x and "thing" in x)

        for div in things:
            if len(posts) >= max_posts:
                break

            if is_promoted(div):
                continue

            reddit_id = div.get("data-fullname")
            if not reddit_id:
                continue

            title_tag = div.find("a", class_="title")
            title = title_tag.text.strip() if title_tag else ""

            body_html = ""
            body_div = div.find("div", class_="usertext-body")
            if body_div:
                body_html = str(body_div)

            author = div.get("data-author")
            created = datetime.utcnow()

            image_url = None
            thumb = div.find("a", class_="thumbnail")
            if thumb and thumb.get("href"):
                href = thumb["href"]
                if re.search(r"\.(jpg|jpeg|png|gif)", href, re.I):
                    image_url = href

            posts.append({
                "reddit_id": reddit_id,
                "subreddit": subreddit,
                "title": title,
                "body_html": body_html,
                "author": author,
                "created": created,
                "image_url": image_url,
                "raw_html": str(div)
            })

        next_btn = soup.find("span", class_="next-button")
        if next_btn and next_btn.find("a"):
            url = next_btn.find("a")["href"]
            time.sleep(2)
        else:
            break

    return posts

# IMAGE OCR

def ocr_image(url):
    try:
        r = safe_get(url)
        if not r:
            return ""
        img = Image.open(BytesIO(r.content))
        return pytesseract.image_to_string(img)
    except:
        return ""

# embeddings and clustering

def embed_and_cluster(records):
    """
    Runs THREE Doc2Vec configurations,
    evaluates with cosine-based clustering metrics,
    and saves the BEST config's embeddings + cluster_id into records.
    """

    configs = [
        {"name": "dbow_100", "vector_size": 100, "min_count": 2, "epochs": 20, "dm": 0},
        {"name": "dm_200",   "vector_size": 200, "min_count": 2, "epochs": 40, "dm": 1},
        {"name": "dbow_50",  "vector_size": 50,  "min_count": 1, "epochs": 30, "dm": 0},
    ]

    texts = [r["cleaned_text"] + " " + r["image_ocr_text"] for r in records]

    k = min(10, max(2, len(records)//50))

    best_result = None
    best_score = -999

    for config in configs:
        logging.info(f"Running Doc2Vec config: {config['name']}")

        tagged = [TaggedDocument(words=word_tokenize(t), tags=[str(i)])
                  for i, t in enumerate(texts)]

        model = Doc2Vec(
            vector_size=config["vector_size"],
            min_count=config["min_count"],
            epochs=config["epochs"],
            dm=config["dm"],
            workers=4,
            seed=42
        )

        model.build_vocab(tagged)
        model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)

        embeddings = np.array([
            model.infer_vector(word_tokenize(t), epochs=20)
            for t in texts
        ])

        # cosine clustering
        normed = normalize(embeddings)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(normed)

        # Evaluate with cosine-based metrics
        silhouette = silhouette_score(normed, labels, metric='cosine')
        db_index = davies_bouldin_score(normed, labels)
        ch_score = calinski_harabasz_score(normed, labels)

        print(f"\nConfig: {config['name']}")
        print("Silhouette (cosine):", round(silhouette, 4))
        print("Davies-Bouldin:", round(db_index, 4))
        print("Calinski-Harabasz:", round(ch_score, 2))

        # Choose best by silhouette (cosine)
        if silhouette > best_score:
            best_score = silhouette
            best_result = (embeddings, labels)

    # Save best embeddings and cluster IDs back to records
    best_embeddings, best_labels = best_result

    for i, r in enumerate(records):
        r["embedding"] = pickle.dumps(best_embeddings[i])
        r["cluster_id"] = int(best_labels[i])

    logging.info("Best configuration selected based on highest cosine silhouette score.")


# pipeline

def run_pipeline(args):
    conn = get_db_conn(args.db_host, args.db_user, args.db_pass)

    all_posts = []

    for sub in args.subs:
        raw_posts = scrape_subreddit(sub, args.num)
        all_posts.extend(raw_posts)

    records = []

    for p in tqdm(all_posts):
        cleaned = clean_text(p["title"] + " " + p["body_html"])
        image_ocr = ocr_image(p["image_url"]) if args.images and p["image_url"] else ""

        records.append({
            "reddit_id": p["reddit_id"],
            "subreddit": p["subreddit"],
            "title": p["title"],
            "body": BeautifulSoup(p["body_html"], "html.parser").get_text(),
            "image_url": p["image_url"],
            "image_path": None,
            "image_ocr_text": image_ocr,
            "author_masked": mask_author(p["author"]),
            "created_utc": p["created"],
            "raw_html": p["raw_html"],
            "cleaned_text": cleaned,
            "embedding": None,
            "cluster_id": None
        })

    if records:
        embed_and_cluster(records)

        for r in records:
            insert_post(conn, r)

    logging.info("Pipeline completed.")


# CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subs", nargs="+", default=["technology", "technews", "tech", "cybersecurity", "netsec", "windowssecurity"])
    parser.add_argument("--num", type=int, default=500)
    parser.add_argument("--db-host", required=True)
    parser.add_argument("--db-user", required=True)
    parser.add_argument("--db-pass", required=True)
    parser.add_argument("--interval", type=int, default=0)
    parser.add_argument("--images", action="store_true")

    args = parser.parse_args()

    if args.interval > 0:
        while True:
            run_pipeline(args)
            time.sleep(args.interval * 60)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
