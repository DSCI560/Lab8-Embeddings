#!/usr/bin/env python3

import argparse
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
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
from datetime import datetime
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
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

#db

def get_db_conn(host, user, password, database='lab5_reddit'):
    print(f"[DB] Attempting to connect to database '{database}' on host '{host}' as user '{user}'..")
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    conn.autocommit = True
    print("[DB] Connection successful. Autocommit enabled.")
    return conn


def insert_post(conn, record):
    cursor = conn.cursor()

    sql = """
    INSERT INTO posts
    (reddit_id, subreddit, title, body, image_url, image_path,
     image_ocr_text, author_masked, created_utc, raw_html,
     cleaned_text, embedding, cluster_id, keywords, distance_to_centroid)
    VALUES (%s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s)
    ON CONFLICT (reddit_id) DO UPDATE SET
        keywords = EXCLUDED.keywords,
        distance_to_centroid = EXCLUDED.distance_to_centroid,
        cluster_id = EXCLUDED.cluster_id,
        embedding = EXCLUDED.embedding

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
        record['cluster_id'],
        record.get('keywords'),
        record.get('distance_to_centroid')
    ))

    cursor.close()


#Scraping old reddit

def safe_get(url, max_retries=5):
    attempt = 0
    backoff = 2

    while attempt < max_retries:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)

            if r.status_code == 429:
                sleep_time = backoff ** attempt
                logging.warning(f"Rate limited (429). Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                attempt += 1
                continue

            r.raise_for_status()
            return r

        except requests.exceptions.RequestException as e:
            sleep_time = backoff ** attempt
            logging.warning(f"Request failed: {e}. Retrying in {sleep_time} seconds.")
            time.sleep(sleep_time)
            attempt += 1

    logging.error("Max retries exceeded. Skipping this page.")
    return None

    
def create_distance_index(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_distance_to_centroid
        ON posts(distance_to_centroid);
    """)
    conn.commit()
    cursor.close()



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


def tokenize(text):
    return [w for w in word_tokenize(text.lower())
            if w.isalpha() and w not in STOPWORDS]


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
            print(f"Collected {len(posts)} posts so far from r/{subreddit}")

        next_btn = soup.find("span", class_="next-button")
        print("Moving to next page...")
        if next_btn and next_btn.find("a"):
            url = next_btn.find("a")["href"]
            time.sleep(2 + np.random.uniform(0, 1)) #adding in random jitter for avoiding bot-like timimgs.
        else:
            break
    print(f"Finished r/{subreddit}. Total posts collected: {len(posts)}")

    return posts

#image ocr for posts with images

def ocr_image(url):
    try:
        r = safe_get(url)
        if not r:
            return ""
        img = Image.open(BytesIO(r.content))
        return pytesseract.image_to_string(img)
    except:
        return ""

#text embeddings and clustering using Doc2Vec and KMeans

def embed_and_cluster(records):

    texts = [r["cleaned_text"] + " " + r["image_ocr_text"] for r in records]
    tokenized_docs = [tokenize(t) for t in texts]

    print(f"Training Word2Vec on {len(tokenized_docs)} documents")

    w2v = Word2Vec(
        sentences=tokenized_docs,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        seed=42
    )

    w2v.save("word2vec_model.model")
    print("Word2Vec model saved.")

    vocab_words = w2v.wv.index_to_key
    word_vectors = np.array([w2v.wv[word] for word in vocab_words])

    # Cluster words into bins
    num_bins = 100
    print(f"Clustering {len(vocab_words)} words into {num_bins} bins")
    word_kmeans = KMeans(n_clusters=num_bins, n_init=10, random_state=42)
    word_labels = word_kmeans.fit_predict(word_vectors)

    word_to_bin = {
        vocab_words[i]: word_labels[i]
        for i in range(len(vocab_words))
    }

    # Build document embeddings
    doc_embeddings = []

    for doc in tokenized_docs:
        vec = np.zeros(num_bins)
        valid_words = [w for w in doc if w in word_to_bin]

        if len(valid_words) > 0:
            for w in valid_words:
                vec[word_to_bin[w]] += 1
            vec = vec / len(valid_words)

        doc_embeddings.append(vec)

    embeddings = np.array(doc_embeddings)
    print("Document embeddings generated.")

    # Cluster documents
    k = min(10, max(2, len(records)//50))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    print(f"Document clustering complete with {k} clusters.")

    centroids = kmeans.cluster_centers_

    # TF-IDF keyword extraction
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    cluster_keywords = {}

    for cluster_id in range(k):
        indices = np.where(labels == cluster_id)[0]
        cluster_tfidf = tfidf_matrix[indices]
        mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-5:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        cluster_keywords[cluster_id] = keywords

    # PCA visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure()
    for cluster_id in range(k):
        cluster_points = reduced[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")
    plt.legend()
    plt.title("Cluster Visualization (PCA)")
    plt.savefig("cluster_visualization.png")
    plt.close()

    # Store results
    for i, r in enumerate(records):
        r["embedding"] = pickle.dumps(embeddings[i])
        r["cluster_id"] = int(labels[i])
        r["keywords"] = ", ".join(cluster_keywords[labels[i]])
        r["distance_to_centroid"] = float(np.linalg.norm(embeddings[i] - centroids[labels[i]]))



def interactive_query(conn):

    print("Entering interactive query mode. Type 'exit' to quit.")
    cursor = conn.cursor()

    cursor.execute("SELECT embedding, cluster_id FROM posts")
    rows = cursor.fetchall()

    embeddings = []
    cluster_ids = []

    for row in rows:
        embeddings.append(pickle.loads(row[0]))
        cluster_ids.append(row[1])

    embeddings = np.array(embeddings)
    model = Word2Vec.load("word2vec_model.model")

    while True:
        query = input("Enter keywords: ")
        if query.lower() == "exit":
            break

        tokens = tokenize(clean_text(query))
        vec = np.zeros(embeddings.shape[1])

        for w in tokens:
            if w in model.wv:
                # simple average fallback for query
                vec += model.wv[w][:embeddings.shape[1]]

        if np.linalg.norm(vec) == 0:
            continue

        centroid_vectors = []

        unique_clusters = list(set(cluster_ids))
        for cid in unique_clusters:
            cluster_vectors = embeddings[np.array(cluster_ids) == cid]
            centroid_vectors.append((cid, np.mean(cluster_vectors, axis=0)))

        distances = [
            (cid, np.linalg.norm(vec - centroid))
            for cid, centroid in centroid_vectors
        ]

        best_cluster = min(distances, key=lambda x: x[1])[0]

        print(f"Closest cluster: {best_cluster}")

        cursor.execute(
            "SELECT title, keywords FROM posts WHERE cluster_id=%s LIMIT 5",
            (best_cluster,)
        )

        for title, keywords in cursor.fetchall():
            print(f"- {title}")
            print(f"  Keywords: {keywords}")

    cursor.close()



#main pipeline to run the entire process - scrape, ocr, embed, cluster, and store in DB

def run_pipeline(args):
    conn = get_db_conn(args.db_host, args.db_user, args.db_pass)

    all_posts = []

    for sub in args.subs:
        raw_posts = scrape_subreddit(sub, args.num)
        all_posts.extend(raw_posts)

    records = []

    for p in tqdm(all_posts):
        print(f"\nWorking on post {p['reddit_id']}")
        print(f"Title: {p['title'][:80]}...")
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
            create_distance_index(conn)


    logging.info("Pipeline completed.")


#cli to run the scraper and clustering pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subs", nargs="+", default=["technology", "technews", "tech", "netsec", "windowssecurity","cybersecurity"])
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
        conn = get_db_conn(args.db_host, args.db_user, args.db_pass)
        interactive_query(conn)


if __name__ == "__main__":
    main()
