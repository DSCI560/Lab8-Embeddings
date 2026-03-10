# Lab 8 – Representing Document Concepts with Embeddings

Team: GamerSups  
Members: Bhargav Limbasia, Harsh Marar, Nishkarsh Mittal   
Course: DSCI-560 Data Science Practicum

---

## What This Is

Two pipeline scripts that embed and cluster 3000 Reddit posts using Doc2Vec and Word2Vec Bag-of-Words. Posts are scraped from `old.reddit.com`, cleaned, embedded using 6 configurations each, clustered with KMeans (cosine distance), and stored in PostgreSQL.

---

## Requirements

```
pip install requests beautifulsoup4 aiohttp trafilatura gensim nltk
pip install scikit-learn numpy matplotlib psycopg2-binary tqdm pillow
```

PostgreSQL database named `lab8` must exist. Both scripts connect on port `5433` by default.

---

## Files

- `doc2vec.py` – Doc2Vec pipeline. Creates and populates `doc2vec_posts` table.
- `Agenticword2vec.py` – Word2Vec BoW pipeline. Creates and populates `word2vec_posts` table.
- `doc2vec_outputs/` – Models, PCA plots, metrics JSON, and bar charts from Doc2Vec run.
- `word2vec_outputs/` – Models, PCA plots, metrics JSON, and bar charts from Word2Vec run.

---

## How to Run

### First time (scrape + embed + store)

Run Doc2Vec first since Word2Vec can reuse its scraped data:

```bash
python doc2vec.py --db-host localhost --db-user <user> --db-pass <pass>
```

Then run Word2Vec, pulling posts directly from the DB instead of re-scraping:

```bash
python Agenticword2vec.py --db-host localhost --db-user <user> --db-pass <pass> --from-db --skip-articles
```

### If you already have posts in the DB

```bash
python doc2vec.py --db-host localhost --db-user <user> --db-pass <pass> --from-db --skip-articles
python Agenticword2vec.py --db-host localhost --db-user <user> --db-pass <pass> --from-db --skip-articles
```

### If the script crashed mid-scrape

Both scripts save a JSON checkpoint after scraping. On the next run, the checkpoint is picked up automatically so no need to re-scrape from scratch.

---

## Optional Flags

| Flag | What it does |
|---|---|
| `--from-db` | Load already-scraped posts from the DB, skip scraping |
| `--from-db-table` | Source table for `--from-db` (default: `doc2vec_posts`) |
| `--skip-articles` | Skip async external article fetch |
| `--enrich` | Fetch top comments per post (slow, off by default) |
| `--no-checkpoint` | Ignore existing checkpoint and re-scrape |
| `--images` | Enable OCR on post image URLs (requires pytesseract) |
| `--query` | Launch interactive cluster query after the pipeline finishes |
| `--num` | Posts to collect per subreddit (default: 500) |
| `--subs` | List of subreddits to scrape |
| `--interval` | Re-run every N minutes (0 = run once) |

---

## Data Collection

- Scrapes 500 posts per subreddit from 6 subreddits: `r/technology`, `r/technews`, `r/tech`, `r/cybersecurity`, `r/netsec`, `r/windowssecurity`.
- Pulls from 5 feeds per subreddit: hot, new, top (month), controversial (month), and rising.
- External article text is fetched asynchronously using `aiohttp` + `trafilatura` with 25 concurrent workers.
- Comment enrichment is off by default: title, selftext, and flair are sufficient for good clustering given how topically distinct the subreddits are.

---

## How the Embeddings Work

### Doc2Vec

- Each Reddit post is tagged and fed into a `gensim` Doc2Vec model.
- 6 configurations are trained: 3 DBOW and 3 DM, covering vector sizes 50, 100, 200, and 300.
- After training, document vectors are inferred (10 epochs per doc for stability), L2-normalized, and clustered with KMeans.
- Cosine distance is used for clustering by normalizing vectors before KMeans.

### Word2Vec Bag-of-Words

- Word2Vec is trained on all tokens across all posts.
- Word vectors are clustered into K semantic bins using KMeans.
- Each document is represented as a K-dimensional vector: for each bin, count how many of the document's words fall in it, then divide by total word count to normalize.
- 6 configurations varying vector size (50–300) and number of bins (50–300) are tested.

---

## Preprocessing

- URLs, punctuation, and standalone numbers stripped from text.
- Lowercased, NLTK stopwords removed, words shorter than 3 characters dropped.
- Posts with fewer than 5 tokens after cleaning are excluded.
- Author names are one-way hashed for privacy.

---

## Clustering Evaluation

Three metrics are computed for every configuration:

- **Silhouette score** – measures how well-separated clusters are (higher is better, range -1 to 1).
- **Davies-Bouldin index** – measures average cluster scatter vs. separation (lower is better).
- **Calinski-Harabasz score** – ratio of between-cluster to within-cluster variance (higher is better).

The best configuration is selected by silhouette score. Results are saved to `metrics_summary.json` and `metrics_summary_w2v.json`.

---

## Outputs

Each run produces:
- Trained model files (`.d2v` for Doc2Vec, `.model` for Word2Vec)
- `*_cluster_comparison.png`: 2×3 PCA scatter grid, one panel per config
- `*_metrics_bar.png`: bar charts comparing Silhouette and Davies-Bouldin across configs
- `metrics_summary*.json`: all metric scores in structured JSON

---

## Comparative Analysis – Which Method is Better?

The assignment asks us to compare Doc2Vec and Word2Vec BoW and determine which is better at representing document meaning, and for which vector dimensions.

**Results summary:**

Word2Vec BoW configs:

| Config | Silhouette | DB | CH |
|---|---|---|---|
| w2v_sz50_bins50 | 0.1604 | 2.551 | 203.5 |
| w2v_sz200_bins100 | 0.1102 | 3.274 | 140.8 |
| w2v_sz100_bins100 | 0.1063 | 3.365 | 126.5 |
| w2v_sz200_bins200 | 0.0916 | 3.801 | 106.8 |
| w2v_sz300_bins300 | 0.0881 | 3.792 | 96.2 |
| w2v_sz100_bins200 | 0.0772 | 3.963 | 96.4 |

Doc2Vec configs:

| Config | Silhouette | DB | CH |
|---|---|---|---|
| dbow_50_fast | 0.0819 | 3.301 | 172.1 |
| dbow_100_std | 0.0525 | 4.246 | 112.5 |
| dbow_200_deep | 0.0296 | 5.052 | 79.1 |
| dm_100_std | -0.0409 | 4.293 | 76.8 |
| dm_200_deep | -0.0692 | 5.198 | 53.6 |
| dm_300_large | -0.0815 | 5.693 | 44.3 |

**Word2Vec BoW outperforms Doc2Vec across all tested dimensions.** Every Word2Vec config produces a higher silhouette score and lower Davies-Bouldin index than the corresponding Doc2Vec config. The DM (Distributed Memory) Doc2Vec configs actually produce negative silhouette scores, meaning clusters are overlapping. Hence, the model isn't capturing enough structure at this corpus size. This is basically in-sync with the discussion that took place in the class regarding the domain and how some models would perform better given a specified domain. Here, our conclusion is that due to similar domain, word2vec outperforms doc2vec - which is objectively better, given it is applied in a varied domain.

The best performing config overall is `w2v_sz50_bins50` (silhouette 0.160, DB 2.55, CH 203.5). Smaller vector sizes and fewer bins worked better here, likely because the corpus vocabulary, while large, centers on a relatively narrow set of tech and security topics. High-dimensional Word2Vec embeddings introduce noise that hurts the word binning step; more bins means finer-grained splits that don't always correspond to meaningful semantic differences at this scale.

For Doc2Vec, the DBOW configs consistently beat DM. DBOW ignores word order and predicts context words from the document vector alone, which is better suited to short Reddit titles where word order carries less meaning than topic identity. DM tries to account for word order but needs more data and longer texts to do this well.

**Reason:** The BoW approach reduces each document to a compact, interpretable frequency profile over semantic word clusters. This is a strong representation for short social media posts where topic is signaled by specific high-frequency vocabulary (e.g., "ransomware", "CVE", "kernel", "GPU"). Doc2Vec's strength is learning document-level context from longer texts, hence, with short Reddit titles, it doesn't have enough signal to train reliably.

**Advantages which we saw from Word2Vec BoW:**
- More stable and interpretable embeddings at smaller corpus sizes.
- Bin frequency vectors are sparse and lightweight.
- Word clustering with smaller K values can capture topic-level groupings well.

**Disadvantages which we saw from Word2Vec BoW:**
- Loses word order and syntactic context entirely.
- The quality of document vectors depends heavily on how well the word bins were formed, so a bad K or poorly trained word vectors cascades into bad document vectors.
- Sensitive to vocabulary size; rare words get mapped to bins that may not represent them well.