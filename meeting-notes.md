# Team Meeting Notes – Representing Document Concepts with Embeddings
Team Name: GamerSups  
Members: Bhargav Limbasia, Harsh Marar, Nishkarsh Mittal  
Assignment: Lab 8 – Doc2Vec and Word2Vec Embeddings

---

## Friday – Reading and Planning

Mode: In-Person  
Duration: 30-45 minutes

What we did:
- Reviewed the Lab 8 assignment instructions together.
- Identified the two main tasks: Doc2Vec embeddings with clustering, and Word2Vec Bag-of-Words embeddings.
- Agreed on reusing the Reddit data (3000 posts across 6 subreddits) scraped in the previous lab.
- Divided responsibilities:
  - Harsh: reading up on Doc2Vec and Word2Vec, understanding configurations and hyperparameters.
  - Bhargav: database setup and the bulk of the pipeline coding.
- Decided to use `gensim` for both Doc2Vec and Word2Vec, `scikit-learn` for KMeans clustering, and PostgreSQL for storage.

Main takeaway:
We understood that the assignment wants us to run multiple configurations with different vector sizes and compare their clustering quality, not just run one and call it done.

---

## Saturday – Core Development

Mode: In-Person  
Duration: 4–5 hours

What we did:
- Harsh researched Doc2Vec and Word2Vec configurations in depth and settled on the config combinations to test:
  - For Doc2Vec: 3 DBOW configs (vec sizes 50, 100, 200) and 3 DM configs (vec sizes 100, 200, 300).
  - For Word2Vec BoW: 6 configs varying vector size (50–300) and number of bins (50–300).
  - Reasoning: covering small, medium, and large vector sizes gives us a meaningful comparison across the board.
- Bhargav built both pipeline scripts (`doc2vec.py` and `Agenticword2vec.py`) from scratch:
  - Scraper for `old.reddit.com` using the next-button pagination, covering hot, new, top, controversial, and rising feeds.
  - Async article fetching using `aiohttp` and `trafilatura` with 25 concurrent workers.
  - PostgreSQL tables (`doc2vec_posts`, `word2vec_posts`) with batch inserts using `execute_batch`.
  - Text cleaning, tokenization, and stopword removal pipeline.
  - Doc2Vec training loop with cosine-normalized KMeans clustering.
  - Word2Vec BoW pipeline: train word vectors, KMeans cluster words into bins, build normalized document frequency vectors, then cluster documents.
  - PCA visualizations (2×3 grid) and metrics bar charts saved as PNG files.
  - Silhouette, Davies-Bouldin, and Calinski-Harabasz scores computed for all configs.
  - TF-IDF keyword extraction per cluster for interpretability.

Issues encountered:
- The comment enrichment phase (fetching comments per post) was taking more than 3 hours for 3000 posts due to old.reddit's rate limiting — hitting 429s constantly with exponential backoff eating up most of the time.
- `asyncio.as_completed` was causing a `KeyError` when used with a future-to-index dict because it wraps futures in new coroutine objects, breaking the dict lookup.

Fixes:
- Moved comment enrichment behind an optional `--enrich` flag, off by default. The title, selftext, and flair alone proved sufficient for good clustering given the subreddits are topically distinct.
- Replaced the `as_completed` + future dict pattern with `asyncio.gather`, which runs all coroutines concurrently and preserves input order. This fixed the KeyError and made article fetching work correctly.

---

## Sunday – Refinement and Optimization (Heavy Work Day)

Mode: In-Person  
Duration: ~4 hours

What we did:
- Identified that re-scraping 3000 posts on every run was wasteful since the data was already in the DB. Bhargav added:
  - A JSON checkpoint file saved after scraping so the script resumes instead of restarting on crash.
  - A `--from-db` flag to load already-scraped posts directly from `doc2vec_posts`, skipping the scrape entirely for the Word2Vec run.
  - A `--skip-articles` flag to bypass external article fetching when not needed.
- Ran both pipelines end-to-end and collected metric results for all 12 configurations (6 Doc2Vec + 6 Word2Vec).
- Harsh reviewed the metric outputs and compared configurations to determine which performed best for the writeup.
- Cleaned up both scripts:
  - Moved all imports to the top.
  - Added controversial and rising feeds to the scraper alongside hot, new, and top.

Issues encountered:
- Some `try/except` blocks were swallowing errors silently in places where the code was already safe to run without them.

Fixes:
- Removed unnecessary try/except wrappers from `_parse_page`, `_host`, `ocr_image`, and the `_get_page` retry loop. Kept the async fetch try/except since network failures there need silent suppression.

---

## Monday – Final Review and Corrections

Mode: In-Person  
Duration: around 1.5 - 2 hours

What we did:
- Did a final read-through of both scripts together.
- Verified metric outputs and confirmed the best configurations for the writeup.
- Confirmed all outputs save to the correct directories (`doc2vec_outputs/`, `word2vec_outputs/`).
- Prepared the README and meeting notes.

---

## Overall Summary

Most of the heavy coding was done on Saturday and Sunday. Friday was planning, Monday was cleanup and verification.

We ensured:
- Both scripts run cleanly end-to-end without re-scraping if data is already in the DB.
- All 6 Doc2Vec and 6 Word2Vec configurations are trained and evaluated.
- Clustering quality is measured with three metrics: Silhouette, Davies-Bouldin, and Calinski-Harabasz.
- PCA visualizations and metric bar charts are saved for each method.
- All results are stored in PostgreSQL with embeddings, cluster IDs, and TF-IDF keywords per cluster.

The pipeline is complete and ready for submission.