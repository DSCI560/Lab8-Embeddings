#!/usr/bin/env python3

import psycopg2
import argparse


def get_db_conn(host, user, password, database="lab5_reddit"):
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    conn.autocommit = True
    return conn


def run_query(cursor, title, query):
    print(f"\n{title}")
    cursor.execute(query)
    rows = cursor.fetchall()
    for row in rows:
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-host", required=True)
    parser.add_argument("--db-user", required=True)
    parser.add_argument("--db-pass", required=True)
    args = parser.parse_args()

    conn = get_db_conn(args.db_host, args.db_user, args.db_pass)
    cursor = conn.cursor()

    # 1. Total entries
    run_query(
        cursor,
        "Total Number of Posts",
        """
        SELECT COUNT(*) FROM posts;
        """
    )

    # 2. Cluster distribution with percentages
    run_query(
        cursor,
        "Cluster Distribution",
        """
        SELECT 
            cluster_id,
            COUNT(*) AS total_posts,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS percentage
        FROM posts
        GROUP BY cluster_id
        ORDER BY total_posts DESC;
        """
    )

    # 3. Sample 3 posts per cluster (representative examples)
    run_query(
        cursor,
        "Sample Posts Per Cluster",
        """
        SELECT cluster_id, title
        FROM (
            SELECT 
                cluster_id,
                title,
                ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY RANDOM()) as rn
            FROM posts
        ) sub
        WHERE rn <= 3
        ORDER BY cluster_id;
        """
    )

    # 4. Subreddit distribution across clusters
    run_query(
        cursor,
        "Subreddit Distribution Across Clusters",
        """
        SELECT 
            subreddit,
            cluster_id,
            COUNT(*) AS count
        FROM posts
        GROUP BY subreddit, cluster_id
        ORDER BY subreddit, count DESC;
        """
    )

    # 5. Top 10 most common words overall (simple frequency from cleaned_text)
    run_query(
        cursor,
        "Top 10 Most Frequent Words (All Posts)",
        """
        SELECT word, COUNT(*) AS frequency
        FROM (
            SELECT regexp_split_to_table(cleaned_text, '\\s+') AS word
            FROM posts
        ) t
        WHERE length(word) > 3
        GROUP BY word
        ORDER BY frequency DESC
        LIMIT 10;
        """
    )

    # 6. Top words per cluster
    """
    run_query(
        cursor,
        "Top Words Per Cluster",
        SELECT cluster_id, word, COUNT(*) AS frequency
        FROM (
            SELECT 
                cluster_id,
                regexp_split_to_table(cleaned_text, '\\s+') AS word
            FROM posts
        ) t
        WHERE length(word) > 3
        GROUP BY cluster_id, word
        ORDER BY cluster_id, frequency DESC;   

    )
    """    #Note: The above query for top words per cluster can be very large. Consider limiting

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
