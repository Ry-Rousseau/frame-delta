#!/usr/bin/env python3
"""
Export silver fine-tuning data from PostgreSQL to parquet files.

Exports to 3 parquet files for RunPod upload:
- data/silver_articles_part1.parquet
- data/silver_articles_part2.parquet
- data/silver_articles_part3.parquet

Columns: text_generic_frame, gpt_topic, title, article_text
"""

import os
import sys
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def export_silver_data():
    """Export silver fine-tuning data to 3 parquet files."""
    print("Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        print("Connected!")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    # Query to get all required columns
    # Join mm_framing_full with newsarticles to get maintext
    query = """
        SELECT
            a.text_generic_frame,
            a.gpt_topic,
            a.title,
            b.maintext as article_text
        FROM mm_framing_full a
        JOIN newsarticles b ON a.url = b.url
        WHERE b.maintext IS NOT NULL
        AND LENGTH(b.maintext) > 100
    """

    print("Executing query...")
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"Loaded {len(df):,} rows")

    # Filter for >100 words
    df['num_words'] = df['article_text'].str.split().str.len()
    df = df[df['num_words'] > 100]
    df = df.dropna()
    df = df.drop(columns=['num_words'])
    df = df.reset_index(drop=True)
    print(f"After filtering: {len(df):,} rows")

    # Split into 3 roughly equal parts
    n = len(df)
    part_size = n // 3

    part1 = df.iloc[:part_size]
    part2 = df.iloc[part_size:2*part_size]
    part3 = df.iloc[2*part_size:]

    print(f"\nSplit sizes:")
    print(f"  Part 1: {len(part1):,} rows")
    print(f"  Part 2: {len(part2):,} rows")
    print(f"  Part 3: {len(part3):,} rows")

    # Get project root (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Export to parquet
    paths = [
        os.path.join(data_dir, 'silver_articles_part1.parquet'),
        os.path.join(data_dir, 'silver_articles_part2.parquet'),
        os.path.join(data_dir, 'silver_articles_part3.parquet'),
    ]

    for i, (part, path) in enumerate(zip([part1, part2, part3], paths), 1):
        part.to_parquet(path, index=False)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Exported Part {i}: {path} ({size_mb:.1f} MB)")

    # Show column info
    print(f"\nColumns exported: {df.columns.tolist()}")
    print(f"\nSample text_generic_frame: {df.iloc[0]['text_generic_frame']}")
    print(f"Sample gpt_topic: {df.iloc[0]['gpt_topic']}")

    print("\nDone!")


if __name__ == "__main__":
    export_silver_data()
