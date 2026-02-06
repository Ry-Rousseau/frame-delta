#!/usr/bin/env python3
"""
Export gold fine-tuning data from PostgreSQL to a parquet file.

Exports to:
- data/gold_train_data.parquet

Columns:
- id
- source
- article_id
- text (formatted_text from DB; must begin with "TOPIC:")
- labels_idx (list[int])
"""

import json
import os
import sys
from typing import Any, List

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def _parse_labels(value: Any) -> List[int]:
    """Parse labels_idx_json into a list of ints."""
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    return []


def export_gold_data() -> None:
    """Export gold fine-tuning data to parquet."""
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
    except Exception as exc:
        print(f"Connection failed: {exc}")
        sys.exit(1)

    query = """
        SELECT
            id,
            source,
            article_id,
            formatted_text,
            labels_idx_json
        FROM gold_train_data
    """

    print("Executing query...")
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"Loaded {len(df):,} rows")

    # Basic cleaning
    df = df.dropna(subset=["formatted_text", "labels_idx_json"]).reset_index(drop=True)
    df["labels_idx"] = df["labels_idx_json"].apply(_parse_labels)
    df = df.drop(columns=["labels_idx_json"])

    # Rename for training script consistency
    df = df.rename(columns={"formatted_text": "text"})

    # Sanity checks
    bad_prefix = df[~df["text"].astype(str).str.startswith("TOPIC:")]
    if len(bad_prefix) > 0:
        print(f"Warning: {len(bad_prefix)} rows do not start with 'TOPIC:'.")
        print("First bad example:")
        print(bad_prefix.iloc[0]["text"][:200])

    empty_labels = df["labels_idx"].apply(len).eq(0).sum()
    if empty_labels > 0:
        print(f"Warning: {empty_labels} rows have empty labels.")

    # Save to parquet
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "gold_train_data.parquet")
    df.to_parquet(out_path, index=False)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nExported: {out_path} ({size_mb:.1f} MB)")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample labels_idx: {df.iloc[0]['labels_idx']}")
    print("Done!")


if __name__ == "__main__":
    export_gold_data()
