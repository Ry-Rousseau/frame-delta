#!/usr/bin/env python3
"""
Load FrAC gold standard data into PostgreSQL.

Creates a table with both the FrAC 9-label scheme and the original 15-label MFC scheme.
"""

import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# Label mappings from original MFC numeric labels
# FrAC collapsed: Legality+Crime, Political+Policy, removed Capacity and External Regulation
LABEL_MAPPING = {
    1: {
        'frac': 'Economic',
        'mfc': 'Economic'
    },
    3: {
        'frac': 'Morality',
        'mfc': 'Morality'
    },
    4: {
        'frac': 'Fairness and Equality',
        'mfc': 'Fairness and equality'
    },
    5: {
        'frac': 'Legality and Crime',  # FrAC merged Legality + Crime
        'mfc': 'Legality, constitutionality and jurisprudence'
    },
    6: {
        'frac': 'Political and Policies',  # FrAC merged Policy + Political
        'mfc': 'Policy prescription and evaluation'
    },
    7: {
        'frac': 'Legality and Crime',  # Would map here if present
        'mfc': 'Crime and punishment'
    },
    8: {
        'frac': 'Security and Defense',
        'mfc': 'Security and defense'
    },
    9: {
        'frac': 'Health and Safety',
        'mfc': 'Health and safety'
    },
    10: {
        'frac': 'Quality of Life',  # Not in gold standard but defining for completeness
        'mfc': 'Quality of life'
    },
    11: {
        'frac': 'Cultural Identity',
        'mfc': 'Cultural identity'
    },
    12: {
        'frac': 'Public Opinion',
        'mfc': 'Public opinion'
    },
    13: {
        'frac': 'Political and Policies',  # Would map here if present
        'mfc': 'Political'
    },
    15: {
        'frac': 'Other',
        'mfc': 'Other'
    }
}


def create_table(cursor):
    """Create the frac_gold_standard table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS frac_gold_standard (
            id SERIAL PRIMARY KEY,
            sentence TEXT NOT NULL,
            label_numeric INTEGER NOT NULL,
            label_frac TEXT NOT NULL,
            label_mfc TEXT NOT NULL,
            source TEXT DEFAULT 'frac_gold_standard',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_frac_label_frac ON frac_gold_standard(label_frac);
        CREATE INDEX IF NOT EXISTS idx_frac_label_mfc ON frac_gold_standard(label_mfc);
    """)


def load_frac_data(csv_path: str, conn):
    """Load FrAC gold standard CSV into PostgreSQL."""
    cursor = conn.cursor()

    # Create table
    create_table(cursor)
    conn.commit()

    # Read CSV
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} rows")

    # Prepare data with both label schemes
    data = []
    unmapped = set()
    for _, row in df.iterrows():
        label_num = row['label']
        if label_num in LABEL_MAPPING:
            data.append((
                row['sentence'],
                label_num,
                LABEL_MAPPING[label_num]['frac'],
                LABEL_MAPPING[label_num]['mfc']
            ))
        else:
            unmapped.add(label_num)
            data.append((
                row['sentence'],
                label_num,
                'Unknown',
                'Unknown'
            ))

    if unmapped:
        print(f"Warning: Unmapped labels found: {unmapped}")

    # Insert data
    print(f"Inserting {len(data)} rows...")
    execute_values(cursor, """
        INSERT INTO frac_gold_standard (sentence, label_numeric, label_frac, label_mfc)
        VALUES %s
    """, data)
    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM frac_gold_standard")
    count = cursor.fetchone()[0]
    print(f"Total rows in frac_gold_standard: {count}")

    # Show distribution
    print("\nLabel distribution (FrAC scheme):")
    cursor.execute("""
        SELECT label_frac, COUNT(*) as cnt
        FROM frac_gold_standard
        GROUP BY label_frac
        ORDER BY cnt DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cursor.close()


def main():
    # Connect to database
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

    # Path to FrAC gold standard
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'FrAC', 'gold_standard_single_label_all.csv')

    try:
        load_frac_data(csv_path, conn)
        print("\nDone!")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
