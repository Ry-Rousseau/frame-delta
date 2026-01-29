#!/usr/bin/env python3
"""
Load SemEval 2023 Task 3 Subtask 2 (English) data into PostgreSQL.

Subtask 2: Multi-label frame classification at article level.
Uses the standard 14 MFC frame categories.
"""

import os
import sys
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# Frame name normalization: SemEval uses underscores, MFC uses spaces
FRAME_NORMALIZE = {
    'Economic': 'Economic',
    'Capacity_and_resources': 'Capacity and resources',
    'Morality': 'Morality',
    'Fairness_and_equality': 'Fairness and equality',
    'Legality_Constitutionality_and_jurisprudence': 'Legality, constitutionality and jurisprudence',
    'Policy_prescription_and_evaluation': 'Policy prescription and evaluation',
    'Crime_and_punishment': 'Crime and punishment',
    'Security_and_defense': 'Security and defense',
    'Health_and_safety': 'Health and safety',
    'Quality_of_life': 'Quality of life',
    'Cultural_identity': 'Cultural identity',
    'Public_opinion': 'Public opinion',
    'Political': 'Political',
    'External_regulation_and_reputation': 'External regulation and reputation',
}


def create_table(cursor):
    """Create the semeval_subtask2 table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS semeval_subtask2 (
            id SERIAL PRIMARY KEY,
            article_id TEXT UNIQUE NOT NULL,
            title TEXT,
            text TEXT NOT NULL,
            frames_raw TEXT[],
            frames_mfc TEXT[],
            split TEXT NOT NULL,
            source TEXT DEFAULT 'semeval_2023_task3_subtask2_en',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_semeval_article_id ON semeval_subtask2(article_id);
        CREATE INDEX IF NOT EXISTS idx_semeval_split ON semeval_subtask2(split);
        CREATE INDEX IF NOT EXISTS idx_semeval_frames_mfc ON semeval_subtask2 USING GIN(frames_mfc);
    """)


def read_article(filepath: str) -> tuple:
    """Read article file and return (title, text)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        return None, None

    # Title is first line, content starts at line 3 (index 2)
    title = lines[0].strip() if lines else None
    text_lines = lines[2:] if len(lines) > 2 else []
    text = ''.join(text_lines).strip()

    return title, text


def read_labels(labels_file: str) -> dict:
    """Read labels file and return dict of article_id -> list of frames."""
    labels = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                article_id = parts[0].strip()
                frames = [f.strip() for f in parts[1].split(',') if f.strip()]
                labels[article_id] = frames
            elif len(parts) == 1:
                # Article with no frames
                labels[parts[0].strip()] = []
    return labels


def normalize_frames(frames: list) -> list:
    """Convert SemEval frame names to MFC standard names."""
    normalized = []
    for frame in frames:
        if frame in FRAME_NORMALIZE:
            normalized.append(FRAME_NORMALIZE[frame])
        else:
            print(f"  Warning: Unknown frame '{frame}'")
            normalized.append(frame)
    return normalized


def load_split(data_dir: str, split: str, conn):
    """Load a single split (train/dev) into the database."""
    cursor = conn.cursor()

    articles_dir = os.path.join(data_dir, f'{split}-articles-subtask-2')
    labels_file = os.path.join(data_dir, f'{split}-labels-subtask-2.txt')

    if not os.path.exists(articles_dir):
        print(f"  Skipping {split}: articles directory not found")
        return 0

    # Read labels if available
    labels = {}
    if os.path.exists(labels_file):
        labels = read_labels(labels_file)
        print(f"  Found {len(labels)} label entries")
    else:
        print(f"  No labels file found for {split}")

    # Read articles
    data = []
    article_files = [f for f in os.listdir(articles_dir)
                     if f.endswith('.txt') and not f.startswith('._')]

    for filename in article_files:
        article_id = filename.replace('article', '').replace('.txt', '')
        filepath = os.path.join(articles_dir, filename)

        title, text = read_article(filepath)
        if not text:
            continue

        frames_raw = labels.get(article_id, [])
        frames_mfc = normalize_frames(frames_raw)

        data.append((
            article_id,
            title,
            text,
            frames_raw,
            frames_mfc,
            split
        ))

    # Insert data
    if data:
        execute_values(cursor, """
            INSERT INTO semeval_subtask2 (article_id, title, text, frames_raw, frames_mfc, split)
            VALUES %s
            ON CONFLICT (article_id) DO UPDATE SET
                title = EXCLUDED.title,
                text = EXCLUDED.text,
                frames_raw = EXCLUDED.frames_raw,
                frames_mfc = EXCLUDED.frames_mfc,
                split = EXCLUDED.split
        """, data)
        conn.commit()

    cursor.close()
    return len(data)


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

    # Create table
    cursor = conn.cursor()
    create_table(cursor)
    conn.commit()
    cursor.close()

    # Path to SemEval English data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'sem_eval_23', 'data', 'en')

    try:
        total = 0
        for split in ['train', 'dev', 'test']:
            print(f"\nLoading {split} split...")
            count = load_split(data_dir, split, conn)
            print(f"  Loaded {count} articles")
            total += count

        # Show summary
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM semeval_subtask2")
        db_count = cursor.fetchone()[0]
        print(f"\nTotal in database: {db_count}")

        # Split breakdown
        print("\nBy split:")
        cursor.execute("""
            SELECT split, COUNT(*),
                   COUNT(*) FILTER (WHERE array_length(frames_mfc, 1) > 0) as labeled
            FROM semeval_subtask2
            GROUP BY split ORDER BY split
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]} articles ({row[2]} labeled)")

        # Frame distribution
        print("\nFrame distribution (MFC names):")
        cursor.execute("""
            SELECT unnest(frames_mfc) as frame, COUNT(*) as cnt
            FROM semeval_subtask2
            WHERE frames_mfc IS NOT NULL AND array_length(frames_mfc, 1) > 0
            GROUP BY frame
            ORDER BY cnt DESC
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")

        cursor.close()
        print("\nDone!")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
