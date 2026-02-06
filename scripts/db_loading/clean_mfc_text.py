"""
Clean LexisNexis metadata from media_frames_corpus.text into text_cleaned.

Behavior:
- Truncates everything from the 'Classification' line onward (always present).
- Removes trailing LexisNexis media/correction blocks and standalone URL lines
  that appear at the end of the article.
- Preserves original text column.

Usage:
    python scripts/db_loading/clean_mfc_text.py --dry-run
    python scripts/db_loading/clean_mfc_text.py
"""

import os
import re
import argparse
from typing import List, Tuple

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------

MARKER_RE = re.compile(
    r"^(?:Graphic|Graphics|Photo|Photos|Map/Diagram|Map|Maps|Chart|Charts|Graph|Graphs|"
    r"Table|Tables|Illustration|Illustrations|Drawing|Drawings|Correction|Online Correction|"
    r"Correction-Date|Correction Date|Image|Images|Caption)(\b|:)",
    re.IGNORECASE,
)
URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)


def _is_marker_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if URL_RE.match(stripped):
        return True
    if MARKER_RE.match(stripped):
        return True

    lower = stripped.lower()
    for prefix in (
        "photo:", "photos:", "graphic:", "graphics:", "map:", "maps:",
        "map/diagram:", "chart:", "charts:", "graph:", "graphs:", "table:", "tables:",
        "illustration:", "illustrations:", "drawing:", "drawings:",
        "correction:", "online correction:", "correction-date:", "correction date:",
        "image:", "images:", "caption:",
    ):
        if lower.startswith(prefix):
            return True

    return False


def clean_text(text: str) -> str:
    if text is None:
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # 1) Truncate at Classification (always present, near end)
    cutoff = None
    for i, line in enumerate(lines):
        if line.strip() == "Classification":
            cutoff = i
            break
    if cutoff is None:
        for i, line in enumerate(lines):
            if line.strip().lower() == "end of document":
                cutoff = i
                break
    if cutoff is not None:
        lines = lines[:cutoff]

    # Trim trailing empty lines
    while lines and lines[-1].strip() == "":
        lines.pop()

    # 2) Remove trailing media/correction/url blocks at end
    def drop_trailing_marker_block(buf: List[str]) -> List[str]:
        if not buf:
            return buf
        last_blank = None
        for i in range(len(buf) - 1, -1, -1):
            if buf[i].strip() == "":
                last_blank = i
                break
        if last_blank is None:
            return buf
        j = last_blank + 1
        while j < len(buf) and buf[j].strip() == "":
            j += 1
        if j >= len(buf):
            return buf[:last_blank]
        if _is_marker_line(buf[j]):
            return buf[:last_blank]
        return buf

    prev_len = None
    while prev_len != len(lines):
        prev_len = len(lines)

        # Strip trailing marker lines (and any blank lines between)
        while lines and _is_marker_line(lines[-1]):
            lines.pop()
        while lines and lines[-1].strip() == "":
            lines.pop()

        # Drop trailing marker block after a blank line
        lines = drop_trailing_marker_block(lines)
        while lines and lines[-1].strip() == "":
            lines.pop()

    cleaned = "\n".join(lines).strip()

    if cleaned:
        return cleaned

    # Fallback: if aggressive cleanup yields empty (e.g., caption-only records),
    # return text truncated at Classification without stripping marker blocks.
    fallback_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    fallback_cutoff = None
    for i, line in enumerate(fallback_lines):
        if line.strip() == "Classification":
            fallback_cutoff = i
            break
    if fallback_cutoff is not None:
        fallback_lines = fallback_lines[:fallback_cutoff]
    while fallback_lines and fallback_lines[-1].strip() == "":
        fallback_lines.pop()
    return "\n".join(fallback_lines).strip()


# ---------------------------------------------------------------------
# DB update
# ---------------------------------------------------------------------


def fetch_rows(cur) -> List[Tuple[int, str]]:
    cur.execute("SELECT id, text FROM media_frames_corpus;")
    return cur.fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean LexisNexis metadata in media_frames_corpus.text")
    parser.add_argument("--dry-run", action="store_true", help="Compute stats and sample output, do not update DB")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for DB updates")
    parser.add_argument("--sample", type=int, default=5, help="Number of random samples to print in dry-run")
    args = parser.parse_args()

    load_dotenv(dotenv_path=".env")

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
    cur = conn.cursor()

    # Ensure target column exists
    cur.execute("ALTER TABLE media_frames_corpus ADD COLUMN IF NOT EXISTS text_cleaned TEXT;")
    conn.commit()

    rows = fetch_rows(cur)
    total = len(rows)

    updates = []
    empty_count = 0
    shortened = 0
    total_removed = 0

    for id_, text in rows:
        cleaned = clean_text(text)
        if cleaned is None:
            cleaned = ""
        if len(cleaned) == 0:
            empty_count += 1
        if len(cleaned) < len(text):
            shortened += 1
            total_removed += (len(text) - len(cleaned))
        updates.append((cleaned, id_))

    if args.dry_run:
        print(f"Rows: {total}")
        print(f"Shortened: {shortened} ({shortened/total:.1%})")
        print(f"Empty after clean: {empty_count}")
        avg_removed = total_removed / total if total else 0
        print(f"Avg chars removed: {avg_removed:.1f}")

        # Print samples
        cur.execute(
            """
            SELECT id, title, text
            FROM media_frames_corpus
            ORDER BY RANDOM()
            LIMIT %s;
            """,
            (args.sample,),
        )
        samples = cur.fetchall()
        for id_, title, text in samples:
            cleaned = clean_text(text)
            print("=" * 80)
            print(f"Sample id={id_} | title={title}")
            print("--- ORIGINAL TAIL ---")
            print(text.replace("\r\n", "\n").replace("\r", "\n")[-800:])
            print("--- CLEANED TAIL ---")
            print(cleaned[-800:])
        cur.close()
        conn.close()
        return

    # Apply updates in batches
    for i in range(0, len(updates), args.batch_size):
        batch = updates[i:i + args.batch_size]
        execute_values(
            cur,
            """
            UPDATE media_frames_corpus AS t
            SET text_cleaned = v.text_cleaned
            FROM (VALUES %s) AS v(text_cleaned, id)
            WHERE t.id = v.id;
            """,
            batch,
        )
        conn.commit()

    print(f"Updated {total} rows. Shortened: {shortened} ({shortened/total:.1%}). Empty: {empty_count}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
