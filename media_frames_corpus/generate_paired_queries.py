"""
Generate paired Nexis Uni search query batches for smoking/samesex topics.

Each super-batch contains two query files (a and b) to be pasted sequentially
before downloading, reducing total download operations by half.

Usage:
    python generate_paired_queries.py smoking
    python generate_paired_queries.py samesex
"""

import json
import os
import sys
from pathlib import Path

MAX_QUERY_CHARS = 5000


def load_nyt_articles(filepath):
    """Load and filter for exact 'new york times' source (no blogs), with valid titles."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nyt_articles = [
        (k, v) for k, v in data.items()
        if v.get('source', '').lower() == 'new york times'
        and v.get('title', '').strip()
    ]

    nyt_articles.sort(key=lambda x: (x[1].get('year', 0), x[1].get('month', 0)))
    return nyt_articles


def generate_batches(articles, max_chars=MAX_QUERY_CHARS):
    """Generate search query batches that fit within character limit."""
    batches = []
    current_batch = []
    current_len = 1

    for article_id, article in articles:
        title = article.get('title', '')
        title_escaped = title.replace('"', '\\"')

        if not current_batch:
            cost = 12 + len(title_escaped)
        else:
            cost = 16 + len(title_escaped)

        if current_len + cost + 1 > max_chars:
            batches.append(current_batch)
            current_batch = [(article_id, article)]
            current_len = 1 + 12 + len(title_escaped)
        else:
            current_batch.append((article_id, article))
            current_len += cost

    if current_batch:
        batches.append(current_batch)

    return batches


def format_search_query(batch):
    """Format batch as Nexis Uni headline search query."""
    parts = []
    for article_id, article in batch:
        title = article.get('title', '').replace('"', '\\"')
        parts.append(f'headline("{title}")')
    return "(" + " OR ".join(parts) + ")"


def pair_batches(batches):
    """Pair consecutive batches into super-batches."""
    paired = []
    for i in range(0, len(batches), 2):
        if i + 1 < len(batches):
            paired.append((batches[i], batches[i + 1]))
        else:
            paired.append((batches[i], None))  # Odd batch at end
    return paired


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_paired_queries.py <topic>")
        print("  topic: smoking or samesex")
        sys.exit(1)

    topic = sys.argv[1].lower()
    if topic not in ['smoking', 'samesex']:
        print(f"Unknown topic: {topic}")
        sys.exit(1)

    base_path = Path(__file__).parent
    os.chdir(base_path)

    input_file = f"{topic}.json"
    queries_dir = base_path / f"{topic}_queries"
    downloads_dir = base_path / f"{topic}_downloads"

    # Load articles
    articles = load_nyt_articles(input_file)
    print(f"Loaded {len(articles)} NYT articles from {input_file}")

    # Generate single batches
    batches = generate_batches(articles)
    print(f"Generated {len(batches)} single batches")

    # Pair them
    paired = pair_batches(batches)
    print(f"Paired into {len(paired)} super-batches")

    # Create directories
    queries_dir.mkdir(exist_ok=True)
    downloads_dir.mkdir(exist_ok=True)

    # Generate manifest and query files
    manifest = {
        "topic": topic,
        "total_articles": len(articles),
        "single_batches": len(batches),
        "super_batches": len(paired),
        "batches": []
    }

    for i, (batch_a, batch_b) in enumerate(paired, 1):
        batch_num = f"{i:02d}"

        # Create download folder
        (downloads_dir / f"batch_{batch_num}").mkdir(exist_ok=True)

        # Batch A
        query_a = format_search_query(batch_a)
        years_a = [a[1].get('year') for a in batch_a]
        query_a_file = queries_dir / f"batch_{batch_num}_query_a.txt"
        with open(query_a_file, 'w', encoding='utf-8') as f:
            f.write(query_a)

        # Batch B (if exists)
        if batch_b:
            query_b = format_search_query(batch_b)
            years_b = [a[1].get('year') for a in batch_b]
            query_b_file = queries_dir / f"batch_{batch_num}_query_b.txt"
            with open(query_b_file, 'w', encoding='utf-8') as f:
                f.write(query_b)
            total_articles = len(batch_a) + len(batch_b)
            year_range = [min(years_a + years_b), max(years_a + years_b)]
        else:
            total_articles = len(batch_a)
            year_range = [min(years_a), max(years_a)]

        batch_info = {
            "batch_id": batch_num,
            "article_count": total_articles,
            "year_range": year_range,
            "query_a_articles": len(batch_a),
            "query_b_articles": len(batch_b) if batch_b else 0,
            "download_folder": f"batch_{batch_num}",
        }
        manifest["batches"].append(batch_info)

        articles_b = len(batch_b) if batch_b else 0
        print(f"  Batch {batch_num}: {len(batch_a)} + {articles_b} = {total_articles} articles, years {year_range[0]}-{year_range[1]}")

    # Write manifest
    manifest_file = queries_dir / "manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nOutput:")
    print(f"  Queries: {queries_dir}/")
    print(f"  Downloads: {downloads_dir}/")


if __name__ == "__main__":
    main()
