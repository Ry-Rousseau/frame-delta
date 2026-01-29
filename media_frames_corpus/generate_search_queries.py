"""
Generate Nexis Uni search query batches for immigration.json NYT articles.

Outputs:
- search_queries/batch_XX.txt: Search query strings ready to paste into Nexis
- search_queries/manifest.json: Tracking file with batch metadata
"""

import json
import os
from pathlib import Path

# Config
MAX_QUERY_CHARS = 5000
INPUT_FILE = "immigration.json"
OUTPUT_DIR = "search_queries"

def load_nyt_articles(filepath):
    """Load and filter for exact 'new york times' source (no blogs), with valid titles."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nyt_articles = [
        (k, v) for k, v in data.items()
        if v.get('source', '').lower() == 'new york times'
        and v.get('title', '').strip()  # Exclude empty titles
    ]

    # Sort by year, month for organized batching
    nyt_articles.sort(key=lambda x: (x[1].get('year', 0), x[1].get('month', 0)))
    return nyt_articles


def generate_batches(articles, max_chars=MAX_QUERY_CHARS):
    """Generate search query batches that fit within character limit."""
    batches = []
    current_batch = []
    current_len = 1  # Opening paren

    for article_id, article in articles:
        title = article.get('title', '')
        title_escaped = title.replace('"', '\\"')

        if not current_batch:
            cost = 12 + len(title_escaped)  # headline("TITLE")
        else:
            cost = 16 + len(title_escaped)  # OR headline("TITLE")

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


def main():
    base_path = Path(__file__).parent
    os.chdir(base_path)

    # Load articles
    articles = load_nyt_articles(INPUT_FILE)
    print(f"Loaded {len(articles)} NYT articles from {INPUT_FILE}")

    # Generate batches
    batches = generate_batches(articles)
    print(f"Generated {len(batches)} batches")

    # Create output directory
    output_path = base_path / OUTPUT_DIR
    output_path.mkdir(exist_ok=True)

    # Generate manifest and query files
    manifest = {
        "total_articles": len(articles),
        "total_batches": len(batches),
        "batches": []
    }

    for i, batch in enumerate(batches, 1):
        batch_num = f"{i:02d}"

        # Get batch metadata
        years = [a[1].get('year') for a in batch]
        months = [a[1].get('month') for a in batch]
        article_ids = [a[0] for a in batch]
        titles = [a[1].get('title') for a in batch]

        # Generate query
        query = format_search_query(batch)

        # Write query file
        query_file = output_path / f"batch_{batch_num}.txt"
        with open(query_file, 'w', encoding='utf-8') as f:
            f.write(query)

        # Add to manifest
        batch_info = {
            "batch_id": batch_num,
            "article_count": len(batch),
            "year_range": [min(years), max(years)],
            "query_length": len(query),
            "query_file": f"batch_{batch_num}.txt",
            "download_folder": f"batch_{batch_num}",
            "status": "pending",  # pending, downloaded, processed
            "articles": [
                {"id": aid, "title": title, "year": y, "month": m}
                for (aid, _), title, y, m in zip(batch, titles, years, months)
            ]
        }
        manifest["batches"].append(batch_info)

        print(f"  Batch {batch_num}: {len(batch)} articles, years {min(years)}-{max(years)}, {len(query)} chars")

    # Write manifest
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nOutput written to {output_path}/")
    print(f"  - {len(batches)} batch query files (batch_XX.txt)")
    print(f"  - manifest.json (tracking file)")


if __name__ == "__main__":
    main()
