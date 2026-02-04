"""
Gold Data Assembly Pipeline for Phase 2 Fine-Tuning

Assembles gold standard data from:
1. Media Frames Corpus (MFC) - with union aggregation across annotators
2. SemEval Task 3 Subtask 2 - article-level multi-label annotations

Output format matches silver data schema:
- text_generic_frame: list of frame labels
- gpt_topic: topic label (for topic injection)
- title: article title
- article_text: full article text

Usage:
    python scripts/db_loading/assemble_gold_data.py [--mfc-only] [--semeval-only] [--skip-topic-classifier]

Requirements:
    - MFC corpus assembled: media_frames_corpus/{topic}_corpus.parquet
    - SemEval data: sem_eval_23/data/en/
"""

import json
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Standard 15-frame labels (matching silver data)
FRAME_LABELS = [
    "Economic",
    "Capacity and resources",
    "Morality",
    "Fairness and equality",
    "Legality, constitutionality and jurisprudence",
    "Policy prescription and evaluation",
    "Crime and punishment",
    "Security and defense",
    "Health and safety",
    "Quality of life",
    "Cultural identity",
    "Public opinion",
    "Political",
    "External regulation and reputation",
    "Other"
]

# MFC code to label mapping
MFC_CODE_TO_LABEL = {
    1: "Economic",
    2: "Capacity and resources",
    3: "Morality",
    4: "Fairness and equality",
    5: "Legality, constitutionality and jurisprudence",
    6: "Policy prescription and evaluation",
    7: "Crime and punishment",
    8: "Security and defense",
    9: "Health and safety",
    10: "Quality of life",
    11: "Cultural identity",
    12: "Public opinion",
    13: "Political",
    14: "External regulation and reputation",
    15: "Other"
}

# SemEval frame name to standard label mapping
SEMEVAL_TO_LABEL = {
    "Economic": "Economic",
    "Capacity_and_resources": "Capacity and resources",
    "Morality": "Morality",
    "Fairness_and_equality": "Fairness and equality",
    "Legality_Constitutionality_and_jurisprudence": "Legality, constitutionality and jurisprudence",
    "Policy_prescription_and_evaluation": "Policy prescription and evaluation",
    "Crime_and_punishment": "Crime and punishment",
    "Security_and_defense": "Security and defense",
    "Health_and_safety": "Health and safety",
    "Quality_of_life": "Quality of life",
    "Cultural_identity": "Cultural identity",
    "Public_opinion": "Public opinion",
    "Political": "Political",
    "External_regulation_and_reputation": "External regulation and reputation",
}

# MFC topic to 19-topic taxonomy mapping
MFC_TOPIC_MAPPING = {
    "immigration": "Immigration",
    "smoking": "Health",
    "samesex": "Social Issues",  # Could also be Legal or Politics
}


def load_mfc_corpus(topic: str, base_path: Path) -> Optional[pd.DataFrame]:
    """Load assembled MFC corpus for a topic."""
    corpus_path = base_path / "media_frames_corpus" / f"{topic}_corpus.parquet"

    if not corpus_path.exists():
        print(f"  Warning: {corpus_path} not found. Run assemble_dataset.py for {topic} first.")
        return None

    df = pd.read_parquet(corpus_path)
    print(f"  Loaded {len(df)} articles from MFC {topic}")
    return df


def aggregate_mfc_labels_union(frame_annotations_json: str) -> list:
    """
    Aggregate frame labels across annotators using UNION strategy.

    Args:
        frame_annotations_json: JSON string like '{"annotator1": [1, 3, 5], "annotator2": [1, 4, 7]}'

    Returns:
        List of frame label strings (union of all annotator codes)
    """
    try:
        annotator_frames = json.loads(frame_annotations_json)
    except (json.JSONDecodeError, TypeError):
        return []

    # Union all frame codes across annotators
    all_codes = set()
    for annotator, codes in annotator_frames.items():
        all_codes.update(codes)

    # Convert codes to labels
    labels = []
    for code in sorted(all_codes):
        if code in MFC_CODE_TO_LABEL:
            labels.append(MFC_CODE_TO_LABEL[code])

    return labels


def process_mfc_data(base_path: Path) -> pd.DataFrame:
    """Load and process all MFC topics with union aggregation."""
    print("\nProcessing MFC data...")

    all_data = []

    for topic in ["immigration", "smoking", "samesex"]:
        df = load_mfc_corpus(topic, base_path)
        if df is None:
            continue

        # Process each article
        for _, row in df.iterrows():
            # Aggregate labels using union
            frame_labels = aggregate_mfc_labels_union(row["frame_annotations"])

            if not frame_labels:
                continue  # Skip articles with no valid labels

            all_data.append({
                "article_id": row["article_id"],
                "title": row["title"],
                "article_text": row["text"],
                "text_generic_frame": frame_labels,
                "gpt_topic": MFC_TOPIC_MAPPING.get(topic, "Other/Unknown"),
                "source": "mfc",
                "mfc_topic": topic,
            })

    if not all_data:
        print("  No MFC data found!")
        return pd.DataFrame()

    result = pd.DataFrame(all_data)
    print(f"  Total MFC articles: {len(result)}")

    # Stats
    labels_per_article = result["text_generic_frame"].apply(len)
    print(f"  Labels per article: mean={labels_per_article.mean():.2f}, median={labels_per_article.median():.1f}")

    return result


def load_semeval_labels(labels_path: Path) -> dict:
    """Load SemEval subtask 2 labels."""
    labels = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            article_id = parts[0]
            frame_names = parts[1].split(',') if len(parts) > 1 and parts[1] else []

            # Map to standard labels
            frame_labels = []
            for name in frame_names:
                if name in SEMEVAL_TO_LABEL:
                    frame_labels.append(SEMEVAL_TO_LABEL[name])
                else:
                    print(f"    Warning: Unknown SemEval frame '{name}'")

            labels[article_id] = frame_labels

    return labels


def load_semeval_article(article_path: Path) -> tuple:
    """Load a SemEval article file. Returns (title, text)."""
    with open(article_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    title = lines[0].strip() if lines else ""
    # Skip title and empty line, get rest as body
    body = '\n'.join(lines[2:]).strip() if len(lines) > 2 else ""

    return title, body


def process_semeval_data(base_path: Path) -> pd.DataFrame:
    """Load and process SemEval Task 3 Subtask 2 data."""
    print("\nProcessing SemEval data...")

    semeval_path = base_path / "sem_eval_23" / "data" / "en"

    all_data = []

    for split in ["train", "dev"]:
        labels_file = semeval_path / f"{split}-labels-subtask-2.txt"
        articles_dir = semeval_path / f"{split}-articles-subtask-2"

        if not labels_file.exists():
            print(f"  Warning: {labels_file} not found")
            continue

        if not articles_dir.exists():
            print(f"  Warning: {articles_dir} not found")
            continue

        # Load labels
        labels = load_semeval_labels(labels_file)
        print(f"  Loaded {len(labels)} labels from {split} set")

        # Load articles
        matched = 0
        for article_id, frame_labels in labels.items():
            article_file = articles_dir / f"article{article_id}.txt"

            if not article_file.exists():
                continue

            title, body = load_semeval_article(article_file)

            if not frame_labels:
                continue

            all_data.append({
                "article_id": f"semeval_{article_id}",
                "title": title,
                "article_text": body,
                "text_generic_frame": frame_labels,
                "gpt_topic": None,  # Will be filled by topic classifier
                "source": "semeval",
                "semeval_split": split,
            })
            matched += 1

        print(f"  Matched {matched} articles from {split} set")

    if not all_data:
        print("  No SemEval data found!")
        return pd.DataFrame()

    result = pd.DataFrame(all_data)
    print(f"  Total SemEval articles: {len(result)}")

    # Stats
    labels_per_article = result["text_generic_frame"].apply(len)
    print(f"  Labels per article: mean={labels_per_article.mean():.2f}, median={labels_per_article.median():.1f}")

    return result


def assign_topics_with_classifier(df: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    """
    Assign topics using the trained topic classifier.

    For articles without topics (e.g., SemEval), predicts using the classifier.
    For MFC articles, uses the known topic mapping but can optionally validate.
    """
    import sys
    sys.path.insert(0, str(base_path))
    from scripts.utils.topic_classifier_utils import TopicClassifier, assign_topics_to_dataframe

    # Initialize classifier
    classifier = TopicClassifier(
        model_path=str(base_path / "notebooks/saved_models/final_topic_classifier")
    )

    df = df.copy()

    # Find articles needing topic assignment
    needs_topic = df["gpt_topic"].isna()

    if needs_topic.any():
        print(f"\nAssigning topics to {needs_topic.sum()} articles...")
        df_needs_topic = df[needs_topic].copy()

        df_with_topics = assign_topics_to_dataframe(
            df_needs_topic,
            classifier,
            text_column="article_text",
            output_column="gpt_topic"
        )

        # Update original DataFrame
        df.loc[needs_topic, "gpt_topic"] = df_with_topics["gpt_topic"].values

    print(f"\nTopic distribution:")
    for topic, count in df["gpt_topic"].value_counts().items():
        print(f"  {topic}: {count}")

    return df


def create_train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    stratify_by_source: bool = True,
    random_state: int = 42
) -> tuple:
    """
    Create train/validation split, stratified by source.

    Args:
        df: Combined gold data
        val_ratio: Fraction for validation (default 0.2)
        stratify_by_source: Whether to stratify by data source
        random_state: Random seed for reproducibility

    Returns:
        (train_df, val_df)
    """
    if stratify_by_source:
        stratify = df["source"]
    else:
        stratify = None

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=stratify,
        random_state=random_state
    )

    print(f"\nTrain/Val split:")
    print(f"  Train: {len(train_df)} ({len(train_df[train_df['source']=='mfc'])} MFC, {len(train_df[train_df['source']=='semeval'])} SemEval)")
    print(f"  Val: {len(val_df)} ({len(val_df[val_df['source']=='mfc'])} MFC, {len(val_df[val_df['source']=='semeval'])} SemEval)")

    return train_df, val_df


def save_for_training(df: pd.DataFrame, output_path: Path, name: str):
    """
    Save DataFrame in format ready for training.

    Converts text_generic_frame from list to numpy array to match silver data format.
    """
    # Convert frame lists to numpy arrays (matching silver data format)
    df = df.copy()
    df["text_generic_frame"] = df["text_generic_frame"].apply(np.array)

    # Select and reorder columns to match silver data
    output_cols = ["text_generic_frame", "gpt_topic", "title", "article_text"]

    # Keep source info in separate columns for analysis
    extra_cols = [c for c in df.columns if c not in output_cols]

    df_out = df[output_cols + extra_cols]

    df_out.to_parquet(output_path, index=False)
    print(f"  Saved {name}: {output_path} ({len(df_out)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Assemble gold data for Phase 2 fine-tuning")
    parser.add_argument("--mfc-only", action="store_true", help="Only process MFC data")
    parser.add_argument("--semeval-only", action="store_true", help="Only process SemEval data")
    parser.add_argument("--skip-topic-classifier", action="store_true",
                        help="Skip topic classifier (use placeholder topics)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")

    args = parser.parse_args()

    # Get project root (script is in scripts/db_loading/)
    base_path = Path(__file__).parent.parent.parent
    output_dir = base_path / args.output_dir
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Gold Data Assembly Pipeline")
    print("=" * 60)

    dfs = []

    # Process MFC
    if not args.semeval_only:
        mfc_df = process_mfc_data(base_path)
        if len(mfc_df) > 0:
            dfs.append(mfc_df)

    # Process SemEval
    if not args.mfc_only:
        semeval_df = process_semeval_data(base_path)
        if len(semeval_df) > 0:
            dfs.append(semeval_df)

    if not dfs:
        print("\nERROR: No data loaded. Check that source files exist.")
        return

    # Combine
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"Combined dataset: {len(combined_df)} articles")

    # Assign topics
    if args.skip_topic_classifier:
        print("\nSkipping topic classifier (using placeholder topics)")
        combined_df.loc[combined_df["gpt_topic"].isna(), "gpt_topic"] = "Other/Unknown"
    else:
        combined_df = assign_topics_with_classifier(combined_df, base_path)

    # Create train/val split
    train_df, val_df = create_train_val_split(
        combined_df,
        val_ratio=args.val_ratio
    )

    # Save outputs
    print(f"\nSaving outputs to {output_dir}/")
    save_for_training(train_df, output_dir / "gold_combined_train.parquet", "train")
    save_for_training(val_df, output_dir / "gold_combined_val.parquet", "val")

    # Also save combined for analysis
    save_for_training(combined_df, output_dir / "gold_combined_all.parquet", "all")

    # Summary stats
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print("=" * 60)

    print("\nLabel distribution (train):")
    from collections import Counter
    all_labels = []
    for labels in train_df["text_generic_frame"]:
        all_labels.extend(labels)
    label_counts = Counter(all_labels)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    print(f"\nLabels per article (train):")
    lpa = train_df["text_generic_frame"].apply(len)
    print(f"  Mean: {lpa.mean():.2f}")
    print(f"  Median: {lpa.median():.1f}")
    print(f"  Min: {lpa.min()}, Max: {lpa.max()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
