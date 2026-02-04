"""
Topic Classifier Utilities

Provides topic prediction for gold data pipeline using the trained RoBERTa topic classifier.
Also includes validation utilities to test classifier on MFC data.
"""

import torch
from pathlib import Path
from typing import List, Optional

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# The 19 consolidated topic labels (alphabetically sorted as per pandas Category)
# This matches the order from training: df['cleaned_topic'].cat.categories
TOPIC_LABELS = [
    "Business & Economy",
    "Crime & Safety",
    "Disaster & Accidents",
    "Education",
    "Entertainment",
    "Environment & Nature",
    "Health",
    "Immigration",
    "Infrastructure & Transport",
    "Legal",
    "Lifestyle & Culture",
    "Media",
    "Other/Unknown",
    "Politics",
    "Science & Technology",
    "Social Issues",
    "Sports",
    "War & Conflict",
    "Weather",
]

# Expected MFC topic -> 19-topic mapping for validation
MFC_EXPECTED_TOPICS = {
    "immigration": ["Immigration"],
    "smoking": ["Health"],
    "samesex": ["Social Issues", "Legal", "Politics"],  # Could be any of these
}


class TopicClassifier:
    """Wrapper for the trained topic classifier."""

    def __init__(
        self,
        model_path: str = "notebooks/saved_models/final_topic_classifier",
        device: Optional[str] = None
    ):
        """
        Initialize the topic classifier.

        Args:
            model_path: Path to saved model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_path = Path(model_path)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading topic classifier from {self.model_path}...")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model with {len(TOPIC_LABELS)} topic classes")

    def predict(self, text: str, return_probs: bool = False) -> str:
        """
        Predict topic for a single text.

        Args:
            text: Article text
            return_probs: If True, return (topic, probabilities) tuple

        Returns:
            Predicted topic label (or tuple with probabilities)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()

        topic = TOPIC_LABELS[pred_idx]

        if return_probs:
            return topic, probs.cpu().numpy()[0]
        return topic

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> List[str]:
        """
        Predict topics for multiple texts.

        Args:
            texts: List of article texts
            batch_size: Batch size for inference
            show_progress: Show progress bar

        Returns:
            List of predicted topic labels
        """
        predictions = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting topics")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                pred_indices = torch.argmax(logits, dim=-1).cpu().tolist()

            batch_topics = [TOPIC_LABELS[idx] for idx in pred_indices]
            predictions.extend(batch_topics)

        return predictions


def validate_on_mfc(
    classifier: TopicClassifier,
    mfc_parquet_path: str,
    mfc_topic: str
) -> dict:
    """
    Validate topic classifier on MFC data where we know the expected topic.

    Args:
        classifier: Initialized TopicClassifier
        mfc_parquet_path: Path to assembled MFC corpus parquet
        mfc_topic: MFC topic name ('immigration', 'smoking', 'samesex')

    Returns:
        Dictionary with validation results
    """
    print(f"\nValidating classifier on MFC {mfc_topic}...")

    df = pd.read_parquet(mfc_parquet_path)
    print(f"Loaded {len(df)} articles")

    # Predict topics
    texts = df["text"].tolist()
    predictions = classifier.predict_batch(texts)

    # Check against expected topics
    expected = MFC_EXPECTED_TOPICS.get(mfc_topic, [])

    results = {
        "mfc_topic": mfc_topic,
        "total_articles": len(df),
        "expected_topics": expected,
        "prediction_counts": {},
        "match_rate": 0.0,
    }

    # Count predictions
    from collections import Counter
    pred_counts = Counter(predictions)
    results["prediction_counts"] = dict(pred_counts.most_common())

    # Calculate match rate (prediction in expected list)
    matches = sum(1 for p in predictions if p in expected)
    results["match_rate"] = matches / len(predictions) if predictions else 0.0

    # Print summary
    print(f"\nExpected topics: {expected}")
    print(f"Prediction distribution:")
    for topic, count in pred_counts.most_common(5):
        pct = count / len(predictions) * 100
        marker = "*" if topic in expected else ""
        print(f"  {topic}: {count} ({pct:.1f}%) {marker}")

    print(f"\nMatch rate: {results['match_rate']:.1%}")

    return results


def assign_topics_to_dataframe(
    df: pd.DataFrame,
    classifier: TopicClassifier,
    text_column: str = "article_text",
    output_column: str = "gpt_topic",
    batch_size: int = 16
) -> pd.DataFrame:
    """
    Assign topics to a DataFrame using the classifier.

    Args:
        df: DataFrame with article text
        classifier: Initialized TopicClassifier
        text_column: Name of column containing article text
        output_column: Name of column to store predictions
        batch_size: Batch size for inference

    Returns:
        DataFrame with topic predictions added
    """
    df = df.copy()

    texts = df[text_column].fillna("").tolist()
    predictions = classifier.predict_batch(texts, batch_size=batch_size)

    df[output_column] = predictions

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Topic classifier utilities")
    parser.add_argument("--validate-mfc", type=str,
                        help="Validate on MFC corpus (provide parquet path)")
    parser.add_argument("--mfc-topic", type=str, choices=["immigration", "smoking", "samesex"],
                        help="MFC topic for validation")
    parser.add_argument("--predict", type=str,
                        help="Predict topic for given text")

    args = parser.parse_args()

    # Initialize classifier
    base_path = Path(__file__).parent.parent
    classifier = TopicClassifier(
        model_path=str(base_path / "notebooks/saved_models/final_topic_classifier")
    )

    if args.validate_mfc and args.mfc_topic:
        results = validate_on_mfc(classifier, args.validate_mfc, args.mfc_topic)
        print(f"\nResults: {results}")

    elif args.predict:
        topic, probs = classifier.predict(args.predict, return_probs=True)
        print(f"\nPredicted topic: {topic}")
        print(f"Top 3 probabilities:")
        top_indices = probs.argsort()[-3:][::-1]
        for idx in top_indices:
            print(f"  {TOPIC_LABELS[idx]}: {probs[idx]:.3f}")

    else:
        # Quick test
        test_text = "The immigration bill passed the Senate yesterday with bipartisan support."
        topic = classifier.predict(test_text)
        print(f"\nTest prediction: '{test_text[:50]}...' -> {topic}")
