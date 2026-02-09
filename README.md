# Frame Delta

A lightweight multi-label classifier that detects 15 standard media frames in news articles, enabling quantitative comparison of framing bias between sources. Built on Longformer with topic injection, it outperforms prior GenAI approaches (Mistral-7B) on the Media Frames Corpus while being 47x smaller.

## Quick Start

```python
from transformers import LongformerForSequenceClassification, LongformerTokenizer
import torch, json

model = LongformerForSequenceClassification.from_pretrained("ry-rousseau/longformer-framing-gold")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

text = "TOPIC:Politics\nHeadline Here\nArticle body text..."
inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)

with torch.no_grad():
    probs = torch.sigmoid(model(**inputs).logits)

# Apply per-class thresholds from optimized_thresholds.json
```

## The 15 Media Frames

Based on the Media Frames Corpus taxonomy (Card et al., 2015):

| Frame | Description |
|-------|-------------|
| Economic | Costs, benefits, financial implications |
| Capacity & Resources | Availability of physical, human, or financial resources |
| Morality | Religious doctrine, ethics, social responsibility |
| Fairness & Equality | Distribution of rights, responsibilities, and resources |
| Legality | Constitutional rights, freedoms, court cases |
| Policy Prescription | Specific policy proposals and evaluations |
| Crime & Punishment | Law enforcement, fraud, sentencing |
| Security & Defense | Threats to welfare of individuals, communities, or nations |
| Health & Safety | Healthcare, disease, public health, mental health |
| Quality of Life | Wealth, happiness, well-being, community impact |
| Cultural Identity | Traditions, customs, values of social groups |
| Public Opinion | Polls, public sentiment, demographics |
| Political | Partisan dynamics, elections, lobbying |
| External Regulation | International reputation, foreign policy, treaties |
| Other | Frames not covered above |

## Architecture

**Model:** `allenai/longformer-base-4096` with topic injection

The model addresses three limitations of existing frame detectors:

1. **Computational cost** -- Prior methods use GenAI prompt engineering (e.g. Mistral-7B), which is expensive to scale. Longformer is 47x smaller.
2. **Context length** -- Frames develop across hundreds of words. Longformer processes up to 2048 tokens (~1500 words) with sparse attention, capturing mid-article framing that truncation-based models miss.
3. **Training data quality** -- A two-phase training pipeline combines the scale of machine-labelled data with the validity of human annotations.

**Key design decisions:**
- Topic prefix injection (`TOPIC:{topic}\n`) provides domain signal as a soft mixture-of-experts, yielding +2.7% Micro F1 with zero inference overhead
- Global attention on `[CLS]` and the first topic token enables cross-document reasoning between the topic signal and article content
- Per-class optimized classification thresholds (range 0.28--0.59)

A lightweight RoBERTa topic classifier (76% accuracy across 19 categories) generates the topic prefix for unseen articles at inference time.

## Training Pipeline

### Phase 1: Silver Training

Trained on ~378,000 Mistral-7B-labelled articles from `copenlu/mm-framing` (Arora et al., 2025). Articles span May 2023 to April 2024 across 28 US news agencies. Class imbalance handled via inverse-frequency weighted BCE loss.

| Parameter | Value |
|-----------|-------|
| Hardware | NVIDIA A40 (48GB), ~72 hours |
| Batch Size | 16 (effective 32 with gradient accumulation) |
| Learning Rate | 2e-5 |
| Epochs | 4 |

### Phase 2: Gold Fine-Tuning

Fine-tuned the silver model on 2,740 human-annotated articles from the Media Frames Corpus (Card et al., 2015) and SemEval 2023 Task 3. Used focal loss (gamma=2) to focus learning on difficult examples, with a plateau-based LR scheduler.

| Source | Articles | Train | Test |
|--------|----------|-------|------|
| Media Frames Corpus | 2,224 | 1,993 | 231 |
| SemEval 2023 Task 3 | 516 | 473 | 43 |
| **Total** | **2,740** | **2,466** | **274** |

## Evaluation

Evaluated on the held-out gold test set (274 articles) with optimized per-class thresholds:

| Metric | Score |
|--------|-------|
| **Weighted F1** | **0.686** |
| Micro F1 | 0.685 |
| Macro F1 | 0.645 |

**Per-class performance:**

| Frame | F1 | Frame | F1 |
|-------|----|-------|----|
| Legality | 0.81 | Health & Safety | 0.68 |
| Political | 0.81 | Quality of Life | 0.64 |
| Economic | 0.74 | Public Opinion | 0.61 |
| Crime & Punishment | 0.73 | Morality | 0.58 |
| Policy Prescription | 0.69 | Security & Defense | 0.58 |
| Cultural Identity | 0.57 | External Regulation | 0.57 |
| Fairness & Equality | 0.56 | Capacity & Resources | 0.46 |

## Repository Structure

| Path | Description |
|------|-------------|
| `docs/final_write_up/` | Full methodology write-up |
| `notebooks/` | Training notebooks (silver and gold phases) |
| `notebooks/final_model_metrics/` | Per-class thresholds and evaluation metrics |
| `scripts/` | Inference scripts |
| `media_frames_corpus/` | MFC gold-standard data |
| `sem_eval_23/` | SemEval 2023 Task 3 data |

## Model Artifacts

- **HuggingFace:** [`ry-rousseau/longformer-framing-gold`](https://huggingface.co/ry-rousseau/longformer-framing-gold)
- **Thresholds:** `notebooks/final_model_metrics/optimized_thresholds.json`

## Citations

```
Card, D., Boydstun, A. E., Gross, J. H., Resnik, P., & Smith, N. A. (2015).
The Media Frames Corpus: Annotations of frames across issues.
ACL-IJCNLP 2015.
```

```
Arora, A. et al. (2025). MM-Framing Dataset. Copenhagen NLU Group.
HuggingFace Datasets: copenlu/mm-framing.
```

## License

Research use only.
