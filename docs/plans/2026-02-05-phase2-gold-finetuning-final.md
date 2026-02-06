# Phase 2: Gold Standard Fine-Tuning - Final Implementation Plan

**Date:** 2026-02-05
**Status:** Ready to Execute
**Approach:** Iterative - baseline first, then advanced experiments

---

## Overview

Fine-tune the Phase 1 silver-trained Longformer on ~4,100 human-annotated gold standard articles to correct systematic biases from Mistral's machine labels.

**Starting Point:** `trained_models/checkpoint_epoch_4/checkpoint_epoch_4/`
**Training Environment:** Local GPU (16GB VRAM)
**Validation Strategy:** 5-Fold Cross-Validation

---

## Critical Constraint: Topic Injection Required

The Phase 1 model was trained with `cls_plus_topic` global attention mode, which applies global attention to position 3 (first token after "TOPIC: "). This is **architecturally baked in** - we cannot remove topic injection without breaking the attention mechanism.

**Implication:** All gold data must use topic prefix format: `TOPIC: {topic}\n{title}\n{text}`

---

## Data Sources

| Dataset | Articles | Topic Strategy | Location |
|---------|----------|----------------|----------|
| MFC Immigration | 1,290 | Direct: "Immigration" | `media_frames_corpus/assembled/immigration_corpus.parquet` |
| MFC Smoking | 737 | Direct: "Health" | `media_frames_corpus/assembled/smoking_corpus.parquet` |
| MFC Samesex | 1,562 | Direct: "Social Issues" | `media_frames_corpus/assembled/samesex_corpus.parquet` |
| SemEval Subtask 2 | 516 | Topic classifier | `sem_eval_23/data/en/` |
| **Total** | **4,105** | | |

---

## Phase 2A: Baseline Experiments

### Task 1: Fix Data Assembly Script

Update `scripts/db_loading/assemble_gold_data.py` line 98:
```python
# Change from:
corpus_path = base_path / "media_frames_corpus" / f"{topic}_corpus.parquet"
# To:
corpus_path = base_path / "media_frames_corpus" / "assembled" / f"{topic}_corpus.parquet"
```

### Task 2: Verify SemEval Subtask-2 Articles

Check existence of:
- `sem_eval_23/data/en/train-articles-subtask-2/`
- `sem_eval_23/data/en/dev-articles-subtask-2/`

If missing, determine if articles are shared with subtask-3 or need extraction.

### Task 3: Run Gold Data Assembly

```bash
conda activate torch-gpu
python scripts/db_loading/assemble_gold_data.py
```

**Outputs:**
- `data/gold_combined_train.parquet`
- `data/gold_combined_val.parquet`
- `data/gold_combined_all.parquet`

### Task 4: Create 5-Fold CV Splits

Create stratified 5-fold splits preserving:
- Source distribution (MFC vs SemEval)
- Label distribution (multi-label stratification)

Save fold indices for reproducibility.

### Task 5: Create Training Notebook

Create `notebooks/gold_finetuning_phase2.ipynb` with:

**Model Loading:**
```python
from transformers import LongformerForSequenceClassification

model = LongformerForSequenceClassification.from_pretrained(
    "trained_models/checkpoint_epoch_4/checkpoint_epoch_4/",
    num_labels=15,
    problem_type="multi_label_classification"
)
```

**Training Configuration:**
| Parameter | Value |
|-----------|-------|
| Loss | Focal Loss (gamma=2 initial) |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Batch Size | 2 |
| Gradient Accumulation | 8 (effective 16) |
| Max Length | 2048 tokens |
| Optimizer | AdamW (weight_decay=0.01) |

**Focal Loss Implementation:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Per-class weights
        self.gamma = gamma  # Focusing parameter

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return focal_loss.mean()
```

### Task 6: Run Baseline Experiments

**Experiment 1: Focal Loss (gamma=2)**
- 5-fold CV
- Record per-fold metrics
- Average Micro F1, Macro F1

**Experiment 2: Standard BCE (comparison)**
- Same 5-fold setup
- Establishes baseline for Focal Loss comparison

**Experiment 3: Focal Loss gamma sweep (if Exp 1 promising)**
- Test gamma = [1, 3, 5]
- Identify optimal focusing strength

### Task 7: Threshold Optimization

Using out-of-fold predictions:
1. Concatenate predictions from all 5 folds
2. Grid search per-class thresholds (0.1 to 0.9, step 0.05)
3. Optimize for F1 per class
4. Save optimized thresholds

### Task 8: Final Model Training

With best hyperparameters:
1. Train on 100% of gold data
2. Use optimized thresholds from Task 7
3. Save final model artifacts

**Output Directory:** `notebooks/saved_models/phase2_gold_final/`

---

## Phase 2B: Advanced Experiments (Iterative)

After baseline results, consider these enhancements:

### Knowledge Distillation from Phase 1

Combined loss to retain Phase 1 knowledge while learning corrections:
```python
L_total = alpha * L_focal_gold + (1-alpha) * L_kl_phase1
# alpha = 0.8 (prioritize gold)
```

**Question to answer:** Does distillation prevent forgetting useful patterns?

### Annotator Agreement Weighting (MFC)

Weight samples by annotation confidence:
- High agreement across annotators = higher loss weight
- Disagreement = lower weight (less certain ground truth)

**Question to answer:** Does weighting improve performance on ambiguous frames?

### Label Smoothing

Soften hard labels:
```python
smoothed = labels * 0.9 + 0.1 / num_classes
```

**Question to answer:** Does smoothing improve calibration?

### Curriculum Learning

Training phases:
1. Epoch 1: High-agreement MFC only
2. Epoch 2: All MFC
3. Epoch 3: Full dataset

**Question to answer:** Does curriculum ordering help convergence?

---

## Evaluation Metrics

**Primary:** Micro F1 (overall accuracy)
**Secondary:** Macro F1 (balanced across classes)

**Per-Class Analysis:**
Track which frames improve most from gold training. Expected gains on frames where Mistral was weakest:
- Fairness (Mistral: 0.28)
- Quality of Life (Mistral: 0.31)
- Regulation (Mistral: 0.34)

---

## Expected Outcomes

| Metric | Phase 1 (Silver) | Phase 2 Target |
|--------|------------------|----------------|
| Micro F1 | 0.755 | 0.78-0.82 |
| Macro F1 | 0.733 | 0.75-0.78 |

---

## File Artifacts

**Data:**
- `data/gold_combined_all.parquet`
- `data/gold_cv_folds.json` (fold indices)

**Models:**
- `notebooks/saved_models/phase2_gold_final/model.safetensors`
- `notebooks/saved_models/phase2_gold_final/config.json`
- `notebooks/saved_models/phase2_gold_final/tokenizer_config.json`
- `notebooks/saved_models/phase2_gold_final/class_thresholds_optimized.json`

**Notebooks:**
- `notebooks/gold_finetuning_phase2.ipynb`

**Experiment Tracking:**
- W&B project: `frame-delta-gold`

---

## Execution Checklist

### Phase 2A: Baseline
- [ ] Task 1: Fix assemble_gold_data.py path
- [ ] Task 2: Verify SemEval subtask-2 articles
- [ ] Task 3: Run gold data assembly
- [ ] Task 4: Create 5-fold CV splits
- [ ] Task 5: Create training notebook
- [ ] Task 6: Run baseline experiments (Focal Loss, BCE, gamma sweep)
- [ ] Task 7: Threshold optimization
- [ ] Task 8: Final model training

### Phase 2B: Advanced (After Baseline)
- [ ] Knowledge distillation experiment
- [ ] Annotator agreement weighting
- [ ] Label smoothing ablation
- [ ] Curriculum learning test

---

## Future Work

### Near-Term
1. **Topic classifier validation:** Test on SemEval articles, measure accuracy
2. **Per-class learning dynamics:** Which frames improve? Which resist?
3. **Calibration analysis:** Are predicted probabilities well-calibrated?

### Medium-Term
1. **Semi-supervised refinement:** Use gold-trained model to clean silver data
2. **Multi-lingual expansion:** Add French, Italian, Polish, Russian, German from SemEval
3. **Active learning:** Identify silver samples most worth human relabeling

### Long-Term
1. **Span-level frame detection:** Use MFC span annotations for finer-grained detection
2. **Real-world deployment:** Live news article comparison (CNN vs Fox)
3. **Temporal analysis:** How do frames shift over time on same topics?

---

## Open Questions

1. **How much does gold training improve rare classes?** Track per-class deltas.
2. **Is the topic classifier reliable on diverse SemEval topics?** Validate before trusting.
3. **What's the optimal gamma for Focal Loss?** Empirical sweep needed.
4. **Does the model overfit on 4k samples?** Monitor train/val gap per fold.
5. **Can we identify systematic Mistral errors?** Compare silver vs gold model predictions.
