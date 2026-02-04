# Phase 2: Gold Standard Fine-Tuning Design

## Overview

Fine-tune the Phase 1 silver-trained Longformer on human-annotated gold standard data to correct systematic biases from Mistral's machine labels and improve frame detection accuracy.

**Prerequisite:** Phase 1 silver training completes on RunPod (~24 hours remaining as of 2026-02-03).

## Data Sources

### Included

| Dataset | Size | Format | Status |
|---------|------|--------|--------|
| **Media Frames Corpus (MFC)** | ~3,952 NYT articles | Multi-annotator span-level | Immigration complete (1,370), smoking/samesex pending |
| **SemEval Task 3 Subtask 2** | ~516 articles (EN) | Article-level multi-label | Ready to assemble |

### Excluded

**FrAC** - Deprioritized due to format mismatch:
- Gold data is sentence-level (158 chars avg), not article-level
- Single-label instead of multi-label
- 9 collapsed frames instead of 15
- Would require additional hydration effort for marginal benefit

### Frame Taxonomy Compatibility

MFC and SemEval use the same 14-15 frame taxonomy - direct 1:1 mapping, no translation needed.

## Data Engineering Decisions

### MFC Annotator Aggregation: Union

Each MFC article has 2+ annotators with span-level frame annotations.

| Strategy | Mean Frames/Article | Match to Silver (4.62)? |
|----------|---------------------|-------------------------|
| Per annotator | 3.18 | Too conservative |
| **Union** | **4.28** | Close match |
| Intersection | 2.14 | Way too conservative |

**Decision:** Use union of all annotator frames per article.

**Implementation:**
1. Extract frame codes from each annotator's spans
2. Collapse sub-codes (10.0, 10.1, 10.2 -> 10 = Quality of Life)
3. Union across annotators to get article-level label set

### SemEval Format

Already article-level multi-label annotations. Mean 3.73 frames/article - compatible with silver distribution.

**Implementation:**
1. Parse train/dev label files (TSV format: `article_id\tframe1,frame2,...`)
2. Load article text from corresponding .txt files
3. Map frame names to standard 15-label schema

## Training Strategy

### Approach: Gold-Only Fine-Tuning

Train exclusively on gold data starting from Phase 1 checkpoint.

**Rationale:**
- Phase 1 provides strong foundation (270k silver samples)
- Gold data is corrective, not additive - recalibrating decision boundaries
- Mixing silver would dilute the gold signal and retain Mistral biases
- ~2-4k samples is sufficient for fine-tuning a pre-trained model

### Regularization: Moderate

Want meaningful weight shifts, not just surface adjustments.

- Learning rate: High enough to move weights (not 1e-6, probably 1e-5 to 2e-5 range)
- Training epochs: Short (2-3 epochs) to prevent memorization
- Dropout/weight decay: Not excessive - allow learning

### Validation Strategy

Hold out ~20% of gold data for validation, stratified by source (MFC vs SemEval).

**Purpose:** Monitor overfitting and provide clean signal on gold pattern learning.

## Loss Function: Focal Loss

Use Focal Loss from the start to focus learning on hard examples.

**Formula:** FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

**Hyperparameter sweep:**
- **α (alpha):** Per-class weighting - try uniform vs class-frequency-based
- **γ (gamma):** Focusing parameter - sweep [1, 2, 3, 5]

## Topic Injection

### The Challenge

Phase 1 model uses `TOPIC: {topic}\n{text}` format with 19 consolidated topics. Gold datasets don't have these topics natively.

**MFC topics vs 19-topic taxonomy:**
| MFC Topic | Likely Mapping |
|-----------|----------------|
| immigration | Immigration (exact match) |
| smoking | Health |
| samesex | Social Issues / Legal / Politics (ambiguous) |

**SemEval:** No topic metadata - would need classification.

### Decision: Run Ablation

**Run A:** Gold fine-tuning WITH topic injection (via trained topic classifier)
**Run B:** Gold fine-tuning WITHOUT topic injection

If Run B performs nearly as well, simplifies inference pipeline. If Run A wins, keep topic classifier.

### Topic Classifier Validation

Use MFC's known topic groupings as a mini-test of the topic classifier:
- Immigration articles should predict "Immigration"
- Smoking articles should predict "Health"
- Samesex articles should predict reasonable topics (Social Issues/Legal/Politics)

This validates classifier reliability before using it for topic injection.

## Evaluation

### Metrics

- **Primary:** Micro F1 (slight preference)
- **Secondary:** Macro F1 (ensure balanced frame detection)

### Per-Class Threshold Optimization

Re-run threshold optimization on gold validation set after fine-tuning. Output distributions will differ from Phase 1.

## Workstreams

### Manual Work (User)

1. **Continue MFC hydration** - Download smoking and samesex batches via Nexis Uni
2. **Assemble MFC datasets** - Run assembly scripts for each topic

### Experiments (While Waiting for RunPod)

Can run on base Longformer to validate pipeline and tune hyperparameters:

1. **Topic classifier validation** - Test on MFC immigration articles
2. **Gold data pipeline** - Assemble MFC immigration + SemEval into unified format
3. **Ablation: Topic injection** - Compare with vs without on base Longformer
4. **Focal Loss sweep** - Grid search over α strategies and γ values [1, 2, 3, 5]
5. **Baseline BCE run** - Establish reference metrics for comparison

### Post-RunPod (Phase 1 Complete)

1. Apply validated hyperparameters to Phase 1 checkpoint
2. Run gold fine-tuning with best settings
3. Re-optimize per-class thresholds on gold validation
4. Evaluate on held-out gold test set

## Expected Outcomes

**Conservative estimate:** Micro F1 improvement of 2-5% over Phase 1, with better alignment to human labeling patterns.

**Key insight:** Mistral achieved only 0.50 F1 against gold standard. By fine-tuning on gold data, we can correct systematic errors that were learned during silver training.

## File Artifacts

**Data outputs:**
- `data/gold_combined_train.parquet` - Combined MFC + SemEval training data
- `data/gold_combined_val.parquet` - Validation split

**Model outputs:**
- `notebooks/saved_models/framing_training_runs_longformer/phase2_gold_*/model_ep{N}.bin`
- `notebooks/saved_models/framing_training_runs_longformer/phase2_gold_*/class_thresholds_optimized.json`

## Pipeline Scripts

### Created Scripts

1. **`scripts/db_loading/assemble_gold_data.py`** - Main gold data assembly pipeline
   - Loads MFC and SemEval data
   - Applies union aggregation for MFC annotators
   - Integrates topic classifier for topic injection
   - Creates stratified train/val splits
   - Outputs: `data/gold_combined_train.parquet`, `data/gold_combined_val.parquet`

2. **`scripts/utils/topic_classifier_utils.py`** - Topic classifier utilities
   - Wrapper for trained RoBERTa topic classifier
   - Batch prediction for efficiency
   - MFC validation function to test classifier accuracy

### Existing Scripts

- **`media_frames_corpus/assemble_dataset.py`** - Assembles MFC corpus from DOCX files

## Execution Workflow

### Prerequisites

1. **MFC DOCX downloads complete** (via Nexis Uni manual process)
2. **Phase 1 training complete** (for final fine-tuning, not experiments)

### Step 1: Assemble MFC Corpus

For each topic with downloaded DOCX files:

```bash
cd media_frames_corpus
python assemble_dataset.py immigration
python assemble_dataset.py smoking    # when ready
python assemble_dataset.py samesex    # when ready
```

Outputs: `{topic}_corpus.parquet`, `{topic}_corpus.csv`

### Step 2: Validate Topic Classifier (Optional)

Test classifier accuracy on MFC data with known topics:

```bash
python scripts/utils/topic_classifier_utils.py \
    --validate-mfc media_frames_corpus/immigration_corpus.parquet \
    --mfc-topic immigration
```

### Step 3: Assemble Gold Data

```bash
# Full pipeline with topic classifier
python scripts/db_loading/assemble_gold_data.py

# Or without topic classifier (placeholder topics)
python scripts/db_loading/assemble_gold_data.py --skip-topic-classifier

# Or just one source
python scripts/db_loading/assemble_gold_data.py --mfc-only
python scripts/db_loading/assemble_gold_data.py --semeval-only
```

Outputs: `data/gold_combined_train.parquet`, `data/gold_combined_val.parquet`

### Step 4: Run Experiments

(Training scripts to be created in Phase 2 implementation)

## Open Questions

1. **Smoking/samesex completion timeline** - How long until MFC hydration is complete?
2. **SemEval test set** - Hold back for final evaluation or include in training?
3. **Early stopping criteria** - Fixed epochs or validation-loss-based?
