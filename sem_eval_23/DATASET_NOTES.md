# SemEval 2023 Task 3 - Dataset Notes for Frame Classification

## Overview

SemEval 2023 Task 3 focuses on detecting news genre, framing, and persuasion techniques. Only **Subtask 2** is relevant for frame classification.

| Subtask | Task | Level | Use for Frames? |
|---------|------|-------|-----------------|
| 1 | Genre classification (opinion/reporting/satire) | Article | No |
| **2** | **Frame classification (14 MFC labels)** | **Article** | **Yes** |
| 3 | Persuasion technique detection (19 propaganda tactics) | Paragraph | No |

## Subtask 2 - Frame Classification

### Data Statistics (English)

| Split | Articles | Labeled |
|-------|----------|---------|
| Train | 433 | 433 |
| Dev | 83 | 83 |
| Test | 54 | 0 (no public labels) |
| **Total** | **570** | **516** |

### Frame Labels

SemEval uses the standard 14 Media Frames Corpus labels (no "Other" category):

| SemEval Format | MFC Standard Format |
|----------------|---------------------|
| Economic | Economic |
| Capacity_and_resources | Capacity and resources |
| Morality | Morality |
| Fairness_and_equality | Fairness and equality |
| Legality_Constitutionality_and_jurisprudence | Legality, constitutionality and jurisprudence |
| Policy_prescription_and_evaluation | Policy prescription and evaluation |
| Crime_and_punishment | Crime and punishment |
| Security_and_defense | Security and defense |
| Health_and_safety | Health and safety |
| Quality_of_life | Quality of life |
| Cultural_identity | Cultural identity |
| Public_opinion | Public opinion |
| Political | Political |
| External_regulation_and_reputation | External regulation and reputation |

### Frame Distribution (Train + Dev)

```
Political: 317
Legality, constitutionality and jurisprudence: 265
Crime and punishment: 262
Morality: 219
External regulation and reputation: 198
Security and defense: 197
Policy prescription and evaluation: 126
Fairness and equality: 123
Quality of life: 98
Health and safety: 64
Public opinion: 52
Economic: 44
Capacity and resources: 37
Cultural identity: 33
```

### Data Format

- Articles are stored as individual `.txt` files: `article{ID}.txt`
- Line 1: Title
- Line 2: Empty
- Line 3+: Article content
- Labels file: Tab-separated `article_id\tframe1,frame2,...,frameN`

## Database Schema

The `semeval_subtask2` table stores:
- `article_id`: Original SemEval article ID
- `title`: Article headline
- `text`: Full article text
- `frames_raw`: Original SemEval frame names (with underscores)
- `frames_mfc`: Normalized MFC frame names (with spaces)
- `split`: train/dev/test

## Why Not Subtask 3?

Subtask 3 detects **persuasion/propaganda techniques** at paragraph level, such as:
- Appeal_to_Authority
- Loaded_Language
- Name_Calling/Labeling
- Flag_Waving
- Whataboutism
- False_Dilemma-No_Choice

These are rhetorical/manipulation techniques, not semantic frames. They cannot be mapped to MFC categories.

## Other Languages Available

| Language | Labeled Articles |
|----------|------------------|
| Italian | 303 |
| French | 211 |
| Polish | 194 |
| Russian | 191 |
| German | 177 |

These could be used for multilingual training if needed.

## Citation

```bibtex
@inproceedings{semeval2023task3,
  title={SemEval-2023 Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup},
  author={Piskorski, Jakub and Stefanovitch, Nicolas and Da San Martino, Giovanni and Nakov, Preslav},
  booktitle={Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)},
  year={2023}
}
```
