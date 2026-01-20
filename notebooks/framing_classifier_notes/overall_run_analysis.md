All metrics below reflect the **Maximum Performance** achieved after applying your optimal classification thresholds (post-processing).

### **1. Global Performance Comparison**

| Metric               | Run 1: Baseline (Generalist) | Run 2: Weighted Loss (Generalist) | Run 3: Politics Subset (Specialist) |
| -------------------- | ---------------------------: | --------------------------------: | ----------------------------------: |
| Micro F1 (Global)    |                        0.731 |                             0.729 |                               0.758 |
| Macro F1 (Average)   |                        0.706 |                             0.706 |                               0.686 |
| Precision (Weighted) |                        0.687 |                 Lower (Trade-off) |                               0.718 |
| Recall (Weighted)    |                        0.784 |                           Highest |                               0.812 |


* **Run 3 (Politics Specialist)** achieved the highest Global Score (Micro F1), mainly by maximizing performance on the high-volume political classes.
* **Run 1 (Baseline)** actually maintained the best balance across all 15 categories (Macro F1), as Run 3 sacrificed performance on non-political edge cases.

---

### **2. Class-Specific Performance (The "Specialist" Effect)**

This breakdown highlights where the "Politics Only" model (Run 3) gained deep contextual understanding versus where the Generalist model (Run 1) remained superior.

#### **A. The "Nuanced" Frames (Real Gains)**

*These frames appear in both general and political contexts. The Specialist model (Run 3) significantly outperformed the Generalist here, proving it understands political nuance better.*

| Frame Class | Run 1 F1 (General) | Run 3 F1 (Specialist) | Improvement |
| --- | --- | --- | --- |
| **Legality & Constitutionality** | 0.783 | **0.864** | **+8.1%** |
| **Fairness & Equality** | 0.703 | **0.789** | **+8.6%** |
| **Public Opinion** | 0.653 | **0.721** | **+6.8%** |
| **Morality** | 0.645 | **0.675** | **+3.0%** |

#### **B. The "Inflated" Frame**

*Performance spiked here because the dataset filtering made this label ubiquitous.*

| Frame Class | Run 1 F1 (General) | Run 3 F1 (Specialist) | Improvement |
| --- | --- | --- | --- |
| **Political** | 0.754 | **0.914** | **+16.0%** |

#### **C. The "Confused" Frames (Performance Drops)**

*The Specialist model struggled with these because they are often defined differently in political contexts vs. general contexts (e.g., "Health" becoming "Health Policy").*

| Frame Class | Run 1 F1 (General) | Run 3 F1 (Specialist) | Drop |
| --- | --- | --- | --- |
| **Health and Safety** | **0.764** | 0.570 | **-19.4%** |
| **Quality of Life** | **0.792** | 0.579 | **-21.3%** |
| **Economic** | **0.788** | 0.715 | **-7.3%** |
| **Cultural Identity** | **0.701** | 0.651 | **-5.0%** |

---

### **3. Threshold Psychology (Run 1 vs Run 3)**

Comparing the optimal thresholds shows how the model's "confidence" shifted when moved to the specialist domain.

* **Political Threshold:** Dropped from **0.50** (Run 1) to **0.10** (Run 3).
* *Insight:* The Specialist model learned it is almost never wrong to guess "Political."


* **Legality Threshold:** Dropped to **0.15** (Run 3).
* *Insight:* The model is extremely confident in identifying legal language within political articles.


* **Health & Safety Threshold:** Spiked to **0.80** (Run 3).
* *Insight:* The model became very conservative, likely confusing "Health" with "Policy" or "Quality of Life," so it only predicts "Health" when absolutely certain.