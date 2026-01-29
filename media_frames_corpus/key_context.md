# Media Frames Corpus: Reconstruction & Implementation Guide

## 1. Project Overview & Objective
**The Goal:** To resurrect the "Media Frames Corpus" (originally created ~2015) by locating and downloading the raw text for ~4,000+ news articles that are currently only represented by metadata.

**The Context:**
We possess the original metadata (titles, dates, annotation codes) but lack the actual article text. The original scraping scripts (Selenium/Python 2.7) are defunct due to the modernization of the LexisNexis interface. We are solving this by leveraging **Nexis Uni** (the academic successor to LexisNexis) to batch-download the missing texts. This is specific, propriety web-interface software, which is why we'll use a fairly manual approach to get the articles. Only the New York Times is available on this platform, **ignore all other sources**. We only doing this process for the New York Times.  

**The Deliverable:**
A populated directory structure containing individual `.docx` files for every article listed in the three primary datasets (`immigration`, `smoking`, `samesex`), matched and labeled for NLP tasks. The files will have the same titles as those in the json data. The title + some filtering by date will be using to link the metadata with the article text.

## 2. The "Common Key" Strategy
**Crucial Lesson:** We cannot search or link files using the original dataset IDs (e.g., `Immigration1.0-25450`). These IDs are internal to the 2015 project and do not exist in LexisNexis.

* **The Linking Key:** **Article Title**.
    * *Observation:* When files are downloaded from Nexis Uni as "Separate Files," the filename defaults to the Article Title (e.g., `The_Citizenship_Surge.docx`).
    * *Implementation:* We will match the filename of the downloaded `.docx` to the `"title"` field in the JSON metadata.
* **The Match Logic:**
    * `JSON Title`: "The Citizenship Surge"
    * `Download Filename`: "The Citizenship Surge.docx"
    * *Action:* Scripts must normalize these strings (lowercase, remove punctuation) to ensure reliable matching.

---

## 3. Search & Retrieval Protocol (Nexis Uni)

The core task is to feed precise search strings into Nexis Uni to retrieve exact batches of articles.

### A. The Search Query Formula
We do not search by ID. We search by **Headline** and a date range for an entire batch of articles.

* **The "Golden" Syntax:**
    ```text
    (headline("Title A") OR headline("Title B") OR headline("Title C"))
    ```
    + manually set the date range in the Nexis interface + filter for New York Times (this drastically reduces the risk of duplicate results)

* **Batching:** We will experiment with appropriate batch sizes, but I'm able to download 200+ articles at once from Lexis. Main constraint may be the search string. We'll start with large batch sizes and extensive date ranges so we can do this whole task faster. 

### B. Download Specifications (Strict) - THIS IS FOR ME THE HUMAN
To ensure the output files are machine-readable and clean, the following **Download Options** must be used in Nexis Uni:

1.  **Format:** **Microsoft Word (.DOCX)**
    * *Why:* Preserves text flow better than PDF; easier to parse than raw text.
2.  **Document Handling:** **Separate Files** (One document per file).
    * *Why:* Critical. "Single File" merges 100 articles into one document, making separation nearly impossible.
3.  **Page Options (Cleanliness):**
    * [ ] **Cover Page:** **UNCHECK** (Removes junk metadata).
    * [ ] **Table of Contents:** **UNCHECK**.
    * [ ] **Bold Search Terms:** **UNCHECK** (Prevents XML tag corruption in the text).
    * [ ] **Dual Column:** **UNCHECK**.

### C. Scalability
* **Throughput:** Tests confirmed that downloading **200+ files** in a single batch is possible and stable. Likely based on the search string that can be handled by Nexis.
* **Daily Limits:** Be aware of institutional "daily download caps" (often ~1,000–3,000). If hit, may use the "Email to Self" option as a fallback (Human note).

---

## 4. Target Directory Structure
Ultimately we want pandas dataframes that correspond to the 3 json files (which span 3 topic areas), and include all their metadata and the actual article texts. We'll start with immigration in our first implementation. 

---

## 5. Data Dictionary: Understanding the Codes
The JSON files use numerical codes to represent framing. During the "Build Dataset" phase, these must be translated into human-readable labels.

### The JSON Structure
```json
"Immigration1.0-25450": {
    "title": "The Citizenship Surge",
    "year": 2007,
    "annotations": {
        "framing": {
             "annotator1": [ { "code": 2.0, "start": 362, "end": 408 } ]
        },
        "tone": { ... }
    }
}

```

### The Code Map (Global)

*Derived from `codes.json*`

| Code | Frame / Label | Description |
| --- | --- | --- |
| **1.0** | **Economic** | Costs, benefits, wages, taxes. |
| **2.0** | **Capacity & Resources** | Lack of staff, backlog, infrastructure limits. |
| **3.0** | **Morality** | Religious, ethical, or social duty. |
| **4.0** | **Fairness & Equality** | Rights, discrimination, "playing by the rules." |
| **5.0** | **Legality & Constitutionality** | Laws, court cases, rights jurisprudence. |
| **6.0** | **Policy Prescription** | specific proposals, "should do X." |
| **7.0** | **Crime & Punishment** | Enforcement, fraud, prisons. |
| **8.0** | **Security & Defense** | Border security, terrorism, national safety. |
| **9.0** | **Health & Safety** | Disease, sanitation, mental health. |
| **10.0** | **Quality of Life** | Community impact, standard of living. |
| **11.0** | **Cultural Identity** | Assimilation, language, social cohesion. |
| **12.0** | **Public Opinion** | Polls, what "people think." |
| **13.0** | **Political** | Partisan fighting, elections, strategy. |
| **14.0** | **External Regulation** | International reputation, treaties. |
| **15.0** | **Other** | Miscellaneous. |
| **17.xx** | **Pro-Tone** | 17.4 (Explicit), 17.35 (Implicit). |
| **19.xx** | **Anti-Tone** | 19.4 (Explicit), 19.35 (Implicit). |

**You can see the actual codes.json for a more accurate encoding, don't trust this version directly. The point is that these match the encodings in the parent project**
---

## 6. Implementation Plan (Step-by-Step)

### Phase 1: Query Generation (COMPLETE)

**Scripts:** `generate_search_queries.py`

**Output:** `search_queries/` folder containing:
- 19 batch query files (`batch_01.txt` through `batch_19.txt`)
- `manifest.json` tracking file with article IDs per batch

**Stats (Immigration topic, NYT only):**
- 1,443 articles total
- 19 batches (~75 articles each, max 5000 chars per query)
- Year range: 1969-2012

### Phase 2: Human Download Loop

For each batch:

1. Open Nexis Uni
2. Copy contents of `batch_XX.txt` into search box
3. Filter: Source = "The New York Times"
4. Download using settings in Section 3B (DOCX, Separate Files, no extras)
5. Unzip and place files in `downloads/batch_XX/`

**Directory structure:**
```
media_frames_corpus/
    downloads/
        batch_01/
            Article Title.DOCX
            ...
        batch_02/
            ...
```

### Phase 3: Dataset Assembly

**Script:** `assemble_dataset.py`

**Process:**
1. Scans all DOCX files in `downloads/`
2. Matches to JSON entries by normalized title
3. Extracts body text (skips DOCX metadata header)
4. Preserves per-annotator frame and tone labels

**Output:**
- `immigration_corpus.parquet` - main dataset
- `immigration_corpus.csv` - CSV backup
- `assembly_report.json` - match statistics, unmatched files

**DataFrame columns:**
- `article_id`, `title`, `year`, `month`, `source`, `byline`, `section`, `length`
- `text` - extracted article body
- `text_length` - character count
- `docx_file` - source file path
- `match_method` - how title was matched (filename/content)
- `frame_annotations` - JSON dict: `{annotator_id: [frame_codes]}`
- `tone_annotations` - JSON dict: `{annotator_id: [tone_labels]}`


## 7. Known Issues & Workarounds

* **Missing Articles:** Some titles in the 2015 dataset may no longer exist in Nexis Uni or have changed headlines (e.g., "Print Headline" vs "Web Headline").
* *Fix:* These will simply be dropped from the final training set. I don't think this is a very big risk, but possible.

* **Character Alignment:** The specific `start` and `end` character indices in the JSON are **invalid** for the new downloaded text (due to header/formatting differences).
* *Fix:* We treat this as a **Multi-Label Classification** task (tagging the whole document) rather than a Span Extraction task. This is what we do in this frame delta project generally anyways.

