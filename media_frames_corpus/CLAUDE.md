# Media Frames Corpus

Gold standard training data from the original Media Frames Corpus (~2015), hydrated with article text via Nexis Uni.

## Background

The original MFC contains metadata and human annotations but lacks article text (original scraping scripts are defunct). We reconstruct the corpus by batch-downloading articles from Nexis Uni using headline searches, then matching downloaded DOCX files to JSON metadata by normalized title.

**Constraint:** Only New York Times articles are available on Nexis Uni. Other sources in the original corpus cannot be retrieved.

## Dataset Coverage

| Topic | Original MFC | NYT Available | Retrievable | Coverage |
|-------|-------------|---------------|-------------|----------|
| immigration | 5,500 | 1,443 (26.2%) | 1,391 | 25.3% |
| smoking | 5,074 | 822 (16.2%) | 821 | 16.2% |
| samesex | 8,407 | 1,764 (21.0%) | 1,740 | 20.7% |
| **Total** | **18,981** | **4,029 (21.2%)** | **3,952** | **20.8%** |

*Retrievable = NYT articles with valid (non-empty) titles that can be searched*

## Current Status

| Topic | Target | Downloaded | Matched | Status |
|-------|--------|------------|---------|--------|
| immigration | 1,391 | 1,467 | 1,370 (98.5%) | Complete |
| smoking | 821 | - | - | Pending |
| samesex | 1,740 | - | - | Pending |

## Directory Structure

```
media_frames_corpus/
    # Source metadata (original corpus)
    immigration.json, smoking.json, samesex.json
    codes.json

    # Immigration (18 single batches)
    search_queries/
        batch_01.txt ... batch_18.txt
        manifest.json
    downloads/
        batch_01/ ... batch_18/

    # Smoking (6 super-batches, 2 queries each)
    smoking_queries/
        batch_XX_query_a.txt, batch_XX_query_b.txt
    smoking_downloads/
        batch_01/ ... batch_06/

    # Samesex (11 super-batches, 2 queries each)
    samesex_queries/
        batch_XX_query_a.txt, batch_XX_query_b.txt
    samesex_downloads/
        batch_01/ ... batch_11/

    # Output datasets
    {topic}_corpus.parquet
    {topic}_corpus.csv
    {topic}_assembly_report.json
```

## Scripts

```bash
# Generate search queries
python generate_search_queries.py          # immigration (legacy, single batches)
python generate_paired_queries.py smoking  # paired batches
python generate_paired_queries.py samesex

# Assemble corpus from downloads
python assemble_dataset.py immigration
python assemble_dataset.py smoking
python assemble_dataset.py samesex
```

## Nexis Uni Download Protocol

### Search Query Syntax
```
(headline("Title A") OR headline("Title B") OR headline("Title C") ...)
```
Max 5000 characters per query. Filter by Source = "The New York Times".

### Download Settings (Critical)
- **Format:** Microsoft Word (.DOCX)
- **Document Handling:** Separate Files (one file per article)
- **Uncheck all:** Cover Page, Table of Contents, Bold Search Terms, Dual Column

### Workflow

**Immigration (single queries):**
1. Paste `search_queries/batch_XX.txt` into Nexis
2. Filter to NYT, download to `downloads/batch_XX/`

**Smoking/Samesex (paired queries):**
1. Paste `batch_XX_query_a.txt`, select articles
2. Paste `batch_XX_query_b.txt`, select articles (accumulates in tray)
3. Download all to `{topic}_downloads/batch_XX/`

## Matching Strategy

**Linking Key:** Article title (not original corpus IDs like `Immigration1.0-25450`)

Downloaded DOCX filenames default to article titles. Scripts normalize both (lowercase, remove punctuation) for matching. ~98% match rate expected; mismatches occur when Nexis returns slightly different headlines (print vs web edition).

## Output Schema

| Column | Description |
|--------|-------------|
| article_id | Original corpus ID |
| title | Article headline |
| year, month | Publication date |
| source | Always "new york times" |
| byline, section, length | Metadata from original corpus |
| text | Extracted article body |
| text_length | Character count |
| frame_annotations | JSON: `{annotator_id: [frame_codes]}` |
| tone_annotations | JSON: `{annotator_id: [tone_labels]}` |

## Frame Codes

| Code | Frame | Description |
|------|-------|-------------|
| 1 | Economic | Costs, benefits, wages, taxes |
| 2 | Capacity & Resources | Staffing, backlog, infrastructure |
| 3 | Morality | Religious, ethical, social duty |
| 4 | Fairness & Equality | Rights, discrimination |
| 5 | Legality & Constitutionality | Laws, court cases |
| 6 | Policy Prescription | Specific proposals |
| 7 | Crime & Punishment | Enforcement, fraud |
| 8 | Security & Defense | Border security, terrorism |
| 9 | Health & Safety | Disease, sanitation |
| 10 | Quality of Life | Community impact |
| 11 | Cultural Identity | Assimilation, language |
| 12 | Public Opinion | Polls, public sentiment |
| 13 | Political | Partisan, elections |
| 14 | External Regulation | International, treaties |
| 15 | Other | Miscellaneous |

**Tone codes:** 17.xx = Pro, 18.xx = Neutral, 19.xx = Anti

## Known Issues

- **Missing articles:** Some titles no longer exist in Nexis or have changed headlines. Dropped from final dataset.
- **Span indices invalid:** Original `start`/`end` character indices don't align with new text. We use document-level multi-label classification, not span extraction.
- **NYT-only:** ~79% of original MFC is from non-NYT sources unavailable on Nexis Uni.

## JSON Structure Reference

```json
"Immigration1.0-25450": {
    "title": "The Citizenship Surge",
    "year": 2007,
    "month": 11,
    "source": "new york times",
    "annotations": {
        "framing": {
            "annotator1_XX": [
                {"code": 2.0, "start": 362, "end": 408}
            ]
        },
        "tone": {
            "annotator1_XX": [
                {"code": 17.35, "start": 0, "end": 100}
            ]
        }
    }
}
```
