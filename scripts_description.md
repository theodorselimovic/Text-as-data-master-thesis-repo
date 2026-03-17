# Scripts Description

Detailed documentation of the pipeline scripts and their outputs.

## Pipeline: Current State and Remaining Work

### 1. PDF Extraction (`01_pdf_extraction/`)

**Main script:** `pdf_reader_enhanced.py`

**Purpose:** Extracts text from PDF files using multiple methods with automatic fallback.

**Extraction chain:** pypdf → pdfplumber → pdfminer → OCR (if `--ocr` enabled)

**OCR preprocessing:** When OCR is triggered and `ocrmypdf` is installed, scanned PDFs are automatically preprocessed before OCR:
- **Deskewing** — corrects skewed scans
- **Cleaning** — removes noise via unpaper
- **Auto-rotation** — corrects page orientation

This improves OCR quality significantly for poorly scanned documents. Preprocessing can be disabled with `--no-preprocess` for speed or debugging.

**Usage:**
```bash
# Standard extraction (no OCR)
python pdf_reader_enhanced.py --input-dir ./pdfs --output-dir ./output

# With OCR for scanned documents (includes preprocessing if ocrmypdf installed)
python pdf_reader_enhanced.py --input-dir ./pdfs --output-dir ./output --ocr

# OCR without preprocessing (faster, lower quality)
python pdf_reader_enhanced.py --input-dir ./pdfs --output-dir ./output --ocr --no-preprocess
```

**File type detection:** Automatically detects standard PDFs vs ZIP archives masquerading as PDFs (containing JPEG scans).

**Output:** `pdf_texts.parquet` with columns `file`, `text`, and optionally `actor`.

### 2. Preprocessing (`02_preprocessing/`)

#### BERT Preprocessing: `preprocessing_bert.py`

**Purpose:** Light preprocessing that produces a clean, sentence-segmented corpus for BERT fine-tuning. Preserves original surface form (no lemmatization, no stopword removal, no lowercasing).

#### BoW Preprocessing: `preprocessing_bow.py`

**Purpose:** Prepares text for bag-of-words analysis with stemming, stopword removal, and n-gram generation.

**Stemming vs Lemmatization:**

The BoW preprocessing uses **stemming** (Snowball Swedish) rather than lemmatization for:

1. **Speed**: Rule-based stemming is ~100x faster than neural lemmatization (~15 min vs ~2.5 hours for 416K sentences)
2. **Consistency**: Same input always produces same output (no model variance)
3. **Dictionary matching**: When both corpus and dictionary are stemmed identically, matching is reliable even if stems are non-words

**N-grams** (bigrams, trigrams) are generated to capture multi-word dictionary phrases. For example, "organiserad brottslighet" becomes tokens: `["organiser", "brottslig", "organiser_brottslig"]`. The dictionary is stemmed the same way, so "organiserad brottslighet" in the dictionary becomes `"organiser_brottslig"` and matches the corpus n-gram.

**Trade-off:** Stemming may over-reduce (merge unrelated words), but for dictionary-based counting this is acceptable since we control both sides.

**Usage:**
```bash
python preprocessing_bow.py \
    --input data/processed/bert_corpus.parquet \
    --output data/processed/bow_corpus_stemmed.parquet

# Adjust n-gram size (default: 3 for trigrams)
python preprocessing_bow.py \
    --input data/processed/bert_corpus.parquet \
    --output data/processed/bow_corpus_stemmed.parquet \
    --max-ngram 2
```

**Output:** Parquet with columns: `tokens` (list of stemmed unigrams + n-grams), `tokens_text` (space-joined), `token_count`.

---

**Main script:** `preprocessing_bert.py`

**What it does:**
1. **OCR artifact cleanup** — mojibake repair (UTF-8→Latin-1 corruption maps), removal of page numbers, separator lines, box-drawing characters, and repeated page headers/footers ("Risk- och sårbarhetsanalys 2023-2025", "Sida X (Y)").
2. **Sentence segmentation with paragraph tracking** — splits text into paragraphs on `\n\n`, then uses Stanza Swedish pipeline with `processors='tokenize'` only (no POS/lemma). Each sentence tracks its `paragraph_id` for downstream filtering.
3. **Quality assessment** — filters artifact sentences (< 3 words, > 300 words, < 50% alphabetic), computes per-document quality score (0.0–1.0), writes JSON quality report alongside the output parquet.

**Output:** Sentence-level parquet with columns: `doc_id`, `municipality`, `year`, `maskad`, `actor_type`, `sentence_id`, `paragraph_id`, `sentence_text`, `word_count`, `doc_quality`.

**Note:** Chapter removal is disabled — use `risk_term_filter.py` for paragraph-level filtering instead, which handles irrelevant content more precisely.

`merge_all_actors.py` merges data across actor types.

#### Quality Audit: `quality_audit.py`

**Purpose:** Identifies semantically garbage sentences that passed basic parsing filters. Detects OCR failures, table fragments, and nonsense text using heuristics and dictionary-based checks. This is an inspection tool — it flags suspicious sentences for manual review without automatically filtering them.

**Heuristics:**
1. **low_dict_coverage** — < 50% of tokens are real Swedish words (builds dictionary from corpus + stopwords)
2. **repetition** — unusual short tokens (1-4 chars) appearing 3+ times (excludes common Swedish words)
3. **letter_spam** — multiple ALL CAPS tokens that aren't known acronyms (whitelist of ~100 Swedish acronyms: MSB, SCB, RSA, etc.)
4. **char_repetition** — same character repeated 3+ times, multiple occurrences (excludes ellipsis, digits, URLs)
5. **mixed_case** — weird capitalization mid-word (e.g., "BEgreppSförklariNG")
6. **short_tokens** — > 50% of tokens are 1-3 character non-words
7. **high_digits** — > 30% of characters are digits

**Usage:**
```bash
python quality_audit.py \
    --input data/processed/bert_corpus.parquet \
    --output results/quality_audit/
```

**Output files in `results/quality_audit/`:**
- `flagged_sentences.csv` — sentences flagged by any heuristic
- `quality_audit_report.json` — summary statistics

**Results (full corpus, 380K sentences):**
- Flagged: 8,948 (2.4%)
- Clean: 371,149 (97.6%)

### 3. Bag-of-Words Analysis (`03_bow_analysis/`)

**Key scripts:**
- `risk_context_analysis.py` — Counts risk terms by category, analyzes qualifications (sannolikhet, konsekvens, risk). Includes lemmatization support.
- `term_document_matrix.py` — Creates term-level and category-level document matrices. Creates both original and lemmatized versions.
- `risk_persistence_analysis.py` — Tracks which risk terms persist/dropout over time for entities with multiple documents. Supports wave-based (municipalities) and year-based (prefectures, MCF) transitions.
- `risk_clustering_analysis.py` — Clusters entities by risk profile using hierarchical clustering per wave.
- `visualize_rsa_results.py` — Generates visualizations for analysis results.
- `generate_analysis_pdf.py` — Combines all persistence and clustering outputs into a single PDF report.

See [BOW Analysis Details](#bow-analysis-details) below for full documentation.

### 4. Sampling (`04_sampling/`)

#### Risk Term Filtering: `risk_term_filter.py`

**Purpose:** Filters the sentence corpus to only include paragraphs containing at least one risk term. This removes methodology, boilerplate, and irrelevant sections while preserving full paragraph context around risk mentions.

**What it does:**
1. Loads sentence corpus with `paragraph_id` column
2. Builds regex pattern from risk terms across 11 categories (including 'riskfamilj' following Boholm 2016)
3. Identifies paragraphs containing ≥N risk terms (default: 1)
4. Keeps all sentences from qualifying paragraphs

**Filtering results (full corpus):**
- Before: 99,053 paragraphs / 380,097 sentences
- After: 26,090 paragraphs / 190,690 sentences (26.3% of paragraphs, 50.2% of sentences retained)

**CLI options:**
- `--categories` — filter by specific risk categories
- `--min-terms` — require multiple risk terms per paragraph (default: 1)

#### Stratified Sampling: `stratified_sample.py`

**Purpose:** Creates reproducible stratified sample for hand-coding theoretical mechanisms.

**What it does:**
1. Two-stage stratified sampling: documents by (actor_type, wave), then sentences within documents
2. Train/test split at document level (prevents data leakage)
3. Outputs CSV files with coding columns for hand-coding

**Sample generated (2025-02-17):**
- 500 sentences from 58 documents
- Train: 338 sentences (67.6%), Test: 162 sentences (32.4%)
- Stratified by actor (MCF/kommun/länsstyrelse) and wave (0-3)

**Output files in `results/sampling/`:**
- `sample_train.csv` — training set for BERT fine-tuning
- `sample_test.csv` — held-out test set for evaluation
- `sample_full.csv` — complete sample
- `sampling_report.json` — metadata and diagnostics

**Coding columns:**
- `mechanism_legitimacy`, `mechanism_functional`, `mechanism_equivalence`, `mechanism_complexity`, `coder_notes`

### 5. Named Entity Recognition (`05_ner/`)

**Main script:** `ner_extraction.py`

**Purpose:** Extracts named entities from the RSA corpus using the Swedish BERT NER model from KBLab (Royal Library of Sweden). Identifies geographic and institutional entities referenced in risk analyses.

**Model:** `KBLab/bert-base-swedish-cased-ner` (Hugging Face)

**Entity types:**
- TME: Time expressions (dates, periods)
- PRS: Personal names
- LOC: Locations (geographic entities)
- EVN: Events
- ORG: Organizations

**Features:**
- Auto-detects device: CUDA > MPS (Apple Silicon) > CPU
- Batch processing with automatic OOM recovery
- Checkpointing for resume capability on long runs
- Merges BERT subword tokens via `aggregation_strategy="simple"`

**Usage:**
```bash
# Full corpus (M1 Mac ~4.5 hours, Colab T4 ~1 hour)
python ner_extraction.py \
    --input data/processed/bert_corpus_filtered.parquet \
    --output results/ner/

# Test on small sample
python ner_extraction.py \
    --input data/processed/bert_corpus_filtered.parquet \
    --output results/ner/ \
    --max-sentences 1000
```

**Output files in `results/ner/`:**
- `entities.csv` — all extracted entities with positions and confidence
- `entities_by_sentence.csv` — entity counts per sentence
- `entities_by_document.csv` — entity counts per document with actor/wave metadata
- `ner_report.json` — summary statistics, top entities by type

### 6. BERT Mechanism Classification (`06_bert_classification/`)

**Main script:** `mechanism_classifier.py`

**Purpose:** Fine-tunes Swedish BERT to classify theoretical mechanisms in RSA sentences. Multi-label classification for two mechanisms:
- **mechanism_legitimacy** — defining parameters of blame / institutional risk management legitimization (Borraz, 2008)
- **mechanism_complexity** — complexity empowerment of local actors

**Model:** `KBLab/bert-base-swedish-cased` (Hugging Face)

**Features:**
- Three modes: `train`, `evaluate`, `predict`
- Multi-label classification with weighted BCE loss for class imbalance
- Auto-detects device: CUDA > MPS (Apple Silicon) > CPU
- Automatic threshold calibration per mechanism
- Saves model checkpoints, training history, and evaluation metrics

**Usage:**
```bash
# Train model
python mechanism_classifier.py --mode train \
    --train-data results/sampling/sample_train.csv \
    --test-data results/sampling/sample_test.csv \
    --output results/bert_classification/ \
    --model-dir models/mechanism_classifier/ \
    --epochs 5 --learning-rate 2e-5

# Evaluate model
python mechanism_classifier.py --mode evaluate \
    --test-data results/sampling/sample_test.csv \
    --model-dir models/mechanism_classifier/ \
    --output results/bert_classification/ \
    --calibrate-thresholds

# Predict on full corpus
python mechanism_classifier.py --mode predict \
    --input data/processed/bert_corpus_filtered.parquet \
    --model-dir models/mechanism_classifier/ \
    --output results/bert_classification/
```

**Input data format:**
- Training/test CSV with columns: `sentence_text`, `mechanism_legitimacy`, `mechanism_complexity`
- Labels: `1` = present, empty = absent (recoded to 0 automatically)
- Metadata columns preserved: `doc_id`, `actor_type`, `year`, `wave`, `sentence_id`

**Output files:**
- `models/mechanism_classifier/` — Model checkpoint, tokenizer, thresholds.json
- `results/bert_classification/training_report.json` — Training metadata and config
- `results/bert_classification/training_history.csv` — Per-epoch loss and F1
- `results/bert_classification/evaluation_report.json` — Per-mechanism metrics, confusion matrices
- `results/bert_classification/predictions.csv` — Full corpus with `prob_*` and `pred_*` columns
- `results/bert_classification/predictions_report.json` — Summary statistics

**Hyperparameters:**
| Parameter | Default | Notes |
|-----------|---------|-------|
| base_model | KBLab/bert-base-swedish-cased | Swedish BERT |
| max_length | 512 | Full BERT context |
| epochs | 5 | Small dataset |
| learning_rate | 2e-5 | Standard for BERT |
| batch_size | 8 (MPS), 16 (CUDA) | Auto-detect |
| gradient_accumulation | 2 | Effective batch = 16-32 |
| warmup_ratio | 0.1 | ~10% of steps |
| threshold | 0.5 | Calibrated per mechanism |

---

## BOW Analysis Details

**Recent improvements (2024-2025):**

1. **Wave mapping** — Years are mapped to waves for longitudinal analysis:
   - Wave 0: pre-2015 (baseline, mostly prefectures)
   - Wave 1: 2015-2018
   - Wave 2: 2019-2022
   - Wave 3: ≥ 2023
   - All matrices include `wave` column in metadata

2. **Stemming (recommended)** — Risk terms are stemmed using Snowball Swedish stemmer to merge inflectional variants:
   - Faster than lemmatization (~15 min vs ~2.5 hours for full corpus)
   - N-grams capture multi-word dictionary phrases (e.g., "organiserad brottslighet" → "organiser_brottslig")
   - Use `--use-stems` flag with `risk_dictionary_counter.py`

   **Legacy lemmatization** — Stanza Swedish pipeline (slower, use `--use-lemmas`):
   - Merges variants like "gräsbrand"/"gräsbränder", "cyberattack"/"cyberattacker"
   - Both original and lemmatized matrices saved (`*_original.csv` and `*.csv`)
   - Lemma mapping saved to JSON for transparency

3. **Low-N flagging** — Persistence metrics flagged when based on small samples:
   - Threshold: 3 entities (configurable via `--min-entities`)
   - Output includes: `n_entities_t0`, `n_entities_persist`, `n_entities_dropout`, `flag_low_n`
   - Prevents misleading persistence rates from single-entity observations

4. **Actor-specific persistence analysis** — Different transition types per actor:
   - **Municipalities**: Wave-based transitions (W0→W1, W1→W2, W2→W3) plus direct W1→W3 comparison
   - **Prefectures (länsstyrelsen)**: Year-by-year transitions (fewer entities, wave grouping less meaningful)
   - **MCF**: Year-by-year transitions (single entity tracked across 4 reports)

5. **Clustering analysis** — Hierarchical clustering of entities by risk profile per wave:
   - Uses category-level term frequencies (10 risk categories)
   - Optimal k determined via silhouette score
   - Tracks cluster transitions between waves

**Output files:**
- `term_document_matrix.csv` / `*_original.csv` — Term counts per document
- `category_document_matrix.csv` / `*_original.csv` — Category counts per document
- `term_metadata.csv` / `*_original.csv` — Term → category mapping
- `lemma_mapping.json` — Lemma → original terms mapping
- `results/persistence/` — Persistence analysis outputs (see below)
- `results/clustering/` — Clustering analysis outputs (see below)
- `results/risk_mapping_analysis_outputs.pdf` — Combined PDF report

See `docs/implementation-wave-lemma-lown.md` for detailed documentation.

## Analysis Results

### Persistence Analysis Results

**Panel:** 162 entities (153 municipalities, 9 prefectures, 1 MCF), 449 documents with ≥2 waves.

**Key findings:**
- Overall persistence rate: **73.8%** (once a risk term enters an RSA, 74% chance it remains)
- Prefectures most stable (75.8%), municipalities similar (73.9%), MCF most volatile (29.6%)
- Mean Jaccard similarity: Prefectures 0.54, Municipalities 0.50, MCF 0.11

**Most persistent terms:** hälsa (97%), dricksvatten (91%), brand (89%), storm (88%), fjärrvärme (87%), pandemi (87%)

**Most frequently dropped terms:** terrorhot (73% dropout), vattenläcka (73%), folkhälsa (70%), influensapandemi (67%)

**Output files in `results/persistence/`:**
- `persistence_heatmap.png` — All actors, consecutive waves
- `persistence_heatmap_kommun.png` — Municipalities W0→W1, W1→W2, W2→W3
- `persistence_heatmap_kommun_w1_w3.png` — Municipalities W1→W3 direct (112 entities)
- `persistence_heatmap_year_länsstyrelse.png` — Prefectures year-by-year (11 entities, 20 year-pairs)
- `persistence_heatmap_year_MCF.png` — MCF year-by-year (10 year-pairs)
- `persistence_transitions.csv` — Raw transition data
- `persistence_by_term.csv` — Aggregated persistence rates per term

### Clustering Analysis Results

**Method:** Hierarchical clustering on category-level risk profiles (10 categories: naturhot, biologiska_hot, olyckor, antagonistiska_hot, cyber_hot, sociala_risker, teknisk_infrastruktur, brand, miljö_klimat, ekonomi).

**Key findings by wave:**

| Wave | Entities | Optimal k | Silhouette | Cluster 0 distinctive | Cluster 1 distinctive |
|------|----------|-----------|------------|----------------------|----------------------|
| W0 (pre-2015) | 21 | 2 | 0.57 | naturhot | antagonistiska_hot (1 outlier) |
| W1 (2015-18) | 126 | 2 | 0.29 | naturhot | teknisk_infrastruktur |
| W2 (2019-22) | 154 | 2 | 0.37 | biologiska_hot | naturhot |
| W3 (2023+) | 201 | 2 | 0.24 | naturhot | teknisk_infrastruktur |

**Cluster transitions:**
- W0→W1: 13% changed cluster
- W1→W2: 69% changed cluster (major shift, possibly COVID-related)
- W2→W3: 59% changed cluster

**Output files in `results/clustering/`:**
- `cluster_assignments.csv` — Entity-wave-cluster mapping
- `clustering_report.txt` — Detailed cluster profiles
- Per-wave visualizations: `elbow_*.png`, `dendrogram_*.png`, `pca_scatter_*.png`, `centroid_heatmap_*.png`, `actor_distribution_*.png`
- `transition_matrix_*.png` — Cluster transition matrices between waves

### Remaining work:
1. **Write the codebook** — defining coding categories for hand-coding the sample.
2. **Hand-code the sample** — code 338 training sentences using the codebook.
3. ~~**BERT fine-tuning script**~~ — **DONE**: `mechanism_classifier.py` with `--mode train`
4. ~~**BERT evaluation script**~~ — **DONE**: `mechanism_classifier.py` with `--mode evaluate`
5. **Results visualisation script** — visualise the final classification results (mechanism prevalence by actor/wave, etc.).

## Language & Key Dependencies

**Python 3** — no `requirements.txt` exists. Key libraries:
- NLP: `stanza` (Swedish lemmatisation)
- PDF: `pypdf`, `pdfplumber`, `pdfminer.six`, `pytesseract`, `ocrmypdf` (optional, for OCR preprocessing)
- Data: `pandas`, `numpy`, `pyarrow`
- ML: `transformers`, `torch` (for BERT fine-tuning)
- Stats: `scipy`, `scikit-learn`
- Viz: `matplotlib`, `seaborn`

External: Tesseract OCR (Swedish), unpaper (for ocrmypdf cleaning), Stanza Swedish model, Swedish BERT model from Hugging Face (Royal Library of Sweden).

## Data Formats

- **Parquet** for large datasets (sentences, metadata)
- **CSV** for metadata, index files, human-readable results, and the hand-coded sample
- **JSON** for processing summaries and configuration

Standard columns: `doc_id`, `municipality`, `year`, `sentence_id`, `sentence_text`, `actor_type`.

**Bag-of-words matrix columns:** `file`, `actor`, `entity`, `year`, `wave` (+ term/category counts).

### Tests

No formal test suite. Testing will be done through manual data inspection at the parsing/preprocessing stage, as well as accuracy/recall/f1 tests at the fine-tuning stage. A potential to do is to implement a formal test suite for the base corpus.