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

#### Document Preview Generator: `document_preview_generator.py`

**Purpose:** Generates human-readable previews of extracted PDF texts for quality inspection. Produces truncated plaintext samples per document to help diagnose extraction failures before running the full preprocessing pipeline.

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

**Note:** Chapter removal is disabled. The PDF extraction stage now handles multi-actor merging directly via the `--merge-with` flag.

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
- `risk_dictionary_counter.py` — Counts risk term occurrences from the centralized dictionary in RSA documents; builds term-level and category-level document matrices from raw text or pre-processed (stemmed/lemmatized) input. Replaces the former `term_document_matrix.py`.
- `risk_context_analysis.py` — Counts risk terms by category, analyzes qualifications (sannolikhet, konsekvens, risk). Includes lemmatization support.
- `risk_persistence_analysis.py` — Tracks which risk terms persist/dropout over time for entities with multiple documents. Supports wave-based (municipalities) and year-based (prefectures, MCF) transitions.
- `risk_prevalence_analysis.py` — Measures which individual risks are most common using mention count and text-devoted metrics, capturing both frequency and depth of discussion per actor/wave.
- `risk_diffusion_analysis.py` — Tracks when risk terms first appear across entities and detects synchronous adoption patterns; tests top-down (MSB → municipalities) vs. bottom-up diffusion hypotheses.
- `risk_convergence_analysis.py` — Identifies which individual risks drive actor convergence across waves using eta-squared variance decomposition.
- `risk_distinctiveness_analysis.py` — Identifies statistically over- or under-represented risk terms per actor type using the Monroe et al. "Fightin' Words" method with informative Dirichlet priors.
- `actor_similarity_analysis.py` — Measures within-group and between-group similarity across actor types using three-level variance decomposition and PERMANOVA testing.
- `allvarliga_storningar_analysis.py` — Analyzes co-occurrence of individual risks with serious-disruption/consequence phrases to reveal how different actors frame risks in terms of operational impact.
- `security_riskification_analysis.py` — Compares analytical language (probability/consequence/risk qualifications) between security risks and other risks to test whether security threats have been absorbed into the standard risk framework (Q6).
- `security_adoption_timing.py` — Computes timing metrics showing security risk concentration by wave; tracks when entities first adopted security risks to support thesis argument about recent absorption.
- `risk_clustering_analysis.py` — Clusters entities by risk profile using hierarchical clustering per wave.
- `visualize_rsa_results.py` — Generates visualizations for analysis results.
- `dictionary_diagnostics.py` — Diagnostic tool: classifies each dictionary term's matching status in the stemmed corpus (stopword conflicts, n-gram length issues, non-appearance).
- `bert_sample_dictionary.py` — Backward-compatibility shim re-exporting centralized dictionary modules; exists only for legacy imports.

See [BOW Analysis Details](#bow-analysis-details) below for full documentation.

### 4. Sampling — ARCHIVED

**Location:** `archive/04_sampling/`

Scripts for creating stratified samples and filtering by risk terms have been archived. The methodology shifted to full-corpus BoW analysis + BERT similarity, making sampling for hand-coding unnecessary.

Archived scripts: `risk_term_filter.py`, `stratified_sample.py`, `recover_corrupted_sample.py`

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
    --input data/processed/bert_corpus.parquet \
    --output results/ner/

# Test on small sample
python ner_extraction.py \
    --input data/processed/bert_corpus.parquet \
    --output results/ner/ \
    --max-sentences 1000
```

**Output files in `results/ner/`:**
- `entities.csv` — all extracted entities with positions and confidence
- `entities_by_sentence.csv` — entity counts per sentence
- `entities_by_document.csv` — entity counts per document with actor/wave metadata
- `ner_report.json` — summary statistics, top entities by type

#### NER Visualization: `plot_ner_over_time.py`

**Purpose:** Creates time-series graphs of unique LOC, ORG, and EVN entity counts by actor type — municipalities grouped by wave, prefectures and MCF by year. Uses `results/ner/entities_by_document.csv` as input.

### Risk Dictionaries (`dictionaries/`)

Centralized three-tier dictionary structure for all risk term detection and categorization.

#### Tier 1: Individual Risk Terms (`risk_terms.py`)

**Purpose:** Base dictionary mapping 100 canonical risk names to their variants (inflections, synonyms). Foundation for all risk detection.

**Structure:**
```python
RISK_TERMS = {
    'oversvamning': ['översvämning', 'översvämningar', 'skyfall', ...],
    'cyberattack': ['cyberattack', 'cyberattacker', 'nätattack', ...],
    ...
}
```

- Keys: ASCII for programmatic access
- Values: Swedish characters for text matching
- 100 canonical risks, 355 total variants

**Sources:** MSB Riskkatalog, MSB NRSB 2025, EU Civil Protection Knowledge Network

#### Tier 2: Risk Categories (`risk_categories.py`)

**Purpose:** Maps individual risks to MSB's official three-part taxonomy:

| Category | Swedish | Description | Count |
|----------|---------|-------------|-------|
| `nature` | Naturhändelser | Weather, geological, biological, climate | 32 |
| `technical` | Tekniska störningar | Infrastructure failures, accidents | 34 |
| `antagonistic` | Antagonistiska händelser | Cyber, terrorism, military, crime | 25 |
| `other` | Övriga | Economic, social, pollution | 9 |

**Legacy support:** `get_legacy_risk_dictionary()` generates the old 12-category format (naturhot, biologiska_hot, olyckor, etc.) for backward compatibility with existing scripts.

#### Tier 3: Extended Dictionary (`risk_extended.py`)

**Purpose:** Adds terms for BERT sampling beyond specific risks:

| Category | Count | Description |
|----------|-------|-------------|
| Riskfamilj | 68 | Risk-related vocabulary (Boholm 2018): säkerhet, sårbarhet, kris... |
| Probability | 29 | 5-level scale: osannolik → mycket sannolik |
| Consequence | 38 | 5-level scale: försumbar → katastrofal |
| Legitimacy | 24 | Trust, democracy, social values |

**Total:** 502 unique terms for paragraph filtering

**Usage:**
```python
# Import from centralized location
from scripts.dictionaries import RISK_TERMS, get_legacy_risk_dictionary

# Tier 1: Individual risks
from scripts.dictionaries import RISK_TERMS, get_canonical_mapping

# Tier 2: Categories  
from scripts.dictionaries import RISK_CATEGORIES, get_category_for_risk

# Tier 3: Extended for BERT sampling
from scripts.dictionaries import get_all_sampling_terms, RISKFAMILJ
```

---

### 6. BERT Mechanism Classification — ARCHIVED

**Location:** `archive/06_bert_classification/`

The BERT fine-tuning approach for mechanism classification has been archived. The project now uses BERT only for similarity analysis (isomorphism), not classification.

Archived scripts: `mechanism_classifier.py`

---

### 7. Isomorphism Analysis (`07_bert_analysis/security_similarity/`)

**Main script:** `isomorphism_analysis.py`

**Purpose:** Measures institutional isomorphism between municipal RSA documents and reference documents (MSB, prefectures) for security risk framing. Tests whether municipalities copy central government framing or adapt discourse to local circumstances.

**Theoretical motivation:** Security risks (cyber, disinformation, military) may show higher isomorphism than other risks because:
- Less local expertise on novel security threats
- More standardized MSB guidance
- Higher perceived uncertainty

**Method:**
1. **Paragraph tagging** — Assigns each paragraph to dominant risk category using `get_legacy_risk_dictionary()` from centralized dictionaries
2. **Sentence embedding** — Extracts embeddings using Swedish Sentence-BERT (`KBLab/sentence-bert-swedish-cased`)
3. **Similarity measures:**
   - **Max-match averaging** — For each municipality sentence, find best match in reference; average maxima. Captures "borrowing" of specific phrases.
   - **Earth Mover's Distance (EMD)** — Optimal transport cost between embedding distributions. Captures distributional similarity.
4. **Baselines:**
   - **Within-document:** Random other-risk paragraph from same RSA (controls for document style)
   - **Cross-municipality:** Same-risk paragraph from different municipality (measures peer similarity)

**Risk categories analyzed:**
- **Security risks:** `cyber_hot`, `antagonistiska_hot` (cyberattacks, terrorism, military threats, disinformation)
- **Comparison risks:** `naturhot`, `teknisk_infrastruktur`, `biologiska_hot`

**Methodology documentation:** See `METHODOLOGY.md` in same folder for detailed thesis-ready explanation.

**Usage:**
```bash
python isomorphism_analysis.py \
    --input data/processed/bert_corpus.parquet \
    --output results/07_bert_analysis/security_similarity/ \
    --min-year 2015 \
    --verbose
```

**Output files:**
- `isomorphism_scores.csv` — Per-municipality similarity scores (1,747 comparisons)
- `example_sentence_pairs.csv` — High/medium/low similarity pairs for verification
- `similarity_distributions.png` — Violin plot: MSB, Prefecture, Within-doc baseline
- `similarity_by_category.png` — Box plots by risk category
- `target_vs_baseline.png` — Bar chart with error bars
- `security_vs_other.png` — Security vs other risk comparison
- `temporal_trends.png` — Trends by wave (2015-18, 2019-22, 2023+)

**Key columns in output:**
| Column | Description |
|--------|-------------|
| `msb_max_match` | Max-match similarity to MSB |
| `msb_emd` | EMD distance to MSB (lower = more similar) |
| `prefecture_max_match` | Max-match similarity to prefecture |
| `prefecture_emd` | EMD distance to prefecture |
| `within_doc_max_match` | Baseline: different risk in same doc |
| `cross_muni_max_match` | Baseline: same risk in other municipality |

**Key findings (preliminary):**
- MSB similarity: mean=0.59, Prefecture: mean=0.60
- Within-doc baseline: mean=0.39
- Gap of ~0.2 confirms municipalities are more similar to MSB/prefecture than to unrelated text in their own documents

**Dependencies:** `sentence-transformers`, `POT` (Python Optimal Transport), `torch`

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

**Output files (from `risk_dictionary_counter.py`):**
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
1. **Interpret BoW results** — synthesize findings from persistence, convergence, distinctiveness analyses.
2. **Interpret isomorphism results** — analyze BERT similarity scores for security risk framing.
3. **Write results chapter** — combine quantitative findings with theoretical framework.

## Language & Key Dependencies

**Python 3** — no `requirements.txt` exists. Key libraries:
- NLP: `stanza` (Swedish lemmatisation)
- PDF: `pypdf`, `pdfplumber`, `pdfminer.six`, `pytesseract`, `ocrmypdf` (optional, for OCR preprocessing)
- Data: `pandas`, `numpy`, `pyarrow`
- ML: `transformers`, `torch`, `sentence-transformers` (for BERT similarity)
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