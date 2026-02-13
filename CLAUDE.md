# CLAUDE.md

## Project Overview

Sciences Po master thesis by Theodor Selimovic. Text-as-data analysis of Swedish Risk & Vulnerability Analyses (RSA) to study how risk analysis instruments structure politics in the Swedish multi-level polity.

## Research Questions

**Main questions:**
1. What explains the increasing rate of de facto adoption of risk analyses as an instrument of civil defence by different actors in the Swedish multi-level polity?
2. What are the structural effects of the adoption on politics within and without the administration?

**Subquestions:**
1. How have municipal, prefectural, and central government risk analyses in Sweden changed between 2015 and 2024 — in terms of the urgency, severity, probability, and complexity assigned to risks?
2. How does the framing and/or understanding of risk change depending on the actor?
3. What other effects do the instruments produce?
4. How do risk analyses as a particular form of analysing the future change how we see the future?

These subquestions live on two levels of analysis. The **first level** concerns the content of the risk analyses: what they emphasise, how severity estimation has changed, etc. The **second level** concerns risk analyses as an instrument in its entirety — how using risk analyses as an "instrument of the future" (as opposed to scenarios, CBA, or the precautionary principle) structures present decisions and distributes power. The second level supervenes on the first. The plan is to begin with the first level to ensure deliverable results, then address the second.

## Theoretical Framework

Risk analyses are theorised as instruments with structuring effects (Salamon, 2002; Kassim & Le Galès, 2010; Le Galès, 2011; Balzaq, 2008). Four core mechanisms:

1. **Functional aptness** — the instrument may be genuinely apt for handling social risks (Paul, 2021).
2. **Institutional legitimacy** — allows institutions to manage risks to their own legitimacy by delimiting responsibilities (Borraz, 2008), potentially spiralling via risk colonisation (Beck, 1998; Rothstein et al., 2006).
3. **Spaces of equivalence** — creates commensurability enabling more effective central control in a Foucauldian fashion (Desrosières, 2011; Foucault, 2009; Borraz et al., 2022).
4. **Complexity empowerment** — may complexify the view of the world, empowering local actors.

The main empirical finding so far has been on the **legitimacy** angle. Evidence for the other three mechanisms remains to be found.

## Material

Three categories of documents:
- **Municipalities** — large sample of risk analyses (2015–2024), collected in three waves (~488 documents).
- **Länsstyrelsen** (prefectures, the state's regional representative) — smaller sample, to be expanded.
- **Central government agency** (MSB, in charge of civil defence) — approximately every other year since 2011.

## Methodology

Text-as-data methods in two stages:
1. **Bag-of-words analysis** — descriptive, exploratory analysis of term frequencies and patterns.
2. **Fine-tuned BERT model** — a Swedish BERT model (from Hugging Face, trained by the Royal Library of Sweden) fine-tuned on a hand-coded sample to classify the presence of the four theoretical mechanisms, analysed over time and between actors.

Combined with qualitative reading of related documents (e.g. crisis preparedness plans).

**Important:** The project has moved away from static word embeddings (FastText). The old pipeline stages for seed term expansion and sentence filtering/vectorisation are deprecated. The current approach is bag-of-words + BERT.

## Repository Structure

```
scripts/
  01_pdf_extraction/       # PDF text extraction (multi-method + OCR fallback)
    pdf_reader_enhanced.py
    document_preview_generator.py
  02_preprocessing/        # Text preprocessing (lemmatisation, sentence segmentation)
    preprocessing.py
    merge_all_actors.py
  03_bow_analysis/         # Bag-of-words analysis
    risk_context_analysis.py       # Term counting by category
    term_document_matrix.py        # Creates term/category document matrices
    risk_persistence_analysis.py   # Tracks term persistence/dropout over time
    risk_clustering_analysis.py    # Clusters entities by risk profile
    visualize_rsa_results.py       # Generates visualizations
    generate_analysis_pdf.py       # Combines all outputs into single PDF report
data/                      # Gitignored: raw PDFs, parquet files, vectors
results/                   # Gitignored: analysis outputs, visualisations
  persistence/             # Persistence analysis outputs
  clustering/              # Clustering analysis outputs
  term_document_matrix/    # Term-document matrices
docs/                      # Guides and documentation
archive/                   # Legacy notebooks and R scripts
logs/                      # Processing logs
```

## Pipeline: Current State and Remaining Work

1. **PDF extraction** — `01_pdf_extraction/pdf_reader_enhanced.py` extracts text via fallback chain (pypdf → pdfplumber → pdfminer → OCR).

2. **Preprocessing Pipeline (Stage 1)**: `preprocessing_bert.py`

**Purpose:** Light preprocessing that produces a clean, sentence-segmented corpus for BERT fine-tuning. Preserves original surface form (no lemmatization, no stopword removal, no lowercasing).

**What it does:**
1. **OCR artifact cleanup** — mojibake repair (UTF-8→Latin-1 corruption maps), removal of page numbers, separator lines, box-drawing characters, and repeated page headers/footers ("Risk- och sårbarhetsanalys 2023-2025", "Sida X (Y)").
2. **Introductory chapter removal** (kommun only) — detects the Chapter 3 heading ("Identifierad samhällsviktig verksamhet ... inom kommunens geografiska område") via regex and discards Chapters 1-2 (municipality description, methodology boilerplate). Safety guard: skips if match is past 50% of document. On failure: keeps full text and flags in quality report.
3. **Sentence segmentation** — Stanza Swedish pipeline with `processors='tokenize'` only (no POS/lemma). Significantly faster than the full Stanza pipeline.
4. **Quality assessment** — filters artifact sentences (< 3 words, > 300 words, < 50% alphabetic), computes per-document quality score (0.0–1.0), writes JSON quality report alongside the output parquet.

**Output:** Sentence-level parquet with columns: `doc_id`, `municipality`, `year`, `maskad`, `actor_type`, `sentence_id`, `sentence_text`, `word_count`, `doc_quality`.

**Tested on** 4 example RSAs (Bjurholm, Borgholm, Borlänge, Hagfors). Chapter detection works across heading variations. No hyphen-rejoining (too risky with Swedish compound constructions like "risk- och").

**The existing `preprocessing.py`** (with lemmatization + stopwords) will become "Stage 2" for BOW analysis, consuming the output of this script.

`merge_all_actors.py` merges data across actor types.

3. **Bag-of-words analysis** — `03_bow_analysis/` contains term-document matrix creation, risk persistence analysis, and clustering analysis.

**Key scripts:**
- `risk_context_analysis.py` — Counts risk terms by category, analyzes qualifications (sannolikhet, konsekvens, risk). Now includes lemmatization support.
- `term_document_matrix.py` — Creates term-level and category-level document matrices. Creates both original and lemmatized versions.
- `risk_persistence_analysis.py` — Tracks which risk terms persist/dropout over time for entities with multiple documents. Supports wave-based (municipalities) and year-based (prefectures, MCF) transitions.
- `risk_clustering_analysis.py` — Clusters entities by risk profile using hierarchical clustering per wave.
- `visualize_rsa_results.py` — Generates visualizations for analysis results.
- `generate_analysis_pdf.py` — Combines all persistence and clustering outputs into a single PDF report (`results/risk_mapping_analysis_outputs.pdf`).

**Recent improvements (2024-2025):**

1. **Wave mapping** — Years are mapped to waves for longitudinal analysis:
   - Wave 0: pre-2015 (baseline, mostly prefectures)
   - Wave 1: 2015-2018
   - Wave 2: 2019-2022
   - Wave 3: ≥ 2023
   - All matrices include `wave` column in metadata

2. **Lemmatization** — Risk terms are lemmatized using Stanza Swedish pipeline to merge inflectional variants:
   - Merges variants like "gräsbrand"/"gräsbränder", "cyberattack"/"cyberattacker"
   - Both original and lemmatized matrices saved (`*_original.csv` and `*.csv`)
   - Lemma mapping saved to JSON for transparency
   - Token-by-token lemmatization handles multi-word terms correctly

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

### Remaining code to write:
1. **Rewrite preprocessing** — possibly split into two scripts: one for bag-of-words preprocessing, one for transformer preprocessing.
2. **Refine bag-of-words analysis** in `03_bow_analysis/`.
3. **Write the codebook** — defining coding categories for hand-coding the sample.
4. **Sampling script** — sample the corpus, split into training set and testing set (70/30). Must produce the same sample each time for reproducibility.
5. **BERT fine-tuning script** — fine-tune a Swedish BERT model (Royal Library of Sweden, via Hugging Face) on the hand-coded training set.
6. **BERT evaluation script** — test the fine-tuned model on the held-out testing set.
7. **Results visualisation script** — visualise the final results.

## Language & Key Dependencies

**Python 3** — no `requirements.txt` exists. Key libraries:
- NLP: `stanza` (Swedish lemmatisation)
- PDF: `pypdf`, `pdfplumber`, `pdfminer.six`, `pytesseract`
- Data: `pandas`, `numpy`, `pyarrow`
- ML: `transformers`, `torch` (for BERT fine-tuning)
- Stats: `scipy`, `scikit-learn`
- Viz: `matplotlib`, `seaborn`

External: Tesseract OCR (Swedish), Stanza Swedish model, Swedish BERT model from Hugging Face (Royal Library of Sweden).

## Data Formats

- **Parquet** for large datasets (sentences, metadata)
- **CSV** for metadata, index files, human-readable results, and the hand-coded sample
- **JSON** for processing summaries and configuration

Standard columns: `doc_id`, `municipality`, `year`, `sentence_id`, `sentence_text`, `actor_type`.

**Bag-of-words matrix columns:** `file`, `actor`, `entity`, `year`, `wave` (+ term/category counts).

## Tests

No formal test suite. Testing will be done through manual data inspection at the parsing/preprocessing stage, as well as accuracy/recall/f1 tests at the fine-tuning stage. A potential to do is to implement a formal test suite for the base corpus.

## Code Conventions
You are an expert Python and R tidyverse programmer tasked with writing, analysing, and improving code. 

When you analyse and write code, you start by breaking down the problem into its constituent parts. When attempting to write code, consider the following aspects: 
- Code structure and organisation
- Naming conventions and readability
- Potential bugs and errors 
- Adherence to python best practices and the PEP 8 guidelines 
- Use of appropriate data structure and algorithms 
- Error handling and edge cases 
- Modularity and resusability 
- Comments and documentation.

More concretely:
- One script per pipeline stage; each is independently runnable
- Module-level docstrings with usage examples
- Section separators: `# ====...====`
- Type hints in function signatures
- Logging with optional `--verbose` flag
- Graceful fallback chains (PDF extraction tries pypdf → pdfplumber → pdfminer → OCR)
- RSA filename pattern: `RSA [Municipality] [Year] [Maskad].pdf`

## Git & Data Policy

- Large files (parquet, PDFs, `.bin` models, results) are gitignored
- Directory structure preserved via `.gitkeep`
- Dual license: MIT (code), CC BY 4.0 (docs)

## Key References

- Balzaq (2008) — policy tools of securitisation
- Beck (1998) — politics of risk society
- Borraz (2008) — institutional risk and legitimacy
- Borraz et al. (2022) — regulatory style and risk-based inspections
- Desrosières (2011) — politics of large numbers / spaces of equivalence
- Foucault (2009) — security, territory, population
- Kassim & Le Galès (2010) — governance and policy instruments
- Le Galès (2011) — policy instruments and governance
- Paul (2021) — varieties of risk analysis in public administrations
- Rothstein et al. (2006) — risk colonisation theory
- Salamon (2002) — tools of government
