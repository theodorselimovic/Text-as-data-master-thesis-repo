# CLAUDE.md

## Project Overview

Swedish Risk & Vulnerability Analysis (RSA) text-as-data pipeline for a Sciences Po master thesis. Analyzes ~488 municipal RSA documents (2011–2024) using NLP, word embeddings, and statistical methods to study how risk analysis instruments structure political discourse.

## Repository Structure

```
scripts/
  01_pdf_extraction/   # PDF text extraction (multi-method + OCR fallback)
  02_preprocessing/    # Lemmatization & sentence segmentation (Stanza Swedish)
  03_expansion/        # Seed term expansion via FastText embeddings
  04_filtering/        # Sentence filtering & vectorization
  05_analysis/         # Co-occurrence (chi-square), clustering, diagnostics
data/                  # Gitignored: raw PDFs, parquet files, vectors
results/               # Gitignored: clustering outputs, visualizations
docs/                  # Guides: pipeline, seed terms, co-occurrence
archive/               # Legacy notebooks and R scripts
```

## Language & Key Dependencies

**Python 3** — no `requirements.txt` exists. Key libraries:
- NLP: `stanza`, `fasttext`, `nltk`
- PDF: `pypdf`, `pdfplumber`, `pdfminer.six`, `pytesseract`
- Data: `pandas`, `numpy`, `pyarrow`, `pyreadr`
- Stats/ML: `scipy`, `scikit-learn`, `umap-learn`
- Viz: `matplotlib`, `seaborn`

External: Tesseract OCR (Swedish), FastText model `cc.sv.300.bin` (~7GB), Stanza Swedish model.

## Pipeline Steps

```bash
# 1. Extract text from PDFs
python scripts/01_pdf_extraction/pdf_reader_enhanced.py --input-dir ./pdfs --output-dir ./output [--ocr]

# 2. Preprocess (lemmatize, segment sentences)
python scripts/02_preprocessing/preprocessing.py --input pdf_texts.parquet --output sentences_lemmatized.parquet

# 3. Expand seed terms via FastText
python scripts/03_expansion/vectoranalysis.py --model-path ./cc.sv.300.bin

# 4. Filter sentences & vectorize
python scripts/04_filtering/sentencefiltering.py --expanded-terms expanded_terms.csv --sentences sentences.parquet --model-path ./cc.sv.300.bin --output-dir output_vectors

# 5. Analysis
python scripts/05_analysis/cooccurrence_analysis.py --input sentence_vectors_with_metadata.parquet
python scripts/05_analysis/clustering_analysis.py --vectors vectors.npy --index index.csv --output-dir results/clustering
python scripts/05_analysis/data_diagnostic.py --input sentence_vectors_with_metadata.parquet
```

## No Test Framework

There is no formal test suite. Validation is done via `data_diagnostic.py` and manual inspection of `processing_summary.json` and `failed_files_details.csv`.

## Data Formats

- **Parquet** for large datasets (sentences, vectors with metadata)
- **NumPy `.npy`** for high-dimensional vector arrays
- **CSV** for metadata, index files, and human-readable results
- **JSON** for processing summaries and configuration

Standard columns: `doc_id`, `municipality`, `year`, `sentence_id`, `sentence_text`, `category`, `target_terms`, `vector`.

## Seed Term Categories (v2.0)

Six analytical categories with Swedish terms:
- **Risk** (9 terms): risk, riskanalys, sårbarhet, kritiska, beroenden, krisberedskap, samhällsviktig, ...
- **Accountability** (5): ansvar, åtagande, skyldighet, förpliktelse, ansvarsområde
- **Complexity** (7): komplex, svår, komplicerad, utmaning, otydlig, ...
- **Efficiency** (7): effektiv, effektivering, rationell, nyttig, ändamålsenlig, ...
- **Equality** (5): jämförbar, ekvivalent, motsvarande, likvärdig, utbytbar
- **Agency** (7 institutional actors): kommun, stat, länsstyrelse, region, näringsliv, civilsamhälle, förening

Full reference: `docs/SEED_TERMS_REFERENCE.md` and `docs/SEED_TERMS_UPDATE_V2.md`.

## Code Conventions

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
