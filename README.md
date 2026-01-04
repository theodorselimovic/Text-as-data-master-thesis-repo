# Swedish Risk Analysis Text-as-Data Project

**Analyzing how risk analysis instruments structure politics in Swedish municipal governance**

## 📋 Project Overview

This repository contains code and documentation for analyzing Swedish municipal Risk and Vulnerability Analysis (RSA) documents using text-as-data methods. The project investigates how risk analysis as an instrument structures political discourse through efficiency, accountability, equality, and complexity effects.

### Research Questions

1. **H1**: Which political effects are most prominent in RSA documents?
2. **H2**: Which institutional actors are associated with which effects?
3. **H3**: Does complexity discourse increase over time (2011-2024)?
4. **H4**: Do political effects interact with each other?

### Corpus

- **Documents**: Swedish municipal RSA reports (2011-2024)
- **Municipalities**: ~290 Swedish municipalities
- **Language**: Swedish
- **Total sentences**: ~260,000 lemmatized sentences
- **Filtered sentences**: ~15,000-20,000 containing target concepts

## 🗂️ Repository Structure

```
.
├── README.md                          # This file
├── docs/                              # Documentation
│   ├── METHODOLOGY_SUMMARY.md         # Detailed methodology
│   ├── COMPLETE_PIPELINE_GUIDE.md     # Step-by-step workflow
│   ├── SEED_TERMS_REFERENCE.md        # Seed term documentation
│   └── README_COOCCURRENCE.md         # Chi-square analysis guide
│
├── scripts/                           # Executable scripts
│   ├── 01_ocr/                        # PDF text extraction
│   ├── 02_preprocessing/              # Text cleaning & lemmatization
│   ├── 03_expansion/                  # Seed term expansion
│   ├── 04_filtering/                  # Sentence filtering
│   └── 05_analysis/                   # Statistical analysis
│
├── notebooks/                         # Jupyter notebooks (exploratory)
│
├── data/                              # Data files (gitignored)
│   ├── raw/                          # Original PDFs
│   ├── processed/                    # Lemmatized sentences
│   ├── expanded_terms/               # Expanded seed terms
│   └── vectors/                      # Sentence embeddings
│
└── results/                          # Analysis outputs (gitignored)
    ├── cooccurrence/                 # Chi-square results
    └── figures/                      # Visualizations
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python packages
pip install pandas numpy scipy stanza fasttext matplotlib seaborn pyarrow

# Download Stanza Swedish model (once)
python -c "import stanza; stanza.download('sv')"

# System requirements
# - Tesseract OCR with Swedish language pack (for OCR pipeline)
# - FastText Swedish model (cc.sv.300.bin, ~7GB)
```

### Complete Pipeline

```bash
# 1. Expand seed terms (requires FastText model)
python scripts/03_expansion/vectoranalysis.py \
    --model-path /path/to/cc.sv.300.bin

# 2. Filter sentences with expanded terms
python scripts/04_filtering/sentencefiltering.py

# 3. Check data quality
python scripts/05_analysis/data_diagnostic.py \
    --input data/vectors/sentence_vectors_with_metadata.parquet

# 4. Run co-occurrence analysis
python scripts/05_analysis/cooccurrence_analysis.py \
    --input data/vectors/sentence_vectors_with_metadata.parquet \
    --output-dir results/cooccurrence
```

## 🛠️ Scripts Overview

### OCR Pipeline (`scripts/01_ocr/`)
- `ocr_swedish_pdfs_improved.py` - Main OCR script
- `run_ocr.py` - Simple runner with pre-configured paths

**Purpose**: Extract text from scanned/image-based PDFs that failed standard extraction

### Preprocessing (`scripts/02_preprocessing/`)
- `readingtexts.py` - Text cleaning and lemmatization

**Purpose**: Clean raw text and lemmatize using Stanza Swedish

### Term Expansion (`scripts/03_expansion/`)
- `vectoranalysis.py` - Seed term expansion using FastText

**Purpose**: Expand 40 seed terms to ~1,000-1,500 terms using semantic similarity

### Sentence Filtering (`scripts/04_filtering/`)
- `sentencefiltering.py` - Filter and vectorize sentences

**Purpose**: Create dataset of sentences containing target concepts with embeddings

### Analysis (`scripts/05_analysis/`)
- `cooccurrence_analysis.py` - Chi-square tests
- `data_diagnostic.py` - Data quality checks

**Purpose**: Statistical testing of hypotheses and data validation

## 📝 Documentation

### Essential Reads

1. **`docs/COMPLETE_PIPELINE_GUIDE.md`**
   - Step-by-step workflow
   - Troubleshooting guide
   - Example outputs

2. **`docs/METHODOLOGY_SUMMARY.md`**
   - Detailed methodological decisions
   - Theoretical justifications
   - Limitations and validity

3. **`docs/SEED_TERMS_REFERENCE.md`**
   - Complete seed term documentation
   - Theoretical grounding
   - Expansion methodology

4. **`docs/README_COOCCURRENCE.md`**
   - Chi-square analysis guide
   - Interpretation guidelines
   - Reporting recommendations

## 🎓 Citation
### Methods
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases"
- Grave, E., et al. (2018). "Learning Word Vectors for 157 Languages"
- Qi, P., et al. (2020). "Stanza: A Python Natural Language Processing Toolkit"

## 📊 Data Access

### Input Data
- **RSA PDFs**: Available from Swedish municipalities (public documents)
- **FastText model**: Download from https://fasttext.cc/docs/en/crawl-vectors.html
  - File: `cc.sv.300.bin` (~7GB)

### Output Data (Gitignored)
- Processed datasets available upon request
- Results tables provided in thesis appendix

## 🤝 Contributing

This is a thesis project repository. For questions or collaboration:
- Open an issue for bugs or questions
- Submit pull requests for improvements
- Contact: theoselimovic@gmail.com

## 📜 License

**Code**: MIT License  
**Documentation**: CC BY 4.0  
**Data**: See individual data sources for licensing

## 🔄 Version History

**Version 2.0** (2025-01-02)
- Updated seed terms with refined categories
- Converted notebooks to Python scripts
- Added comprehensive documentation
- Reorganized project structure

**Version 1.0** (2024-12-31)
- Initial pipeline development
- OCR processing
- Basic term expansion

## 📧 Contact

**Project**: Theodor Selimovic Master Thesis
**Institution**: Sciences Po  
**Year**: 2024-2025

---

**Status**: Active Development  
**Last Updated**: 2025-01-04
