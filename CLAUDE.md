# CLAUDE.md

## Project Overview

Sciences Po master thesis by Theodor Selimovic. Text-as-data analysis of Swedish Risk & Vulnerability Analyses (RSA) to study how risk analysis instruments structure politics in the Swedish multi-level polity.

## Research Questions

**Main questions:**
1. What explains the increasing rate of de facto adoption of risk analyses as an instrument of civil defence by different actors in the Swedish multi-level polity?
2. What are the structural effects of the adoption on politics within and without the administration?

**Empirical questions (text-as-data):**

- **Q1: Why have municipal risk analyses increased in length?** (main empirical question)

Subquestions enabling Q1:
- **Q2:** What risks matter? What risks are only mentioned, and what risks are analysed?
- **Q3:** How are probability, consequence, and risk linked in the analyses?
- **Q4:** How have risk analyses changed over time? Do risks persist? Have the analyses diverged or converged?
- **Q5:** Who leads? Are risks diffused bottom-up (municipalities lead) or top-down (prefectures and central government agency lead)?
- **Q6:** How do security threats become riskified? How do they interact with previous risk clusters?

## Theoretical Framework

Risk analyses are theorised as instruments with structuring effects (Salamon, 2002; Kassim & Le Galès, 2010; Le Galès, 2011; Balzaq, 2008). Four core mechanisms:

1. **Functional aptness** — the instrument may be genuinely apt for handling social risks (Paul, 2021).
2. **Institutional legitimacy** — allows institutions to manage risks to their own legitimacy by delimiting responsibilities (Borraz, 2008), potentially spiralling via risk colonisation (Beck, 1998; Rothstein et al., 2006).
3. **Spaces of equivalence** — creates commensurability enabling more effective central control in a Foucauldian fashion (Desrosières, 2011; Foucault, 2009; Borraz et al., 2022).
4. **Complexity empowerment** — may complexify the view of the world, empowering local actors. Closely related to resilience discourse. The actual effects of resilience discourse and how it relates to risk and security remains to be seen and is one of the questions to be answered by the thesis.

The main empirical finding so far has been on the **legitimacy** angle. Evidence for the other three mechanisms remains to be found.

## Material

Three categories of documents:
- **Municipalities** — large sample of risk analyses (2015–2024), collected in three waves (~488 documents).
- **Länsstyrelsen** (prefectures, the state's regional representative) — smaller sample, to be expanded.
- **Central government agency** (MSB, in charge of civil defence) — approximately every other year since 2011.

## Methodology

Text-as-data methods in two stages:
1. **Bag-of-words analysis** — descriptive, exploratory analysis of term frequencies and patterns using the risk dictionary.
2. **Sentence-BERT similarity** — Swedish BERT (KBLab) used for semantic similarity analysis, measuring how municipal risk framing converges toward MSB/prefecture language over time (isomorphism analysis).

Combined with qualitative reading of related documents (e.g. crisis preparedness plans).

**Important:** The project has moved away from static word embeddings (FastText) and fine-tuned BERT classification. The current approach is bag-of-words + BERT similarity.

## Repository Structure

```
scripts/
  01_pdf_extraction/       # PDF text extraction (multi-method + OCR fallback)
    pdf_reader_enhanced.py
    document_preview_generator.py
  02_preprocessing/        # Text preprocessing (lemmatisation, sentence segmentation)
    preprocessing_bert.py        # Light preprocessing for BERT (sentence + paragraph segmentation)
    quality_audit.py             # Semantic quality checker for OCR garbage detection
    merge_all_actors.py
  03_bow_analysis/         # Bag-of-words analysis
    risk_context_analysis.py       # Term counting by category
    term_document_matrix.py        # Creates term/category document matrices
    risk_persistence_analysis.py   # Tracks term persistence/dropout over time
    risk_clustering_analysis.py    # Clusters entities by risk profile
    visualize_rsa_results.py       # Generates visualizations
    generate_analysis_pdf.py       # Combines all outputs into single PDF report
  04_sampling/             # (deprecated) Sampling for hand-coding
  05_ner/                  # Named Entity Recognition
    ner_extraction.py            # Swedish BERT NER extraction (KBLab model)
  06_bert_classification/  # (deprecated) BERT mechanism classification
  07_bert_analysis/        # BERT-based semantic analysis
    security_similarity/         # Isomorphism analysis for security risk framing
      isomorphism_analysis.py    # Sentence-BERT similarity to MSB/prefectures
  dictionaries/            # Risk term dictionaries (3-tier structure)
    risk_terms.py                # Tier 1: Individual risks with variants
    risk_categories.py           # Tier 2: MSB taxonomy (nature/technical/antagonistic)
    risk_extended.py             # Tier 3: Extended terms for filtering
data/                      # Gitignored: raw PDFs, parquet files, vectors
models/                    # Gitignored: trained model checkpoints
results/                   # Gitignored: analysis outputs, visualisations
  00_data_preparation/     # Data preparation outputs
  01_bow_analysis/         # Bag-of-words analysis outputs
    actor_similarity/      # Actor similarity matrices, PERMANOVA
    clustering/            # Risk profile clustering
    convergence/           # Eta² convergence drivers
    diffusion/             # Lead-lag adoption analysis
    distinctiveness/       # Actor distinctiveness metrics
    persistence/           # Term persistence/dropout analysis
    prevalence/            # Risk prevalence distributions
    term_matrices/         # Term-document matrices
  02_bert_analysis/        # BERT semantic analysis outputs
    ner/                   # NER extraction outputs
    security_similarity/   # Isomorphism scores, visualizations
  archive/                 # Old/deprecated outputs
docs/                      # Guides and documentation
archive/                   # Legacy notebooks and R scripts
logs/                      # Processing logs
```

The pipeline and a more extensive description of the scripts is in scripts_description.md.

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
- When measuring mentions, especially when creating graphs per actor, mentions should be averaged per document, as otherwise it simply reflects the number of documents.

## Actor Names and Colors

Use these **English names** and **hex colors** consistently across all visualizations:

| Internal name | English display name | Color (hex) |
|---------------|---------------------|-------------|
| `kommun` | Municipality | `#e41a1c` (red) |
| `lansstyrelse` | Prefecture | `#377eb8` (blue) |
| `MCF` | MSB | `#4daf4a` (green) |

## Git & Data Policy

- Large files (parquet, PDFs, `.bin` models, results) are gitignored
- Directory structure preserved via `.gitkeep`
- Dual license: MIT (code), CC BY 4.0 (docs)
