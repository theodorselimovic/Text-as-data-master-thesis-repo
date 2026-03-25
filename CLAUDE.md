# CLAUDE.md

## Project Overview

Sciences Po master thesis by Theodor Selimovic. Text-as-data analysis of Swedish Risk & Vulnerability Analyses (RSA) to study how risk analysis instruments structure politics in the Swedish multi-level polity.

## Research Questions

**Main questions:**
1. What explains the increasing rate of de facto adoption of risk analyses as an instrument of civil defence by different actors in the Swedish multi-level polity?
2. What are the structural effects of the adoption on politics within and without the administration?

**Subquestions:**
1. How have municipal, prefectural, and central government risk analyses in Sweden changed between 2015 and 2024?
2. How does the framing and/or understanding of risk change depending on the actor?
3. What other effects do the instruments produce?
4. How do risk analyses as a particular form of analysing the future change how we see the future?

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
1. **Bag-of-words analysis** — descriptive, exploratory analysis of term frequencies and patterns.
2. **Fine-tuned BERT model** — a Swedish BERT model (from Hugging Face, trained by the Royal Library of Sweden) fine-tuned on a hand-coded sample to classify the presence of theoretical mechanisms, analysed over time and between actors. Currently classifying 2 mechanisms (legitimacy, complexity); plan is to extend to all 4.

Combined with qualitative reading of related documents (e.g. crisis preparedness plans).

**Important:** The project has moved away from static word embeddings (FastText). The old pipeline stages for seed term expansion and sentence filtering/vectorisation are deprecated. The current approach is bag-of-words + BERT.

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
  04_sampling/             # Sampling for hand-coding
    risk_term_filter.py          # Filters corpus to paragraphs containing risk terms
    stratified_sample.py         # Stratified sampling by actor/wave with train/test split
  05_ner/                  # Named Entity Recognition
    ner_extraction.py            # Swedish BERT NER extraction (KBLab model)
  06_bert_classification/  # BERT mechanism classification
    mechanism_classifier.py      # Fine-tune Swedish BERT for mechanism detection
data/                      # Gitignored: raw PDFs, parquet files, vectors
models/                    # Gitignored: trained model checkpoints
results/                   # Gitignored: analysis outputs, visualisations
  persistence/             # Persistence analysis outputs
  clustering/              # Clustering analysis outputs
  term_document_matrix/    # Term-document matrices
  sampling/                # Sampling outputs (train/test CSVs for hand-coding)
  ner/                     # NER extraction outputs
  quality_audit/           # Quality audit outputs
  bert_classification/     # BERT classification outputs (predictions, metrics)
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
