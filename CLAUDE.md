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
    risk_context_analysis.py
    visualize_rsa_results.py
data/                      # Gitignored: raw PDFs, parquet files, vectors
results/                   # Gitignored: analysis outputs, visualisations
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

3. **Bag-of-words analysis** — `03_bow_analysis/` contains initial analysis and visualisation scripts.

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
