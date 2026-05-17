# Text-as-Data Analysis of Swedish Risk & Vulnerability Analyses

Sciences Po master thesis analyzing Swedish municipal Risk & Vulnerability Analyses (RSA) using text-as-data methods.

## Research Questions

**Main questions:**
1. Why have risk analyses become increasingly adopted by municipalities?
2. Why do municipalities treat security as risk?

**Empirical focus:** What explains the growth in length and scope of municipal risk analyses over time?

## Methodology

Two-stage text-as-data approach:

1. **Bag-of-words analysis** — Term frequencies, persistence, convergence, and actor similarity using a custom risk dictionary aligned with MSB's taxonomy
2. **Sentence-BERT similarity** — Semantic similarity analysis measuring how municipal risk framing converges toward central government (MSB) and prefecture language over time

## Data

- **Municipalities** — ~488 RSA documents (2015–2024)
- **Prefectures** (Länsstyrelsen) — Regional risk analyses
- **MSB** (Central government agency) — National risk assessments

## Repository Structure

```
scripts/
  01_pdf_extraction/       # PDF text extraction
  02_preprocessing/        # Lemmatisation, sentence segmentation
  03_bow_analysis/         # Bag-of-words analysis scripts
  05_ner/                  # Named Entity Recognition
  07_bert_analysis/        # BERT semantic similarity
  dictionaries/            # Risk term dictionaries (3-tier)
```

See `scripts_description.md` for detailed pipeline documentation.

## Key Outputs

- Term-document matrices by risk category
- Risk persistence and dropout analysis
- Actor convergence metrics (Eta²)
- Institutional isomorphism scores (BERT similarity to MSB/prefectures)

## Requirements

- Python 3.10+
- Swedish BERT model (KBLab)
- See `requirements.txt` for dependencies

## License

Code: MIT | Documentation: CC BY 4.0

## Citation

```bibtex
@software{selimovic_rsa_2025,
  author = {Selimovic, Theodor},
  title = {Text-as-Data Analysis of Swedish Risk & Vulnerability Analyses},
  year = {2025},
  institution = {Sciences Po Paris}
}
```

## Author

Theodor Selimovic  
Sciences Po Paris, Master in Public Policy
