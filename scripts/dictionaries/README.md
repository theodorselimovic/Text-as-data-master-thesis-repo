# Risk Dictionaries

Three-tier dictionary structure for Swedish RSA text analysis.

## Tier 1: Individual Risk Terms (`risk_terms.py`)

Base dictionary mapping 100 canonical risk names to their variants (inflections, synonyms).

```python
from scripts.dictionaries import RISK_TERMS

# Example: 'oversvamning' -> ['översvämning', 'översvämningar', 'skyfall', ...]
```

**Keys**: ASCII for programmatic access (e.g., `oversvamning`, `varmebolja`)
**Values**: Swedish characters for text matching (e.g., `översvämning`, `värmebölja`)

## Tier 2: Risk Categories (`risk_categories.py`)

Maps individual risks into MSB's official three-part taxonomy:

| Category | Swedish | Description | Count |
|----------|---------|-------------|-------|
| `nature` | Naturhändelser | Weather, geological, biological, climate | 32 |
| `technical` | Tekniska störningar | Infrastructure failures, accidents | 34 |
| `antagonistic` | Antagonistiska händelser | Cyber, terrorism, military, crime | 25 |
| `other` | Övriga | Economic, social, pollution | 9 |

```python
from scripts.dictionaries import RISK_CATEGORIES, get_category_for_risk

# Get all risks in a category
nature_risks = RISK_CATEGORIES['nature']

# Get category for a specific risk
category = get_category_for_risk('cyberattack')  # -> 'antagonistic'
```

**Methodological basis**: MSB's Nationell risk- och sårbarhetsbedömning (NRSB) structures risk assessment around these three categories, reflecting the source of the threat.

## Tier 3: Extended Dictionary (`risk_extended.py`)

Adds terms for BERT sampling beyond specific risks:

- **Riskfamilj** (68 terms): Risk-related vocabulary following Boholm (2018) - säkerhet, sårbarhet, kris, etc.
- **Probability qualifications** (29 terms): 5-level scale - osannolik to mycket sannolik
- **Consequence qualifications** (38 terms): 5-level scale - försumbar to katastrofal
- **Legitimacy terms** (24 terms): Trust, democracy, social values

```python
from scripts.dictionaries import get_all_sampling_terms, RISKFAMILJ

# Get all 502 unique terms for paragraph filtering
all_terms = get_all_sampling_terms()
```

## Usage

```python
# From project root
from scripts.dictionaries import (
    RISK_TERMS,           # Tier 1
    RISK_CATEGORIES,      # Tier 2
    get_all_sampling_terms,  # Tier 3
)

# Or import specific modules
from scripts.dictionaries.risk_terms import get_canonical_mapping
from scripts.dictionaries.risk_categories import RISK_TO_CATEGORY
from scripts.dictionaries.risk_extended import PROBABILITY_TERMS
```

## Sources

- MSB Riskkatalog (2025)
- MSB Nationell risk- och sårbarhetsbedömning 2025
- EU Civil Protection Knowledge Network
- Boholm (2018): "Risk association: towards a linguistically informed framework"
