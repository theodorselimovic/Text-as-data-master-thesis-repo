"""
Risk Dictionaries for Swedish RSA Analysis

Three-tier structure:
- Tier 1 (risk_terms): Individual risks with variants
- Tier 2 (risk_categories): MSB taxonomy mapping (nature/technical/antagonistic/other)
- Tier 3 (risk_extended): Extended terms for BERT sampling (riskfamilj, qualifications)

Usage:
    # Tier 1: Individual risk terms
    from scripts.dictionaries import RISK_TERMS, get_canonical_mapping

    # Tier 2: Categories
    from scripts.dictionaries import RISK_CATEGORIES, get_category_for_risk

    # Tier 3: Extended for sampling
    from scripts.dictionaries import get_all_sampling_terms, RISKFAMILJ

    # Or import specific modules
    from scripts.dictionaries.risk_terms import RISK_TERMS
    from scripts.dictionaries.risk_categories import RISK_TO_CATEGORY
    from scripts.dictionaries.risk_extended import PROBABILITY_TERMS
"""

# Re-export commonly used items
from .risk_terms import RISK_TERMS, get_canonical_mapping, get_all_variants
from .risk_categories import (
    RISK_CATEGORIES,
    RISK_TO_CATEGORY,
    CATEGORY_NAMES,
    get_category_for_risk,
    get_risks_for_category,
    # Legacy support
    LEGACY_RISK_TO_CATEGORY,
    LEGACY_CATEGORY_NAMES,
    get_legacy_risk_dictionary,
)
from .risk_extended import (
    get_all_sampling_terms,
    RISKFAMILJ,
    PROBABILITY_TERMS,
    CONSEQUENCE_TERMS,
    LEGITIMACY_TERMS,
)

__all__ = [
    # Tier 1
    'RISK_TERMS',
    'get_canonical_mapping',
    'get_all_variants',
    # Tier 2
    'RISK_CATEGORIES',
    'RISK_TO_CATEGORY',
    'CATEGORY_NAMES',
    'get_category_for_risk',
    'get_risks_for_category',
    # Tier 2 - Legacy support
    'LEGACY_RISK_TO_CATEGORY',
    'LEGACY_CATEGORY_NAMES',
    'get_legacy_risk_dictionary',
    # Tier 3
    'get_all_sampling_terms',
    'RISKFAMILJ',
    'PROBABILITY_TERMS',
    'CONSEQUENCE_TERMS',
    'LEGITIMACY_TERMS',
]
