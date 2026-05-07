#!/usr/bin/env python3
"""
Tier 3: Extended Risk Dictionary

Builds on Tier 1 (individual risks) and Tier 2 (categories) by adding:
- Riskfamilj: Risk-related vocabulary following Boholm (2018)
- Probability qualifications: 5-level scale terms
- Consequence qualifications: 5-level scale terms
- Legitimacy terms: Trust, democracy, social values

This extended dictionary is used for BERT sampling - filtering paragraphs
that discuss risk in any form, not just specific risk events.

Sources:
    - Boholm (2018): "Risk association: towards a linguistically informed
      framework for analysing risk in discourse"
    - Swedish RSA practice: sannolikhet/konsekvens scales

Usage:
    from scripts.dictionaries.risk_extended import (
        get_all_sampling_terms,
        RISKFAMILJ,
        PROBABILITY_TERMS,
        CONSEQUENCE_TERMS,
    )

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-04-06
"""

from typing import Dict, List, Set

# Handle both package imports and standalone execution
try:
    from .risk_terms import RISK_TERMS, get_all_variants
except ImportError:
    from risk_terms import RISK_TERMS, get_all_variants

# =============================================================================
# RISKFAMILJ - Risk-related vocabulary (Boholm 2018)
# =============================================================================

RISKFAMILJ: List[str] = [
    # Core risk family (Boholm 2018)
    'säkerhet', 'säker', 'osäker', 'osäkerhet',
    'riskabel', 'riskerar', 'riskera',
    'fara', 'farlig', 'farligt', 'faror',
    'hot', 'hota', 'hotar', 'hotbild',
    'sårbar', 'sårbarhet', 'sårbarheter',
    'exponering', 'exponerad', 'exponeras',
    'känslig', 'känslighet',
    'utsatt', 'utsatthet',

    # Preparedness/defense
    'beredskap', 'krisberedskap', 'civilt försvar',
    'försvar', 'skydd', 'skydda',
    'förebygga', 'förebyggande', 'prevention',
    'motståndskraft', 'resiliens',

    # Assessment/oversight
    'granskning', 'granska',
    'bevakning', 'bevaka',
    'tillsyn', 'kontroll',
    'bedömning', 'bedöma', 'värdering',
    'analys', 'analysera',

    # Impact
    'drabba', 'drabbad', 'drabbas',
    'påverka', 'påverkan', 'påverkad',
    'skada', 'skador', 'skadlig',

    # Crisis/disaster
    'haveri', 'haverier',
    'katastrof', 'katastrofer', 'katastrofal',
    'kris', 'kriser', 'krishantering',
    'nödläge', 'nödsituation',
    'olycka', 'olyckor',
]


# =============================================================================
# PROBABILITY QUALIFICATIONS (Swedish: sannolikhet)
# =============================================================================

PROBABILITY_TERMS: Dict[str, List[str]] = {
    'very_low': [
        'mycket låg sannolikhet', 'mycket liten sannolikhet',
        'osannolik', 'osannolikt', 'sällsynt', 'sällsynta',
    ],
    'low': [
        'låg sannolikhet', 'liten sannolikhet',
        'mindre sannolik', 'mindre sannolikt', 'tvivelaktig',
    ],
    'medium': [
        'medelhög sannolikhet', 'mellan sannolikhet',
        'möjlig', 'möjligt', 'möjliga',
    ],
    'high': [
        'hög sannolikhet', 'stor sannolikhet',
        'sannolik', 'sannolikt', 'sannolika',
        'trolig', 'troligt', 'troliga',
    ],
    'very_high': [
        'mycket hög sannolikhet', 'mycket stor sannolikhet',
        'mycket sannolik', 'mycket sannolikt',
        'mycket trolig', 'mycket troligt',
    ],
}


# =============================================================================
# CONSEQUENCE QUALIFICATIONS (Swedish: konsekvens)
# =============================================================================

CONSEQUENCE_TERMS: Dict[str, List[str]] = {
    'very_low': [
        'mycket begränsade konsekvenser', 'mycket liten konsekvens',
        'försumbar', 'försumbara', 'obetydlig', 'obetydliga',
    ],
    'low': [
        'begränsade konsekvenser', 'lindriga konsekvenser',
        'liten konsekvens', 'små konsekvenser',
        'lindrig', 'lindriga', 'begränsad', 'begränsade',
    ],
    'medium': [
        'kännbara konsekvenser', 'måttliga konsekvenser',
        'kännbar', 'kännbara', 'måttlig', 'måttliga',
    ],
    'high': [
        'allvarliga konsekvenser', 'betydande konsekvenser',
        'stora konsekvenser', 'omfattande konsekvenser',
        'allvarlig', 'allvarliga', 'betydande', 'omfattande',
        'svåra konsekvenser', 'svår', 'svåra',
    ],
    'very_high': [
        'mycket allvarliga konsekvenser', 'mycket stora konsekvenser',
        'katastrofala konsekvenser', 'förödande konsekvenser',
        'katastrofal', 'katastrofala', 'förödande',
    ],
}


# =============================================================================
# LEGITIMACY RISK TERMS
# =============================================================================

LEGITIMACY_TERMS: List[str] = [
    # Trust/confidence
    'tillit', 'förtroende', 'misstro', 'misstroende',
    'trovärdighet', 'legitimitet',

    # Social values
    'samhällsvärden', 'värdesystem', 'grundvärden',
    'demokrati', 'demokratisk', 'rättsstat',

    # Social unrest
    'social oro', 'sociala oroligheter',
    'civila oroligheter', 'upplopp',
    'polarisering', 'splittring',

    # Institutional
    'ansvar', 'ansvarsutkrävande', 'ansvarsfördelning',
    'myndighet', 'myndigheter', 'myndigheters',
]


# =============================================================================
# COMBINED SAMPLING DICTIONARY
# =============================================================================

def _flatten_qualification_dict(qual_dict: Dict[str, List[str]]) -> List[str]:
    """Flatten a qualification dictionary to a list of terms."""
    terms = []
    for level_terms in qual_dict.values():
        terms.extend(level_terms)
    return terms


def get_all_sampling_terms() -> Set[str]:
    """
    Return flat set of all terms for BERT paragraph filtering.

    Includes:
    - All individual risk terms (canonical + variants) from Tier 1
    - Riskfamilj terms
    - Probability/consequence qualifications
    - Legitimacy terms
    """
    all_terms = set()

    # Individual risks from Tier 1
    for variant in get_all_variants():
        all_terms.add(variant.lower())

    # Riskfamilj
    all_terms.update(t.lower() for t in RISKFAMILJ)

    # Qualifications
    all_terms.update(t.lower() for t in _flatten_qualification_dict(PROBABILITY_TERMS))
    all_terms.update(t.lower() for t in _flatten_qualification_dict(CONSEQUENCE_TERMS))

    # Legitimacy
    all_terms.update(t.lower() for t in LEGITIMACY_TERMS)

    return all_terms


def get_flat_probability_terms() -> List[str]:
    """Return flattened list of all probability terms."""
    return _flatten_qualification_dict(PROBABILITY_TERMS)


def get_flat_consequence_terms() -> List[str]:
    """Return flattened list of all consequence terms."""
    return _flatten_qualification_dict(CONSEQUENCE_TERMS)


def count_terms() -> dict:
    """Return term counts by source."""
    risk_count = sum(len(v) for v in RISK_TERMS.values())

    return {
        'individual_risks': risk_count,
        'individual_risks_canonical': len(RISK_TERMS),
        'riskfamilj': len(RISKFAMILJ),
        'probability': len(get_flat_probability_terms()),
        'consequence': len(get_flat_consequence_terms()),
        'legitimacy': len(LEGITIMACY_TERMS),
        'total_unique': len(get_all_sampling_terms()),
    }


if __name__ == '__main__':
    counts = count_terms()
    print("Tier 3: Extended Risk Dictionary (BERT Sampling)")
    print("=" * 60)
    print(f"  Individual risks (canonical): {counts['individual_risks_canonical']}")
    print(f"  Individual risks (with variants): {counts['individual_risks']}")
    print(f"  Riskfamilj (Boholm 2018): {counts['riskfamilj']}")
    print(f"  Probability qualifications: {counts['probability']}")
    print(f"  Consequence qualifications: {counts['consequence']}")
    print(f"  Legitimacy terms: {counts['legitimacy']}")
    print("-" * 60)
    print(f"  Total unique terms: {counts['total_unique']}")
