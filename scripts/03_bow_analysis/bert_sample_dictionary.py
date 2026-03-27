#!/usr/bin/env python3
"""
BERT Sampling Dictionary

Dictionary for filtering paragraphs to include in BERT training/test samples.
Combines multiple sources to capture risk-relevant text:

1. Individual risk dictionary (101 canonical risks from MSB/EU taxonomies)
2. Riskfamilj - risk-related vocabulary following Boholm (2018)
3. Probability/Consequence qualifications (5-level scales)
4. Legitimacy risk terms

Sources:
- Individual risks: MSB Riskkatalog, MSB NRSB 2025, EU Civil Protection
- Riskfamilj: Boholm (2018) "Risk association: towards a linguistically
  informed framework for analysing risk in discourse"
- Qualifications: Swedish RSA practice (sannolikhet/konsekvens scales)

Usage:
    from bert_sample_dictionary import (
        BERT_SAMPLE_DICTIONARY,
        get_all_sample_terms,
        get_flat_risk_terms,
    )

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-03-25
"""

from typing import Dict, List, Set

# Import individual risk dictionary
from risk_dictionary_individual import RISK_DICTIONARY_INDIVIDUAL

# =============================================================================
# RISKFAMILJ - Risk-related vocabulary (Boholm 2018)
# =============================================================================

RISKFAMILJ = [
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

PROBABILITY_TERMS = {
    'very_low': [
        'mycket låg sannolikhet', 'mycket liten sannolikhet',
        'osannolik', 'osannolikt', 'sällsynt', 'sällsynta',
    ],
    'low': [
        'låg sannolikhet', 'liten sannolikhet',
        'mindre sannolik', 'mindre sannolikt',
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

CONSEQUENCE_TERMS = {
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

LEGITIMACY_TERMS = [
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
# COMBINED BERT SAMPLE DICTIONARY
# =============================================================================

def _flatten_qualification_dict(qual_dict: Dict[str, List[str]]) -> List[str]:
    """Flatten a qualification dictionary to a list of terms."""
    terms = []
    for level_terms in qual_dict.values():
        terms.extend(level_terms)
    return terms


BERT_SAMPLE_DICTIONARY = {
    # Individual risks (imported)
    'individual_risks': RISK_DICTIONARY_INDIVIDUAL,

    # Additional sampling categories
    'riskfamilj': RISKFAMILJ,
    'probability': _flatten_qualification_dict(PROBABILITY_TERMS),
    'consequence': _flatten_qualification_dict(CONSEQUENCE_TERMS),
    'legitimacy': LEGITIMACY_TERMS,
}

# Structured qualification dictionaries (for detailed analysis)
QUALIFICATION_SCALES = {
    'probability': PROBABILITY_TERMS,
    'consequence': CONSEQUENCE_TERMS,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_sample_terms() -> Set[str]:
    """
    Return flat set of all terms for paragraph filtering.

    Includes:
    - All individual risk terms (canonical + variants)
    - Riskfamilj terms
    - Probability/consequence qualifications
    - Legitimacy terms
    """
    all_terms = set()

    # Individual risks (dict of canonical -> variants)
    for canonical, variants in RISK_DICTIONARY_INDIVIDUAL.items():
        all_terms.add(canonical.lower())
        for v in variants:
            all_terms.add(v.lower())

    # Other categories (flat lists)
    all_terms.update(t.lower() for t in RISKFAMILJ)
    all_terms.update(t.lower() for t in _flatten_qualification_dict(PROBABILITY_TERMS))
    all_terms.update(t.lower() for t in _flatten_qualification_dict(CONSEQUENCE_TERMS))
    all_terms.update(t.lower() for t in LEGITIMACY_TERMS)

    return all_terms


def get_flat_risk_terms() -> Set[str]:
    """Return only individual risk terms (no qualifications/riskfamilj)."""
    terms = set()
    for canonical, variants in RISK_DICTIONARY_INDIVIDUAL.items():
        terms.add(canonical.lower())
        for v in variants:
            terms.add(v.lower())
    return terms


def count_terms() -> dict:
    """Return term counts by category."""
    individual_count = sum(1 + len(v) for v in RISK_DICTIONARY_INDIVIDUAL.values())

    return {
        'individual_risks': individual_count,
        'individual_risks_canonical': len(RISK_DICTIONARY_INDIVIDUAL),
        'riskfamilj': len(RISKFAMILJ),
        'probability': len(_flatten_qualification_dict(PROBABILITY_TERMS)),
        'consequence': len(_flatten_qualification_dict(CONSEQUENCE_TERMS)),
        'legitimacy': len(LEGITIMACY_TERMS),
        'total_unique': len(get_all_sample_terms()),
    }


if __name__ == '__main__':
    counts = count_terms()
    print("BERT Sample Dictionary Statistics:")
    print("=" * 50)
    print(f"  Individual risks (canonical): {counts['individual_risks_canonical']}")
    print(f"  Individual risks (with variants): {counts['individual_risks']}")
    print(f"  Riskfamilj (Boholm 2018): {counts['riskfamilj']}")
    print(f"  Probability qualifications: {counts['probability']}")
    print(f"  Consequence qualifications: {counts['consequence']}")
    print(f"  Legitimacy terms: {counts['legitimacy']}")
    print("-" * 50)
    print(f"  Total unique terms: {counts['total_unique']}")

    print("\n\nRiskfamilj terms (Boholm 2018):")
    print("-" * 50)
    for i, term in enumerate(RISKFAMILJ, 1):
        print(f"  {i:2}. {term}")
