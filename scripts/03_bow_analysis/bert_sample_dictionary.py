#!/usr/bin/env python3
"""
BERT Sampling Dictionary (Backward Compatibility Wrapper)

This module re-exports from the centralized dictionary at scripts/dictionaries/.
For new code, import directly from:
    from scripts.dictionaries import get_all_sampling_terms, RISKFAMILJ, ...

Maintained for backward compatibility with existing imports.
"""

import sys
from pathlib import Path

# Import from centralized dictionary
sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries import (
    RISK_TERMS as RISK_DICTIONARY_INDIVIDUAL,
    RISKFAMILJ,
    PROBABILITY_TERMS,
    CONSEQUENCE_TERMS,
    LEGITIMACY_TERMS,
    get_all_sampling_terms,
)
from dictionaries.risk_extended import count_terms

# Re-export for backward compatibility
__all__ = [
    'RISK_DICTIONARY_INDIVIDUAL',
    'RISKFAMILJ',
    'PROBABILITY_TERMS',
    'CONSEQUENCE_TERMS',
    'LEGITIMACY_TERMS',
    'BERT_SAMPLE_DICTIONARY',
    'QUALIFICATION_SCALES',
    'get_all_sample_terms',
    'get_flat_risk_terms',
    'count_terms',
]


def _flatten_qualification_dict(qual_dict):
    """Flatten a qualification dictionary to a list of terms."""
    terms = []
    for level_terms in qual_dict.values():
        terms.extend(level_terms)
    return terms


# Backward-compatible structures
BERT_SAMPLE_DICTIONARY = {
    'individual_risks': RISK_DICTIONARY_INDIVIDUAL,
    'riskfamilj': RISKFAMILJ,
    'probability': _flatten_qualification_dict(PROBABILITY_TERMS),
    'consequence': _flatten_qualification_dict(CONSEQUENCE_TERMS),
    'legitimacy': LEGITIMACY_TERMS,
}

QUALIFICATION_SCALES = {
    'probability': PROBABILITY_TERMS,
    'consequence': CONSEQUENCE_TERMS,
}


def get_all_sample_terms():
    """Return flat set of all terms for paragraph filtering."""
    return get_all_sampling_terms()


def get_flat_risk_terms():
    """Return only individual risk terms (no qualifications/riskfamilj)."""
    terms = set()
    for canonical, variants in RISK_DICTIONARY_INDIVIDUAL.items():
        terms.add(canonical.lower())
        for v in variants:
            terms.add(v.lower())
    return terms


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
