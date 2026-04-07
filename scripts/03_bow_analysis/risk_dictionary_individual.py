#!/usr/bin/env python3
"""
Individual Risk Dictionary (Backward Compatibility Wrapper)

This module re-exports from the centralized dictionary at scripts/dictionaries/.
For new code, import directly from:
    from scripts.dictionaries import RISK_TERMS, get_canonical_mapping, ...

Maintained for backward compatibility with existing imports.
"""

import sys
from pathlib import Path

# Import from centralized dictionary
sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries import RISK_TERMS, get_canonical_mapping, get_all_variants
from dictionaries.risk_terms import count_terms

# Re-export with old name for backward compatibility
RISK_DICTIONARY_INDIVIDUAL = RISK_TERMS

__all__ = [
    'RISK_DICTIONARY_INDIVIDUAL',
    'get_all_variants',
    'get_canonical_mapping',
    'count_risks',
]


def count_risks():
    """Return (number of canonical risks, number of total variants)."""
    return count_terms()


if __name__ == '__main__':
    n_risks, n_variants = count_risks()
    print(f"Individual Risk Dictionary Statistics:")
    print(f"  Canonical risks: {n_risks}")
    print(f"  Total variants:  {n_variants}")
    print(f"  Avg variants per risk: {n_variants/n_risks:.1f}")

    print("\n\nAll risks by category:")
    print("-" * 50)
    for risk in sorted(RISK_DICTIONARY_INDIVIDUAL.keys()):
        variants = RISK_DICTIONARY_INDIVIDUAL[risk]
        print(f"  {risk}: {len(variants)} variants")
