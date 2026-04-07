#!/usr/bin/env python3
"""
Tier 2: Risk Categories

Maps individual risks from Tier 1 into thematic categories based on MSB's
official three-part taxonomy from Nationell risk- och sårbarhetsbedömning:

1. NATURE (naturhändelser) - Weather, geological, biological, climate
2. TECHNICAL (tekniska störningar) - Infrastructure failures, accidents, fires
3. ANTAGONISTIC (antagonistiska händelser) - Cyber, terrorism, military, crime

Plus a fourth category for risks that don't fit cleanly:
4. OTHER - Economic, social, environmental pollution

Methodological basis:
    MSB's NRSB 2025 structures risk assessment around these three categories,
    reflecting the source of the threat (natural processes, technical systems,
    or intentional human action). This enables comparison of how different
    actors frame the same underlying taxonomy.

Sources:
    - MSB Nationell risk- och sårbarhetsbedömning 2025 (msb_nrsb_2025)
    - MSB Riskkatalog (msb_riskkatalog_2025)

Usage:
    from scripts.dictionaries.risk_categories import (
        RISK_CATEGORIES,
        get_category_for_risk,
        get_risks_for_category,
    )

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-04-06
"""

from typing import Dict, List, Optional, Set

# Handle both package imports and standalone execution
try:
    from .risk_terms import RISK_TERMS
except ImportError:
    from risk_terms import RISK_TERMS

# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================

# MSB's three-part taxonomy + other
CATEGORY_NAMES = {
    'nature': 'Naturhändelser',
    'technical': 'Tekniska störningar',
    'antagonistic': 'Antagonistiska händelser',
    'other': 'Övriga',
}

# Map each individual risk (from Tier 1) to its category
RISK_TO_CATEGORY: Dict[str, str] = {
    # =========================================================================
    # NATURE (naturhändelser)
    # Weather, geological, biological, climate-related
    # =========================================================================

    # Weather / Meteorological
    'oversvamning': 'nature',
    'varmebolja': 'nature',
    'torka': 'nature',
    'storm': 'nature',
    'isstorm': 'nature',
    'snoovader': 'nature',
    'extrem_kyla': 'nature',
    'halka': 'nature',
    'hagel': 'nature',
    'blixtnedslag': 'nature',
    'lagvatten': 'nature',

    # Geological / Geophysical
    'skred': 'nature',
    'erosion': 'nature',
    'jordbaevning': 'nature',
    'tsunami': 'nature',
    'vulkanutbrott': 'nature',

    # Fire (natural origin)
    'skogsbrand': 'nature',
    'grasbrand': 'nature',

    # Climate long-term
    'klimatforandring': 'nature',
    'havsnivahojning': 'nature',
    'forsurning': 'nature',
    'grundvattenbrist': 'nature',

    # Space weather
    'solstorm': 'nature',

    # Biological / Health (natural disease spread)
    'epidemi': 'nature',
    'pandemi': 'nature',
    'epizooti': 'nature',
    'coronavirus': 'nature',
    'smittspridning': 'nature',
    'antibiotikaresistens': 'nature',
    'fororenat_vatten': 'nature',
    'vaxtsjukdom': 'nature',
    'invasiva_arter': 'nature',

    # =========================================================================
    # TECHNICAL (tekniska störningar)
    # Infrastructure failures, accidents, system disruptions
    # =========================================================================

    # Fire (technical/urban origin)
    'byggnadsbrand': 'technical',
    'explosion': 'technical',

    # Transport accidents
    'trafikolycka': 'technical',
    'tagolycka': 'technical',
    'bussolycka': 'technical',
    'farligt_godsolycka': 'technical',
    'fartygsolycka': 'technical',
    'flygolycka': 'technical',
    'tunnelolycka': 'technical',

    # Industrial accidents
    'industriolycka': 'technical',
    'olycka_farlig_verksamhet': 'technical',
    'dammbrott': 'technical',
    'byggnadskollaps': 'technical',
    'karnteknisk_olycka': 'technical',

    # CBRN (accidental, not weaponized)
    'kemisk_olycka': 'technical',
    'biologisk_olycka': 'technical',
    'radiologisk_olycka': 'technical',
    'cbrn_handelse': 'technical',

    # Infrastructure disruptions
    'elavbrott': 'technical',
    'fjarrvarmebrott': 'technical',
    'vattenforsorjning': 'technical',
    'avloppsbrott': 'technical',
    'it_teleavbrott': 'technical',
    'transportavbrott': 'technical',
    'drivmedelsbrist': 'technical',
    'livsmedelsforsorjning': 'technical',
    'personalbortfall': 'technical',
    'storda_leveranskedjor': 'technical',
    'finansiella_storningar': 'technical',
    'satellitstorningar': 'technical',
    'elektromagnetiska_hot': 'technical',

    # Supply chain / critical supplies
    'lakemedelsbrist': 'technical',
    'medicinteknisk_brist': 'technical',
    'ravarubrist': 'technical',

    # =========================================================================
    # ANTAGONISTIC (antagonistiska händelser)
    # Intentional human action: cyber, terrorism, military, crime
    # =========================================================================

    # Cyber threats
    'cyberattack': 'antagonistic',
    'dataintrang': 'antagonistic',
    'ddos_attack': 'antagonistic',
    'ransomware': 'antagonistic',
    'it_sabotage': 'antagonistic',

    # Terrorism / Violence
    'terrorism': 'antagonistic',
    'hot_och_vald': 'antagonistic',
    'pagaende_dodligt_vald': 'antagonistic',
    'valdsbejakande_extremism': 'antagonistic',

    # Espionage / Sabotage / Hybrid
    'sabotage': 'antagonistic',
    'spionage': 'antagonistic',
    'desinformation': 'antagonistic',
    'hybridhot': 'antagonistic',
    'hot_mot_demokrati': 'antagonistic',

    # Military / Armed conflict
    'vapnat_angrepp': 'antagonistic',
    'vapnat_angrepp_naromradet': 'antagonistic',
    'fjarrangrepp': 'antagonistic',
    'strid_svenskt_territorium': 'antagonistic',
    'blockad': 'antagonistic',
    'militar_konflikt_naromradet': 'antagonistic',

    # Organized crime
    'organiserad_brottslighet': 'antagonistic',
    'vandalism': 'antagonistic',
    'manniskohandel': 'antagonistic',
    'narkotikabrottslighet': 'antagonistic',
    'gangkriminalitet': 'antagonistic',

    # =========================================================================
    # OTHER (övriga)
    # Risks that don't fit cleanly into the MSB taxonomy
    # =========================================================================

    # Social risks (complex causation)
    'social_oro': 'other',
    'flyktingstrom': 'other',
    'forsvunnen_person': 'other',

    # Environmental pollution (can be technical or natural)
    'oljeutslapp': 'other',
    'kemikalieutslapp': 'other',
    'miljoforening': 'other',

    # Economic (systemic, multiple causes)
    'ekonomisk_kris': 'other',
    'arbetsloshet': 'other',
    'inflation': 'other',
}


# =============================================================================
# LEGACY CATEGORY MAPPING (for backward compatibility)
# =============================================================================

# Maps individual risks to the finer-grained categories used by older scripts
# (risk_term_filter.py, risk_context_analysis.py, isomorphism_analysis.py)
LEGACY_RISK_TO_CATEGORY: Dict[str, str] = {
    # naturhot (weather, geological, climate)
    'oversvamning': 'naturhot',
    'varmebolja': 'naturhot',
    'torka': 'naturhot',
    'storm': 'naturhot',
    'isstorm': 'naturhot',
    'snoovader': 'naturhot',
    'extrem_kyla': 'naturhot',
    'halka': 'naturhot',
    'hagel': 'naturhot',
    'blixtnedslag': 'naturhot',
    'lagvatten': 'naturhot',
    'skred': 'naturhot',
    'erosion': 'naturhot',
    'jordbaevning': 'naturhot',
    'tsunami': 'naturhot',
    'vulkanutbrott': 'naturhot',
    'klimatforandring': 'naturhot',
    'havsnivahojning': 'naturhot',
    'forsurning': 'naturhot',
    'grundvattenbrist': 'naturhot',
    'solstorm': 'naturhot',

    # biologiska_hot (health, disease)
    'epidemi': 'biologiska_hot',
    'pandemi': 'biologiska_hot',
    'epizooti': 'biologiska_hot',
    'coronavirus': 'biologiska_hot',
    'smittspridning': 'biologiska_hot',
    'antibiotikaresistens': 'biologiska_hot',
    'fororenat_vatten': 'biologiska_hot',
    'vaxtsjukdom': 'biologiska_hot',
    'invasiva_arter': 'biologiska_hot',

    # olyckor (accidents)
    'trafikolycka': 'olyckor',
    'tagolycka': 'olyckor',
    'bussolycka': 'olyckor',
    'farligt_godsolycka': 'olyckor',
    'fartygsolycka': 'olyckor',
    'flygolycka': 'olyckor',
    'tunnelolycka': 'olyckor',
    'industriolycka': 'olyckor',
    'olycka_farlig_verksamhet': 'olyckor',
    'dammbrott': 'olyckor',
    'byggnadskollaps': 'olyckor',
    'karnteknisk_olycka': 'olyckor',
    'kemisk_olycka': 'olyckor',
    'biologisk_olycka': 'olyckor',
    'radiologisk_olycka': 'olyckor',
    'cbrn_handelse': 'olyckor',
    'forsvunnen_person': 'olyckor',

    # brand (fire, explosions)
    'skogsbrand': 'brand',
    'grasbrand': 'brand',
    'byggnadsbrand': 'brand',
    'explosion': 'brand',

    # antagonistiska_hot (terrorism, violence, military)
    'terrorism': 'antagonistiska_hot',
    'hot_och_vald': 'antagonistiska_hot',
    'pagaende_dodligt_vald': 'antagonistiska_hot',
    'valdsbejakande_extremism': 'antagonistiska_hot',
    'sabotage': 'antagonistiska_hot',
    'spionage': 'antagonistiska_hot',
    'desinformation': 'antagonistiska_hot',
    'hybridhot': 'antagonistiska_hot',
    'hot_mot_demokrati': 'antagonistiska_hot',
    'vapnat_angrepp': 'antagonistiska_hot',
    'vapnat_angrepp_naromradet': 'antagonistiska_hot',
    'fjarrangrepp': 'antagonistiska_hot',
    'strid_svenskt_territorium': 'antagonistiska_hot',
    'blockad': 'antagonistiska_hot',
    'militar_konflikt_naromradet': 'antagonistiska_hot',
    'organiserad_brottslighet': 'antagonistiska_hot',
    'vandalism': 'antagonistiska_hot',
    'manniskohandel': 'antagonistiska_hot',
    'narkotikabrottslighet': 'antagonistiska_hot',
    'gangkriminalitet': 'antagonistiska_hot',

    # cyber_hot (cyber threats)
    'cyberattack': 'cyber_hot',
    'dataintrang': 'cyber_hot',
    'ddos_attack': 'cyber_hot',
    'ransomware': 'cyber_hot',
    'it_sabotage': 'cyber_hot',

    # teknisk_infrastruktur (infrastructure disruptions)
    'elavbrott': 'teknisk_infrastruktur',
    'fjarrvarmebrott': 'teknisk_infrastruktur',
    'vattenforsorjning': 'teknisk_infrastruktur',
    'avloppsbrott': 'teknisk_infrastruktur',
    'it_teleavbrott': 'teknisk_infrastruktur',
    'transportavbrott': 'teknisk_infrastruktur',
    'drivmedelsbrist': 'teknisk_infrastruktur',
    'livsmedelsforsorjning': 'teknisk_infrastruktur',
    'personalbortfall': 'teknisk_infrastruktur',
    'storda_leveranskedjor': 'teknisk_infrastruktur',
    'finansiella_storningar': 'teknisk_infrastruktur',
    'satellitstorningar': 'teknisk_infrastruktur',
    'elektromagnetiska_hot': 'teknisk_infrastruktur',
    'lakemedelsbrist': 'teknisk_infrastruktur',
    'medicinteknisk_brist': 'teknisk_infrastruktur',
    'ravarubrist': 'teknisk_infrastruktur',

    # sociala_risker (social risks)
    'social_oro': 'sociala_risker',
    'flyktingstrom': 'sociala_risker',

    # miljö_klimat (environmental pollution)
    'oljeutslapp': 'miljo_klimat',
    'kemikalieutslapp': 'miljo_klimat',
    'miljoforening': 'miljo_klimat',

    # ekonomi (economic)
    'ekonomisk_kris': 'ekonomi',
    'arbetsloshet': 'ekonomi',
    'inflation': 'ekonomi',
}

# Legacy category display names
LEGACY_CATEGORY_NAMES = {
    'naturhot': 'Natural hazards',
    'biologiska_hot': 'Biological threats',
    'olyckor': 'Accidents',
    'brand': 'Fire',
    'antagonistiska_hot': 'Antagonistic threats',
    'cyber_hot': 'Cyber threats',
    'teknisk_infrastruktur': 'Technical infrastructure',
    'sociala_risker': 'Social risks',
    'miljo_klimat': 'Environment/Climate',
    'ekonomi': 'Economic',
}


def get_legacy_risk_dictionary(include_extended: bool = True) -> Dict[str, List[str]]:
    """
    Generate the old RISK_DICTIONARY format for backward compatibility.

    Returns dictionary mapping legacy category names to flat lists of all
    variant terms in that category:
        {
            'naturhot': ['översvämning', 'översvämningar', 'storm', ...],
            'cyber_hot': ['cyberattack', 'ransomware', ...],
            ...
        }

    Parameters:
        include_extended: If True, includes 'riskfamilj' and 'legitimitetsrisker'
                         categories from Tier 3 (risk_extended.py)
    """
    legacy_dict: Dict[str, List[str]] = {}

    for risk, legacy_cat in LEGACY_RISK_TO_CATEGORY.items():
        if legacy_cat not in legacy_dict:
            legacy_dict[legacy_cat] = []

        # Add all variants for this risk
        if risk in RISK_TERMS:
            legacy_dict[legacy_cat].extend(RISK_TERMS[risk])

    # Optionally add extended categories from Tier 3
    if include_extended:
        try:
            from .risk_extended import RISKFAMILJ, LEGITIMACY_TERMS
        except ImportError:
            from risk_extended import RISKFAMILJ, LEGITIMACY_TERMS

        legacy_dict['riskfamilj'] = RISKFAMILJ.copy()
        legacy_dict['legitimitetsrisker'] = LEGITIMACY_TERMS.copy()

    return legacy_dict


# =============================================================================
# AGGREGATED CATEGORY DICTIONARIES
# =============================================================================

def _build_category_dict() -> Dict[str, Dict[str, List[str]]]:
    """
    Build dictionary of categories, each containing its risks and their variants.

    Returns:
        {
            'nature': {
                'oversvamning': ['översvämning', 'översvämningar', ...],
                ...
            },
            ...
        }
    """
    categories = {cat: {} for cat in CATEGORY_NAMES}

    for risk, category in RISK_TO_CATEGORY.items():
        if risk in RISK_TERMS:
            categories[category][risk] = RISK_TERMS[risk]

    return categories


RISK_CATEGORIES: Dict[str, Dict[str, List[str]]] = _build_category_dict()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_category_for_risk(risk: str) -> Optional[str]:
    """
    Get category name for a given risk.

    Parameters:
        risk: Canonical risk name (e.g., 'oversvamning')

    Returns:
        Category name ('nature', 'technical', 'antagonistic', 'other') or None
    """
    return RISK_TO_CATEGORY.get(risk)


def get_risks_for_category(category: str) -> List[str]:
    """
    Get list of risk names belonging to a category.

    Parameters:
        category: Category name ('nature', 'technical', 'antagonistic', 'other')

    Returns:
        List of canonical risk names
    """
    return [risk for risk, cat in RISK_TO_CATEGORY.items() if cat == category]


def get_all_terms_for_category(category: str) -> Set[str]:
    """
    Get all variant terms for a category (for pattern matching).

    Parameters:
        category: Category name

    Returns:
        Set of all terms (lowercase) in the category
    """
    terms = set()
    for risk, variants in RISK_CATEGORIES.get(category, {}).items():
        for v in variants:
            terms.add(v.lower())
    return terms


def get_category_counts() -> Dict[str, int]:
    """Return count of risks per category."""
    return {cat: len(risks) for cat, risks in RISK_CATEGORIES.items()}


def validate_mapping() -> List[str]:
    """
    Check that all risks in RISK_TERMS are mapped to a category.

    Returns:
        List of unmapped risk names (empty if all are mapped)
    """
    unmapped = []
    for risk in RISK_TERMS:
        if risk not in RISK_TO_CATEGORY:
            unmapped.append(risk)
    return unmapped


if __name__ == '__main__':
    print("Tier 2: Risk Categories (MSB Taxonomy)")
    print("=" * 60)

    counts = get_category_counts()
    for cat, name in CATEGORY_NAMES.items():
        print(f"\n{name} ({cat}): {counts.get(cat, 0)} risks")
        risks = get_risks_for_category(cat)
        for r in sorted(risks)[:5]:
            print(f"  - {r}")
        if len(risks) > 5:
            print(f"  ... and {len(risks) - 5} more")

    unmapped = validate_mapping()
    if unmapped:
        print(f"\nWARNING: {len(unmapped)} unmapped risks: {unmapped}")
    else:
        print(f"\nAll {len(RISK_TERMS)} risks are mapped to categories.")
