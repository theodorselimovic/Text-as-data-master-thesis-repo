#!/usr/bin/env python3
"""
Risk Dictionary (Category-based)

Contains the category-based risk dictionary for Swedish RSA analysis.
Categories group related risk terms for aggregate analysis.

Categories:
- naturhot: Natural hazards including weather, climate, and wildfires
- biologiska_hot: Epidemics, pandemics, diseases
- olyckor: Accidents including transport, industrial, fires, and building collapse
- antagonistiska_hot: Security threats, terrorism, violence
- cyber_hot: Cyber attacks and IT security
- sociala_risker: Social unrest and civil disturbances
- teknisk_infrastruktur: Infrastructure disruptions (power, water, communications)
- miljö_klimat: Environmental pollution and contamination
- ekonomi: Economic crises

Note: Fire risks have been reorganized:
- Wildfires (skogsbrand, gräsbrand) -> naturhot
- Building/urban fires -> olyckor

Usage:
    from risk_dictionary_categories import RISK_DICTIONARY_CATEGORIES

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-03-18
"""

# =============================================================================
# CATEGORY-BASED RISK DICTIONARY
# =============================================================================

RISK_DICTIONARY_CATEGORIES = {
    'naturhot': [
        # General natural hazards
        'naturhändelser', 'naturhot', 'väderrelaterade händelser',
        # Climate change
        'klimatförändring', 'klimatförändringarna', 'klimatförändringar',
        # Flooding
        'översvämning', 'översvämningar', 'skyfall', 'höga flöden', 'högvatten',
        # Heat/drought
        'värme', 'värmebölja', 'värmeböljor', 'torka', 'torkor',
        'extrem värme', 'extremvärme',
        # Landslides
        'ras', 'skred', 'jordskred', 'slamskred', 'erosion',
        # Storms
        'storm', 'stormar', 'stormfällning', 'isstorm',
        # Weather
        'blixt', 'blixtnedslag', 'hagel', 'halka', 'köldknäpp',
        'stora snömängder', 'snöoväder',
        'extrem kyla', 'extremt väder',
        'låga flöden', 'lågvatten',
        # Wildfires (moved from brand category)
        'skogsbrand', 'skogsbränder', 'gräsbrand', 'gräsbränder',
    ],
    'biologiska_hot': [
        'epidemi', 'epidemier', 'pandemi', 'pandemier',
        'epizooti', 'epizootier', 'covid', 'coronaviruset',
        'smittsam sjukdom', 'smittsamma sjukdomar',
        'smitta', 'smittspridning', 'sjukdomsutbrott',
        'influensa', 'influensapandemi',
        'djursjukdom', 'djursjukdomar', 'zoonos', 'zoonoser',
        'antibiotikaresistens', 'resistenta bakterier',
        'hälsa', 'folkhälsa',
    ],
    'olyckor': [
        # Industrial accidents
        'olycka vid farlig verksamhet', 'farlig verksamhet',
        'industriolycka', 'kemikalieolycka',
        # Transport accidents
        'olycka med transport av farligt gods', 'olycka med farligt gods',
        'farligt gods', 'transport av farligt gods',
        'vägolycka', 'vägolyckor', 'trafikolycka', 'trafikolyckor',
        'tågolycka', 'tågolyckor', 'järnvägsolycka', 'järnvägsolyckor',
        'bussolycka', 'bussolyckor', 'spårbundna olyckor',
        'fartygsolycka', 'fartygsolyckor', 'fartygskollision', 'båtolycka', 'båtolyckor',
        'flygolycka', 'flygolyckor', 'flyghaveri',
        # Infrastructure collapse
        'dammbrott',
        'brokollaps', 'tunnelolycka',
        'byggnadsras', 'byggnadskollaps',
        # Nuclear
        'olyckor med nukleära ämnen', 'olyckor med radioaktiva ämnen', 'kärnteknisk olycka',
        # Missing persons
        'försvunnen person', 'försvunna personer', 'försvunnen brukare',
        'försvinnande', 'saknad person',
        # Fires and explosions (moved from brand category)
        'brand', 'bränder', 'storbrand', 'byggnadsbrand', 'fordonsbrand',
        'explosion', 'explosioner', 'gasexplosion', 'brandfarligt gods',
    ],
    'antagonistiska_hot': [
        'statliga antagonister', 'statlig antagonist',
        'icke-statliga antagonister', 'icke-statlig antagonist',
        'terror', 'terrorism', 'terrorhot', 'terrorattentat', 'terrorhandling',
        'hot och våld', 'våld', 'våldsbrott',
        'pågående dödligt våld', 'våldsbejakande extremism',
        'sabotage', 'spionage',
        'brott', 'kriminalitet', 'organiserad brottslighet',
        'vandalism', 'skadegörelse', 'inbrott',
        'desinformation', 'påverkanskampanj', 'påverkanskampanjer',
        'hybrid hot', 'hybridhot',
        'säkerhetshot',
        'väpnat angrepp',  # removed 'höjd beredskap' - not a risk
    ],
    'cyber_hot': [
        'dataintrång', 'cyberattack', 'cyberattacker', 'cybersäkerhet',
        'nätattack', 'nätattacker', 'hackerattack', 'hackerattacker',
        'DDoS-attack', 'ddos-attack', 'ransomware', 'datavirus', 'virus',
        'IT-sabotage',
    ],
    'sociala_risker': [
        'samhällsvärden', 'värdesystem',
        'social oro', 'sociala oroligheter', 'civila oroligheter', 'upplopp',
    ],
    'teknisk_infrastruktur': [
        # Power
        'strömavbrott', 'elavbrott', 'kraftförsörjning', 'elförsörjning', 'effektbrist',
        'energiförsörjning', 'energibrist',
        # Heating
        'fjärrvärmebrott', 'fjärrvärme', 'värmeförsörjning',
        # Water
        'vattenläcka', 'vattenläckor', 'vattenförsörjning', 'dricksvatten',
        'dricksvattenförsörjning',
        'avloppsbrott', 'avloppssystem',
        # IT/Communications
        'IT-bortfall', 'it-bortfall', 'IT-avbrott', 'it-avbrott',
        'dataförlust', 'systemfel', 'nätverksavbrott',
        'kommunikationsavbrott', 'teleavbrott', 'telebrott',
        'elektroniska kommunikationer', 'elektronisk kommunikation',
        # Transport/Supply
        'distributionsstörning', 'logistikavbrott', 'transportavbrott',
        'transporter', 'transportstörning', 'transportstörningar',
        'drivsmedelsbrist', 'bränslebrist', 'försörjningsbrist',
        'drivmedel', 'drivmedelsförsörjning',
        'livsmedelsförsörjning', 'livsmedelsbrist', 'matförsörjning',
    ],
    'miljö_klimat': [
        'miljöförorening', 'kemikalieutsläpp', 'oljeutsläpp',
        'markförorening', 'luftföroreningar', 'vattenförorening',
        'miljöhot', 'miljöskada', 'utsläpp', 'föroreningar', 'klimatförändring',
        'klimatpåverkan', 'klimatrelaterade', 'klimatförändringen', 'försurning',
    ],
    'ekonomi': [
        'ekonomisk kris', 'finanskris', 'recession', 'lågkonjuktur'
        'arbetslöshet', 'inflation', 'ekonomisk nedgång',
    ],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_terms() -> list:
    """Return flat list of all terms across all categories."""
    all_terms = []
    for terms in RISK_DICTIONARY_CATEGORIES.values():
        all_terms.extend(terms)
    return all_terms


def get_term_to_category() -> dict:
    """
    Return mapping from each term to its category.

    Returns:
        dict: term -> category
    """
    mapping = {}
    for category, terms in RISK_DICTIONARY_CATEGORIES.items():
        for term in terms:
            if term not in mapping:  # Keep first if duplicates
                mapping[term.lower()] = category
    return mapping


def count_terms() -> tuple:
    """Return (number of categories, number of total terms)."""
    n_categories = len(RISK_DICTIONARY_CATEGORIES)
    n_terms = sum(len(t) for t in RISK_DICTIONARY_CATEGORIES.values())
    return n_categories, n_terms


# For backwards compatibility
RISK_DICTIONARY_ORIGINAL = RISK_DICTIONARY_CATEGORIES


if __name__ == '__main__':
    n_cats, n_terms = count_terms()
    print(f"Category-based Risk Dictionary Statistics:")
    print(f"  Categories: {n_cats}")
    print(f"  Total terms: {n_terms}")
    print(f"  Avg terms per category: {n_terms/n_cats:.1f}")

    print("\n\nTerms by category:")
    print("-" * 50)
    for cat, terms in RISK_DICTIONARY_CATEGORIES.items():
        print(f"  {cat}: {len(terms)} terms")
