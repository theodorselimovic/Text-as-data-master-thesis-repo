#!/usr/bin/env python3
"""
Individual Risk Dictionary

A dictionary of individual risks (not categories) for prevalence analysis.
Designed to:
1. Collapse singular/plural variants into canonical forms
2. Focus on actual risks (not general concepts like 'säkerhet')
3. Include all specific risk terms from Swedish RSA documents

Usage:
    from risk_dictionary_individual import RISK_DICTIONARY_INDIVIDUAL

    # Each key is the canonical form, values are all variants to match
    for canonical, variants in RISK_DICTIONARY_INDIVIDUAL.items():
        # variants includes the canonical form itself

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-03-18
"""

# =============================================================================
# INDIVIDUAL RISK DICTIONARY
# =============================================================================

# Dictionary mapping canonical risk name -> list of all variants (including itself)
# Variants are collapsed when counting to get a single count per risk concept

RISK_DICTIONARY_INDIVIDUAL = {
    # -------------------------------------------------------------------------
    # NATURAL HAZARDS
    # -------------------------------------------------------------------------
    'översvämning': [
        'översvämning', 'översvämningar',
        'skyfall', 'höga flöden', 'högvatten',
    ],
    'värmebölja': [
        'värmebölja', 'värmeböljor',
        'extrem värme', 'extremvärme',
    ],
    'torka': ['torka', 'torkor'],
    'skred': [
        'ras', 'skred', 'jordskred', 'slamskred',
        'ras och skred', 'ras eller skred',
    ],
    'erosion': ['erosion'],
    'storm': ['storm', 'stormar', 'stormfällning'],
    'isstorm': ['isstorm', 'isbildning', 'isbildning och isstorm'],
    'skogsbrand': ['skogsbrand', 'skogsbränder'],
    'gräsbrand': ['gräsbrand', 'gräsbränder'],
    'brand': ['brand', 'bränder', 'storbrand', 'byggnadsbrand'],
    'blixtnedslag': ['blixt', 'blixtnedslag'],
    'snöoväder': ['stora snömängder', 'snöoväder'],
    'extrem kyla': ['extrem kyla', 'köldknäpp'],
    'halka': ['halka'],
    'hagel': ['hagel'],
    'lågvatten': ['låga flöden', 'lågvatten'],
    'klimatförändring': ['klimatförändring', 'klimatförändringarna', 'klimatförändringar'],
    'solstorm': ['solstorm', 'solstormar'],

    # -------------------------------------------------------------------------
    # BIOLOGICAL THREATS / HEALTH
    # -------------------------------------------------------------------------
    'epidemi': ['epidemi', 'epidemier'],
    'pandemi': ['pandemi', 'pandemier', 'influensapandemi'],
    'epizooti': [
        'epizooti', 'epizootier',
        'djursjukdom', 'djursjukdomar', 'zoonos', 'zoonoser',
    ],
    'covid': ['covid', 'coronaviruset', 'corona'],
    'smittspridning': [
        'smittsam sjukdom', 'smittsamma sjukdomar',
        'smitta', 'smittspridning', 'sjukdomsutbrott',
    ],
    'antibiotikaresistens': ['antibiotikaresistens', 'resistenta bakterier'],
    'förorenat vatten': [
        'förorenat vatten', 'smittat vatten',
        'förorenat eller smittat vatten',
        'kontaminerat vatten', 'kontaminerat dricksvatten',
    ],

    # -------------------------------------------------------------------------
    # ACCIDENTS
    # -------------------------------------------------------------------------
    'trafikolycka': ['trafikolycka', 'trafikolyckor', 'vägolycka', 'vägolyckor'],
    'tågolycka': ['tågolycka', 'tågolyckor', 'järnvägsolycka', 'järnvägsolyckor'],
    'bussolycka': ['bussolycka', 'bussolyckor', 'spårbundna olyckor'],
    'farligt godsolycka': [
        'olycka med transport av farligt gods', 'olycka med farligt gods',
        'farligt gods', 'transport av farligt gods',
        'farligt godsolycka', 'farligt godsolyckor',
    ],
    'industriolycka': ['industriolycka', 'industriolyckor', 'kemikalieolycka'],
    'olycka vid farlig verksamhet': ['olycka vid farlig verksamhet', 'farlig verksamhet'],
    'dammbrott': ['dammbrott'],
    'fartygsolycka': [
        'fartygsolycka', 'fartygsolyckor', 'fartygskollision',
        'båtolycka', 'båtolyckor',
    ],
    'flygolycka': ['flygolycka', 'flygolyckor', 'flyghaveri'],
    'kärnteknisk olycka': [
        'olyckor med nukleära ämnen', 'olyckor med radioaktiva ämnen',
        'kärnteknisk olycka', 'kärnkraftsolycka',
    ],
    'byggnadskollaps': ['byggnadsras', 'byggnadskollaps', 'brokollaps'],
    'tunnelolycka': ['tunnelolycka'],
    'explosion': ['explosion', 'explosioner', 'gasexplosion'],
    'försvunnen person': [
        'försvunnen person', 'försvunna personer', 'försvunnen brukare',
        'försvinnande', 'saknad person',
    ],

    # -------------------------------------------------------------------------
    # ANTAGONISTIC THREATS
    # -------------------------------------------------------------------------
    'terrorism': [
        'terror', 'terrorism', 'terrorhot', 'terrorattentat', 'terrorhandling',
    ],
    'hot och våld': ['hot och våld', 'våldsbrott'],
    'pågående dödligt våld': ['pågående dödligt våld', 'pdv'],
    'våldsbejakande extremism': [
        'våldsbejakande extremism', 'vansinnesdåd',
        'våldsbejakande extremism eller vansinnesdåd',
    ],
    'sabotage': ['sabotage'],
    'spionage': ['spionage'],
    'organiserad brottslighet': ['organiserad brottslighet', 'kriminalitet'],
    'vandalism': ['vandalism', 'skadegörelse', 'inbrott'],
    'desinformation': ['desinformation', 'påverkanskampanj', 'påverkanskampanjer'],
    'hybridhot': ['hybrid hot', 'hybridhot'],
    'väpnat angrepp': ['väpnat angrepp'],
    'väpnat angrepp i närområdet': ['väpnat angrepp i närområdet'],
    'hot mot demokrati': [
        'hot mot demokrati', 'hot mot demokratin',
        'hot mot mänskliga fri- och rättigheter',
        'hot mot demokrati och mänskliga fri- och rättigheter',
    ],

    # -------------------------------------------------------------------------
    # CYBER THREATS
    # -------------------------------------------------------------------------
    'cyberattack': ['cyberattack', 'cyberattacker', 'nätattack', 'nätattacker'],
    'dataintrång': ['dataintrång', 'hackerattack', 'hackerattacker'],
    'ddos-attack': ['DDoS-attack', 'ddos-attack', 'ddos'],
    'ransomware': ['ransomware', 'utpressningsvirus'],
    'IT-sabotage': ['IT-sabotage', 'it-sabotage'],

    # -------------------------------------------------------------------------
    # SOCIAL RISKS
    # -------------------------------------------------------------------------
    'social oro': [
        'social oro', 'sociala oroligheter', 'civila oroligheter', 'upplopp',
        'misstro',  # user requested to add to social oro
    ],
    'flyktingström': [
        'flyktingström', 'flyktingströmmar',
        'interna flyktingströmmar', 'intern flyktingström',
        'flyktingkris', 'migrationskris',
    ],

    # -------------------------------------------------------------------------
    # INFRASTRUCTURE DISRUPTIONS
    # -------------------------------------------------------------------------
    'elavbrott': [
        'elavbrott', 'strömavbrott', 'kraftavbrott',
        'effektbrist', 'elbrist',
    ],
    'elförsörjning': ['elförsörjning', 'kraftförsörjning', 'energiförsörjning'],
    'fjärrvärmebrott': ['fjärrvärmebrott', 'fjärrvärmeavbrott'],
    'vattenförsörjning': [
        'vattenförsörjning', 'dricksvattenförsörjning', 'dricksvatten',
        'vattenläcka', 'vattenläckor',
    ],
    'avloppsbrott': ['avloppsbrott', 'avloppssystem', 'avloppshaveri'],
    'IT-avbrott': [
        'IT-bortfall', 'it-bortfall', 'IT-avbrott', 'it-avbrott',
        'dataförlust', 'systemfel', 'nätverksavbrott',
    ],
    'teleavbrott': [
        'kommunikationsavbrott', 'teleavbrott', 'telebrott',
        'telenätavbrott', 'mobilnätsavbrott',
    ],
    'it- och teleavbrott': [
        'it- och teleavbrott', 'IT- och teleavbrott',
        'it/teleavbrott', 'IT/teleavbrott',
    ],
    'elektroniska kommunikationer': [
        'elektroniska kommunikationer', 'elektronisk kommunikation',
    ],
    'transportavbrott': [
        'transportavbrott', 'transportstörning', 'transportstörningar',
        'logistikavbrott', 'distributionsstörning',
    ],
    'transporter': ['transporter'],
    'drivmedelsbrist': [
        'drivmedelsbrist', 'drivsmedelsbrist', 'bränslebrist',
    ],
    'drivmedel': ['drivmedel', 'drivmedelsförsörjning'],
    'livsmedelsförsörjning': [
        'livsmedelsförsörjning', 'livsmedelsbrist', 'matförsörjning',
        'livsmedelsförsörjningen',
    ],
    'personalbortfall': ['personalbortfall', 'personalbrist'],
    'störda leveranskedjor': [
        'störda leveranskedjor', 'störd leveranskedja',
        'leveransstörningar', 'försörjningsstörningar',
    ],
    'störningar i finansiella system': [
        'störningar i finansiella system', 'störningar i det finansiella systemet',
        'finansiell störning', 'betalningssystem',
    ],
    'störningar i satellitbaserade system': [
        'störningar i satellitbaserade navigationssystem',
        'störningar i satellitsystem', 'gps-störningar',
        'gnss-störningar', 'satellitstörningar',
    ],
    'elektromagnetiska hot': [
        'elektromagnetiska hot', 'elektromagnetisk störning',
        'emp', 'elektromagnetisk puls',
    ],

    # -------------------------------------------------------------------------
    # ENVIRONMENTAL
    # -------------------------------------------------------------------------
    'oljeutsläpp': ['oljeutsläpp', 'oljespill'],
    'kemikalieutsläpp': ['kemikalieutsläpp', 'kemikaliespill'],
    'miljöförorening': [
        'miljöförorening', 'markförorening',
        'luftföroreningar', 'vattenförorening',
        'föroreningar', 'utsläpp',
    ],

    # -------------------------------------------------------------------------
    # ECONOMIC
    # -------------------------------------------------------------------------
    'ekonomisk kris': ['ekonomisk kris', 'finanskris', 'recession', 'lågkonjuktur', 'ekonomisk nedgång'],
    'arbetslöshet': ['arbetslöshet'],
    'inflation': ['inflation'],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_variants() -> list:
    """Return flat list of all variants across all risks."""
    all_variants = []
    for variants in RISK_DICTIONARY_INDIVIDUAL.values():
        all_variants.extend(variants)
    return all_variants


def get_canonical_mapping() -> dict:
    """
    Return mapping from each variant to its canonical form.

    Returns:
        dict: variant -> canonical_form
    """
    mapping = {}
    for canonical, variants in RISK_DICTIONARY_INDIVIDUAL.items():
        for variant in variants:
            mapping[variant.lower()] = canonical
    return mapping


def count_risks() -> tuple:
    """Return (number of canonical risks, number of total variants)."""
    n_risks = len(RISK_DICTIONARY_INDIVIDUAL)
    n_variants = sum(len(v) for v in RISK_DICTIONARY_INDIVIDUAL.values())
    return n_risks, n_variants


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
