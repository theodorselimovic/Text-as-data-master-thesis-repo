#!/usr/bin/env python3
"""
Tier 1: Individual Risk Terms

Base dictionary mapping canonical risk names to their variants (inflections,
synonyms, abbreviations). This is the foundation for all risk detection.

Structure:
    RISK_TERMS = {
        'canonical_name': ['variant1', 'variant2', ...],
    }

Keys use ASCII for programmatic access; values use Swedish characters for matching.
Variants include the canonical form itself and are used for:
- Pattern matching in text
- Collapsing counts to a single concept

Sources:
    - MSB Riskkatalog (msb_riskkatalog_2025)
    - MSB Nationell risk- och sårbarhetsbedömning 2025 (msb_nrsb_2025)
    - EU Civil Protection Knowledge Network (eu_natural_disaster_risks_2025)
    - EU Global Threats Programme (eu_global_threats_2025)
    - EU CBRN Risk Mitigation (eu_cbrn_2025)

Usage:
    from scripts.dictionaries.risk_terms import RISK_TERMS, get_all_variants

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-04-06
"""

from typing import Dict, List, Set

# =============================================================================
# INDIVIDUAL RISK TERMS
# =============================================================================

RISK_TERMS: Dict[str, List[str]] = {
    # =========================================================================
    # WEATHER / METEOROLOGICAL
    # =========================================================================
    'oversvamning': [
        'översvämning', 'översvämningar',
        'skyfall', 'höga flöden', 'högvatten',
    ],
    'varmebolja': [
        'värmebölja', 'värmeböljor',
        'extrem värme', 'extremvärme',
    ],
    'torka': ['torka', 'torkor'],
    'storm': ['storm', 'stormar', 'stormfällning'],
    'isstorm': ['isstorm', 'isbildning', 'isbildning och isstorm'],
    'snoovader': ['stora snömängder', 'snöoväder'],
    'extrem_kyla': ['extrem kyla', 'köldknäpp'],
    'halka': ['halka'],
    'hagel': ['hagel'],
    'blixtnedslag': ['blixt', 'blixtnedslag'],
    'lagvatten': ['låga flöden', 'lågvatten'],

    # =========================================================================
    # GEOLOGICAL / GEOPHYSICAL
    # =========================================================================
    'skred': [
        'ras', 'skred', 'jordskred', 'slamskred',
        'ras och skred', 'ras eller skred',
    ],
    'erosion': ['erosion'],
    'jordbaevning': ['jordbävning', 'jordbävningar', 'jordskalv'],
    'tsunami': ['tsunami', 'tsunamier', 'flodvåg', 'flodvågor'],
    'vulkanutbrott': ['vulkanutbrott', 'vulkanisk aktivitet', 'vulkanaska'],

    # =========================================================================
    # FIRE (NATURAL)
    # =========================================================================
    'skogsbrand': ['skogsbrand', 'skogsbränder'],
    'grasbrand': ['gräsbrand', 'gräsbränder'],

    # =========================================================================
    # CLIMATE LONG-TERM
    # =========================================================================
    'klimatforandring': ['klimatförändring', 'klimatförändringarna', 'klimatförändringar'],
    'havsnivahojning': ['havsnivåhöjning', 'stigande havsnivåer', 'havsnivå'],
    'forsurning': ['försurning', 'havsförsurning'],
    'grundvattenbrist': [
        'grundvattenbrist', 'sjunkande grundvattennivåer',
        'grundvattennivå', 'vattenbrist',
    ],

    # =========================================================================
    # SPACE WEATHER
    # =========================================================================
    'solstorm': ['solstorm', 'solstormar'],

    # =========================================================================
    # BIOLOGICAL / HEALTH
    # =========================================================================
    'epidemi': ['epidemi', 'epidemier'],
    'pandemi': ['pandemi', 'pandemier', 'influensapandemi'],
    'epizooti': [
        'epizooti', 'epizootier',
        'djursjukdom', 'djursjukdomar', 'zoonos', 'zoonoser',
    ],
    'coronavirus': ['covid', 'coronaviruset', 'corona', 'coronavirus', 'sars-cov', 'mers-cov'],
    'smittspridning': [
        'smittsam sjukdom', 'smittsamma sjukdomar',
        'smitta', 'smittspridning', 'sjukdomsutbrott',
    ],
    'antibiotikaresistens': ['antibiotikaresistens', 'resistenta bakterier'],
    'fororenat_vatten': [
        'förorenat vatten', 'smittat vatten',
        'förorenat eller smittat vatten',
        'kontaminerat vatten', 'kontaminerat dricksvatten',
    ],
    'vaxtsjukdom': [
        'växtsjukdom', 'växtsjukdomar', 'växtskadegörare',
        'skadeinsekter', 'svampsjukdomar',
    ],
    'invasiva_arter': ['invasiva arter', 'främmande arter', 'invasiv art'],

    # =========================================================================
    # FIRE (TECHNICAL/URBAN)
    # =========================================================================
    'byggnadsbrand': ['brand', 'bränder', 'storbrand', 'byggnadsbrand'],
    'explosion': ['explosion', 'explosioner', 'gasexplosion'],

    # =========================================================================
    # TRANSPORT ACCIDENTS
    # =========================================================================
    'trafikolycka': ['trafikolycka', 'trafikolyckor', 'vägolycka', 'vägolyckor'],
    'tagolycka': ['tågolycka', 'tågolyckor', 'järnvägsolycka', 'järnvägsolyckor'],
    'bussolycka': ['bussolycka', 'bussolyckor', 'spårbundna olyckor'],
    'farligt_godsolycka': [
        'olycka med transport av farligt gods', 'olycka med farligt gods',
        'farligt gods', 'transport av farligt gods',
        'farligt godsolycka', 'farligt godsolyckor',
    ],
    'fartygsolycka': [
        'fartygsolycka', 'fartygsolyckor', 'fartygskollision',
        'båtolycka', 'båtolyckor',
    ],
    'flygolycka': ['flygolycka', 'flygolyckor', 'flyghaveri'],
    'tunnelolycka': ['tunnelolycka'],

    # =========================================================================
    # INDUSTRIAL ACCIDENTS
    # =========================================================================
    'industriolycka': ['industriolycka', 'industriolyckor', 'kemikalieolycka'],
    'olycka_farlig_verksamhet': ['olycka vid farlig verksamhet', 'farlig verksamhet'],
    'dammbrott': ['dammbrott'],
    'byggnadskollaps': ['byggnadsras', 'byggnadskollaps', 'brokollaps'],
    'karnteknisk_olycka': [
        'olyckor med nukleära ämnen', 'olyckor med radioaktiva ämnen',
        'kärnteknisk olycka', 'kärnkraftsolycka',
    ],

    # =========================================================================
    # CBRN
    # =========================================================================
    'kemisk_olycka': [
        'kemisk olycka', 'kemiska olyckor',
        'kemiskt utsläpp', 'giftiga ämnen', 'giftutsläpp',
    ],
    'biologisk_olycka': [
        'biologisk olycka', 'biologiska olyckor',
        'biologiskt utsläpp', 'smittämnen',
    ],
    'radiologisk_olycka': [
        'radiologisk olycka', 'radiologiska olyckor',
        'strålningsolycka', 'radioaktivt utsläpp', 'strålning',
    ],
    'cbrn_handelse': [
        'cbrn', 'cbrn-händelse', 'cbrne',
        'kemiska, biologiska, radiologiska och nukleära hot',
    ],

    # =========================================================================
    # INFRASTRUCTURE DISRUPTIONS
    # =========================================================================
    'elavbrott': [
        'elavbrott', 'strömavbrott', 'kraftavbrott',
        'effektbrist', 'elbrist',
        'elförsörjning', 'kraftförsörjning', 'energiförsörjning',
    ],
    'fjarrvarmebrott': ['fjärrvärmebrott', 'fjärrvärmeavbrott'],
    'vattenforsorjning': [
        'vattenförsörjning', 'dricksvattenförsörjning', 'dricksvatten',
        'vattenläcka', 'vattenläckor',
    ],
    'avloppsbrott': ['avloppsbrott', 'avloppssystem', 'avloppshaveri'],
    'it_teleavbrott': [
        'IT-bortfall', 'it-bortfall', 'IT-avbrott', 'it-avbrott',
        'dataförlust', 'systemfel', 'nätverksavbrott',
        'kommunikationsavbrott', 'teleavbrott', 'telebrott',
        'telenätavbrott', 'mobilnätsavbrott',
        'it- och teleavbrott', 'IT- och teleavbrott',
        'it/teleavbrott', 'IT/teleavbrott',
        'elektroniska kommunikationer', 'elektronisk kommunikation',
    ],
    'transportavbrott': [
        'transportavbrott', 'transportstörning', 'transportstörningar',
        'logistikavbrott', 'distributionsstörning',
    ],
    'drivmedelsbrist': [
        'drivmedelsbrist', 'drivsmedelsbrist', 'bränslebrist',
        'drivmedel', 'drivmedelsförsörjning',
    ],
    'livsmedelsforsorjning': [
        'livsmedelsförsörjning', 'livsmedelsbrist', 'matförsörjning',
        'livsmedelsförsörjningen',
    ],
    'personalbortfall': ['personalbortfall', 'personalbrist'],
    'storda_leveranskedjor': [
        'störda leveranskedjor', 'störd leveranskedja',
        'leveransstörningar', 'försörjningsstörningar',
    ],
    'finansiella_storningar': [
        'störningar i finansiella system', 'störningar i det finansiella systemet',
        'finansiell störning', 'betalningssystem',
    ],
    'satellitstorningar': [
        'störningar i satellitbaserade navigationssystem',
        'störningar i satellitsystem', 'gps-störningar',
        'gnss-störningar', 'satellitstörningar',
    ],
    'elektromagnetiska_hot': [
        'elektromagnetiska hot', 'elektromagnetisk störning',
        'emp', 'elektromagnetisk puls',
    ],

    # =========================================================================
    # SUPPLY CHAIN / CRITICAL SUPPLIES
    # =========================================================================
    'lakemedelsbrist': [
        'läkemedelsbrist', 'läkemedelsförsörjning',
        'brist på läkemedel', 'medicinbrist',
    ],
    'medicinteknisk_brist': [
        'medicinteknisk brist', 'brist på medicinteknisk utrustning',
        'sjukvårdsmaterial', 'skyddsutrustning',
    ],
    'ravarubrist': ['råvarubrist', 'materialbrist', 'komponentbrist'],

    # =========================================================================
    # CYBER THREATS
    # =========================================================================
    'cyberattack': ['cyberattack', 'cyberattacker', 'nätattack', 'nätattacker'],
    'dataintrang': ['dataintrång', 'hackerattack', 'hackerattacker'],
    'ddos_attack': ['DDoS-attack', 'ddos-attack', 'ddos'],
    'ransomware': ['ransomware', 'utpressningsvirus'],
    'it_sabotage': ['IT-sabotage', 'it-sabotage'],

    # =========================================================================
    # TERRORISM / VIOLENCE
    # =========================================================================
    'terrorism': [
        'terror', 'terrorism', 'terrorhot', 'terrorattentat', 'terrorhandling',
    ],
    'hot_och_vald': ['hot och våld', 'våldsbrott'],
    'pagaende_dodligt_vald': ['pågående dödligt våld', 'pdv'],
    'valdsbejakande_extremism': [
        'våldsbejakande extremism', 'vansinnesdåd',
        'våldsbejakande extremism eller vansinnesdåd',
    ],

    # =========================================================================
    # ESPIONAGE / SABOTAGE / HYBRID
    # =========================================================================
    'sabotage': ['sabotage'],
    'spionage': [
        'spionage', 'underrättelseverksamhet', 'främmande underrättelsetjänst',
        'utländsk underrättelseverksamhet',
    ],
    'desinformation': [
        'desinformation', 'påverkanskampanj', 'påverkanskampanjer',
        'informationspåverkan', 'påverkansoperation', 'påverkansoperationer',
        'falsk information', 'vilseledande information',
    ],
    'hybridhot': ['hybrid hot', 'hybridhot'],
    'hot_mot_demokrati': [
        'hot mot demokrati', 'hot mot demokratin',
        'hot mot mänskliga fri- och rättigheter',
        'hot mot demokrati och mänskliga fri- och rättigheter',
    ],

    # =========================================================================
    # MILITARY / ARMED CONFLICT
    # =========================================================================
    'vapnat_angrepp': ['väpnat angrepp'],
    'vapnat_angrepp_naromradet': ['väpnat angrepp i närområdet'],
    'fjarrangrepp': [
        'fjärrangrepp', 'fjärrvapen', 'robotangrepp',
        'missilangrepp', 'drönarangrepp',
    ],
    'strid_svenskt_territorium': [
        'strid på svenskt territorium', 'markstrid',
        'invasion', 'ockupation',
    ],
    'blockad': ['blockad', 'sjöblockad', 'handelsblockad'],
    'militar_konflikt_naromradet': [
        'militär konflikt i närområdet', 'krig i närområdet',
        'regional konflikt', 'konflikt i östersjöområdet',
    ],

    # =========================================================================
    # ORGANIZED CRIME
    # =========================================================================
    'organiserad_brottslighet': ['organiserad brottslighet', 'kriminalitet'],
    'vandalism': ['vandalism', 'skadegörelse', 'inbrott'],
    'manniskohandel': ['människohandel', 'trafficking'],
    'narkotikabrottslighet': [
        'narkotikabrottslighet', 'narkotikahandel',
        'droghandel', 'narkotika',
    ],
    'gangkriminalitet': [
        'gängkriminalitet', 'gängvåld', 'gängkonflikter',
        'skjutningar', 'sprängningar',
    ],

    # =========================================================================
    # SOCIAL RISKS
    # =========================================================================
    'social_oro': [
        'social oro', 'sociala oroligheter', 'civila oroligheter', 'upplopp',
        'misstro',
    ],
    'flyktingstrom': [
        'flyktingström', 'flyktingströmmar',
        'interna flyktingströmmar', 'intern flyktingström',
        'flyktingkris', 'migrationskris',
    ],
    'forsvunnen_person': [
        'försvunnen person', 'försvunna personer', 'försvunnen brukare',
        'försvinnande', 'saknad person',
    ],

    # =========================================================================
    # ENVIRONMENTAL POLLUTION
    # =========================================================================
    'oljeutslapp': ['oljeutsläpp', 'oljespill'],
    'kemikalieutslapp': ['kemikalieutsläpp', 'kemikaliespill'],
    'miljoforening': [
        'miljöförorening', 'markförorening',
        'luftföroreningar', 'vattenförorening',
        'föroreningar', 'utsläpp',
    ],

    # =========================================================================
    # ECONOMIC
    # =========================================================================
    'ekonomisk_kris': ['ekonomisk kris', 'finanskris', 'recession', 'lågkonjuktur', 'ekonomisk nedgång'],
    'arbetsloshet': ['arbetslöshet'],
    'inflation': ['inflation'],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_variants() -> List[str]:
    """Return flat list of all variants across all risks."""
    all_variants = []
    for variants in RISK_TERMS.values():
        all_variants.extend(variants)
    return all_variants


def get_canonical_mapping() -> Dict[str, str]:
    """
    Return mapping from each variant to its canonical form.

    Returns:
        dict: variant (lowercase) -> canonical_form
    """
    mapping = {}
    for canonical, variants in RISK_TERMS.items():
        for variant in variants:
            mapping[variant.lower()] = canonical
    return mapping


def get_variant_set() -> Set[str]:
    """Return set of all variants (lowercase) for fast lookup."""
    return {v.lower() for variants in RISK_TERMS.values() for v in variants}


def count_terms() -> tuple:
    """Return (number of canonical risks, number of total variants)."""
    n_risks = len(RISK_TERMS)
    n_variants = sum(len(v) for v in RISK_TERMS.values())
    return n_risks, n_variants


if __name__ == '__main__':
    n_risks, n_variants = count_terms()
    print(f"Tier 1: Individual Risk Terms")
    print(f"  Canonical risks: {n_risks}")
    print(f"  Total variants:  {n_variants}")
    print(f"  Avg variants per risk: {n_variants/n_risks:.1f}")
