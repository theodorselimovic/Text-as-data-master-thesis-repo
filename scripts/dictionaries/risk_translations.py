#!/usr/bin/env python3
"""
Risk Term Translations: Swedish → English

Maps canonical Swedish risk terms to English equivalents for:
- Visualization labels
- Thesis appendix tables

Structure:
    RISK_TRANSLATIONS = {
        'canonical_swedish': 'English Translation',
    }

Usage:
    from scripts.dictionaries.risk_translations import RISK_TRANSLATIONS, translate_term

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-04-27
"""

from typing import Dict, Optional

# =============================================================================
# TRANSLATIONS: Swedish canonical → English
# =============================================================================

RISK_TRANSLATIONS: Dict[str, str] = {
    # =========================================================================
    # WEATHER / METEOROLOGICAL
    # =========================================================================
    'oversvamning': 'Flooding',
    'varmebolja': 'Heat wave',
    'torka': 'Drought',
    'storm': 'Storm',
    'isstorm': 'Ice storm',
    'snoovader': 'Snowstorm',
    'extrem_kyla': 'Extreme cold',
    'halka': 'Slippery conditions',
    'hagel': 'Hail',
    'blixtnedslag': 'Lightning strike',
    'lagvatten': 'Low water levels',

    # =========================================================================
    # GEOLOGICAL / GEOPHYSICAL
    # =========================================================================
    'skred': 'Landslide',
    'erosion': 'Erosion',
    'jordbaevning': 'Earthquake',
    'jordbavning': 'Earthquake',
    'tsunami': 'Tsunami',
    'vulkanutbrott': 'Volcanic eruption',

    # =========================================================================
    # FIRE (NATURAL)
    # =========================================================================
    'skogsbrand': 'Forest fire',
    'grasbrand': 'Grass fire',

    # =========================================================================
    # CLIMATE LONG-TERM
    # =========================================================================
    'klimatforandring': 'Climate change',
    'havsnivahojning': 'Sea level rise',
    'forsurning': 'Acidification',
    'grundvattenbrist': 'Groundwater shortage',

    # =========================================================================
    # SPACE WEATHER
    # =========================================================================
    'solstorm': 'Solar storm',

    # =========================================================================
    # BIOLOGICAL / HEALTH
    # =========================================================================
    'epidemi': 'Epidemic',
    'pandemi': 'Pandemic',
    'epizooti': 'Epizootic',
    'coronavirus': 'Coronavirus',
    'smittspridning': 'Disease transmission',
    'antibiotikaresistens': 'Antibiotic resistance',
    'fororenat_vatten': 'Contaminated water',
    'vaxtsjukdom': 'Plant disease',
    'invasiva_arter': 'Invasive species',

    # =========================================================================
    # FIRE (TECHNICAL/URBAN)
    # =========================================================================
    'byggnadsbrand': 'Building fire',
    'explosion': 'Explosion',

    # =========================================================================
    # TRANSPORT ACCIDENTS
    # =========================================================================
    'trafikolycka': 'Traffic accident',
    'tagolycka': 'Train accident',
    'bussolycka': 'Bus accident',
    'farligt_godsolycka': 'Hazardous goods accident',
    'fartygsolycka': 'Ship accident',
    'flygolycka': 'Aviation accident',
    'tunnelolycka': 'Tunnel accident',

    # =========================================================================
    # INDUSTRIAL ACCIDENTS
    # =========================================================================
    'industriolycka': 'Industrial accident',
    'olycka_farlig_verksamhet': 'Accident at hazardous facility',
    'dammbrott': 'Dam failure',
    'byggnadskollaps': 'Building collapse',
    'karnteknisk_olycka': 'Nuclear accident',

    # =========================================================================
    # CBRN
    # =========================================================================
    'kemisk_olycka': 'Chemical accident',
    'biologisk_olycka': 'Biological accident',
    'radiologisk_olycka': 'Radiological accident',
    'cbrn_handelse': 'CBRN incident',

    # =========================================================================
    # INFRASTRUCTURE DISRUPTIONS
    # =========================================================================
    'elavbrott': 'Power outage',
    'fjarrvarmebrott': 'District heating failure',
    'vattenforsorjning': 'Water supply disruption',
    'avloppsbrott': 'Sewage system failure',
    'avfallshantering': 'Waste management disruption',
    'it_teleavbrott': 'IT/telecom outage',
    'transportavbrott': 'Transport disruption',
    'drivmedelsbrist': 'Fuel shortage',
    'livsmedelsforsorjning': 'Food supply disruption',
    'personalbortfall': 'Staff shortage',
    'storda_leveranskedjor': 'Supply chain disruption',
    'finansiella_storningar': 'Financial system disruption',
    'satellitstorningar': 'Satellite system disruption',
    'elektromagnetiska_hot': 'Electromagnetic threats',

    # =========================================================================
    # SUPPLY CHAIN / CRITICAL SUPPLIES
    # =========================================================================
    'lakemedelsbrist': 'Medicine shortage',
    'medicinteknisk_brist': 'Medical equipment shortage',
    'ravarubrist': 'Raw material shortage',

    # =========================================================================
    # CYBER THREATS
    # =========================================================================
    'cyberattack': 'Cyberattack',
    'dataintrang': 'Data breach',
    'ddos_attack': 'DDoS attack',
    'ransomware': 'Ransomware',
    'it_sabotage': 'IT sabotage',

    # =========================================================================
    # TERRORISM / VIOLENCE
    # =========================================================================
    'terrorism': 'Terrorism',
    'hot_och_vald': 'Threats and violence',
    'pagaende_dodligt_vald': 'Active shooter',
    'valdsbejakande_extremism': 'Violent extremism',

    # =========================================================================
    # ESPIONAGE / SABOTAGE / HYBRID
    # =========================================================================
    'sabotage': 'Sabotage',
    'spionage': 'Espionage',
    'insiderhot': 'Insider threats',
    'desinformation': 'Disinformation',
    'hybridhot': 'Hybrid threats',
    'hot_mot_demokrati': 'Threats to democracy',

    # =========================================================================
    # MILITARY / ARMED CONFLICT
    # =========================================================================
    'vapnat_angrepp': 'Armed attack',
    'vapnat_angrepp_naromradet': 'Armed attack in vicinity',
    'fjarrangrepp': 'Long-range attack',
    'strid_svenskt_territorium': 'Combat on Swedish territory',
    'blockad': 'Blockade',
    'militar_konflikt_naromradet': 'Military conflict in vicinity',

    # =========================================================================
    # ORGANIZED CRIME
    # =========================================================================
    'organiserad_brottslighet': 'Organized crime',
    'vandalism': 'Vandalism',
    'manniskohandel': 'Human trafficking',
    'narkotikabrottslighet': 'Drug crime',
    'gangkriminalitet': 'Gang crime',
    'sexualbrott': 'Sexual crime',
    'stold_ran': 'Theft/robbery',
    'trafikbrott': 'Traffic offense',

    # =========================================================================
    # SOCIAL RISKS
    # =========================================================================
    'social_oro': 'Social unrest',
    'flyktingstrom': 'Refugee influx',
    'forsvunnen_person': 'Missing person',
    'befolkningsokning': 'Population growth',
    'suicid': 'Suicide',
    'drunkning': 'Drowning',
    'fallolycka': 'Fall accident',
    'forgiftning': 'Poisoning',

    # =========================================================================
    # ENVIRONMENTAL POLLUTION
    # =========================================================================
    'oljeutslapp': 'Oil spill',
    'kemikalieutslapp': 'Chemical spill',
    'miljoforening': 'Environmental pollution',

    # =========================================================================
    # ECONOMIC
    # =========================================================================
    'ekonomisk_kris': 'Economic crisis',
    'arbetsloshet': 'Unemployment',
    'inflation': 'Inflation',
}


# =============================================================================
# CATEGORY TRANSLATIONS
# =============================================================================

CATEGORY_TRANSLATIONS: Dict[str, str] = {
    # MSB taxonomy (Tier 2)
    'naturrisker': 'Natural risks',
    'tekniska_risker': 'Technical risks',
    'antagonistiska_risker': 'Antagonistic risks',

    # Subcategories
    'vader_meteorologi': 'Weather/Meteorological',
    'geologi_geofysik': 'Geological/Geophysical',
    'brand_natur': 'Fire (natural)',
    'klimat_langsiktigt': 'Climate (long-term)',
    'rymdvader': 'Space weather',
    'biologiska_halsa': 'Biological/Health',
    'brand_teknisk': 'Fire (technical)',
    'transportolyckor': 'Transport accidents',
    'industriolyckor': 'Industrial accidents',
    'cbrn': 'CBRN',
    'infrastrukturavbrott': 'Infrastructure disruptions',
    'forsorjning': 'Supply chain',
    'cyberhot': 'Cyber threats',
    'terrorism_vald': 'Terrorism/Violence',
    'spionage_sabotage': 'Espionage/Sabotage',
    'militar_konflikt': 'Military conflict',
    'organiserad_brottslighet': 'Organized crime',
    'sociala_risker': 'Social risks',
    'miljofororeningar': 'Environmental pollution',
    'ekonomi': 'Economic',
}


# =============================================================================
# ACTOR TRANSLATIONS (for consistency)
# =============================================================================

ACTOR_TRANSLATIONS: Dict[str, str] = {
    'kommun': 'Municipality',
    'länsstyrelse': 'Prefecture',
    'lansstyrelse': 'Prefecture',
    'MCF': 'MSB',
}


# =============================================================================
# STEMMED → CANONICAL MAPPING
# =============================================================================
# Maps stemmed term forms (as used in term_document_matrix.csv) to canonical names.
# Generated from RISK_TERMS variants using Swedish Snowball stemmer.

STEMMED_TO_CANONICAL: Dict[str, str] = {
    'antibiotikaresist': 'antibiotikaresistens',
    'arbetslös': 'arbetsloshet',
    'avloppsbrot': 'avloppsbrott',
    'avloppshaveri': 'avloppsbrott',
    'avloppssystem': 'avloppsbrott',
    'betalningssystem': 'finansiella_storningar',
    'biologisk_olyck': 'biologisk_olycka',
    'biologisk_utsläpp': 'biologisk_olycka',
    'blixt': 'blixtnedslag',
    'blixtnedslag': 'blixtnedslag',
    'brand': 'byggnadsbrand',
    'brist_läkemedel': 'lakemedelsbrist',
    'brist_medicinteknisk_utrustning': 'medicinteknisk_brist',
    'brist_på_läkemedel': 'lakemedelsbrist',
    'brist_på_medicinteknisk_utrustning': 'medicinteknisk_brist',
    'brokollap': 'byggnadskollaps',
    'bränd': 'byggnadsbrand',
    'bränslebrist': 'drivmedelsbrist',
    'bussolyck': 'bussolycka',
    'byggnadsbrand': 'byggnadsbrand',
    'byggnadskollap': 'byggnadskollaps',
    'byggnadsr': 'byggnadskollaps',
    'båtolyck': 'fartygsolycka',
    'cbrn': 'cbrn_handelse',
    'cbrn-händ': 'cbrn_handelse',
    'cbrne': 'cbrn_handelse',
    'civil_oro': 'social_oro',
    'coron': 'coronavirus',
    'coronavirus': 'coronavirus',
    'coronaviruset': 'coronavirus',
    'covid': 'coronavirus',
    'cyberattack': 'cyberattack',
    'dammbrot': 'dammbrott',
    'dataförlust': 'it_teleavbrott',
    'dataintrång': 'dataintrang',
    'ddos': 'ddos_attack',
    'ddos-attack': 'ddos_attack',
    'desinformation': 'desinformation',
    'distributionsstörning': 'transportavbrott',
    'djursjukdom': 'epizooti',
    'dricksvat': 'vattenforsorjning',
    'dricksvattenförsörjning': 'vattenforsorjning',
    'drivmedel': 'drivmedelsbrist',
    'drivmedelsbrist': 'drivmedelsbrist',
    'drivmedelsförsörjning': 'drivmedelsbrist',
    'drivsmedelsbrist': 'drivmedelsbrist',
    'droghandel': 'narkotikabrottslighet',
    'drönarangrepp': 'fjarrangrepp',
    'effektbrist': 'elavbrott',
    'ekonomisk_block': 'blockad',
    'ekonomisk_kris': 'ekonomisk_kris',
    'ekonomisk_nedgång': 'ekonomisk_kris',
    'elavbrot': 'elavbrott',
    'elbrist': 'elavbrott',
    'elektromagnetisk_hot': 'elektromagnetiska_hot',
    'elektromagnetisk_pul': 'elektromagnetiska_hot',
    'elektromagnetisk_störning': 'elektromagnetiska_hot',
    'elektronisk_kommunikation': 'it_teleavbrott',
    'elförsörjning': 'elavbrott',
    'emp': 'elektromagnetiska_hot',
    'energiförsörjning': 'elavbrott',
    'epidemi': 'epidemi',
    'epizooti': 'epizooti',
    'erosion': 'erosion',
    'explosion': 'explosion',
    'extrem_kyl': 'extrem_kyla',
    'extrem_värm': 'varmebolja',
    'extremvärm': 'varmebolja',
    'falsk_information': 'desinformation',
    'far_god': 'farligt_godsolycka',
    'far_godsolyck': 'farligt_godsolycka',
    'far_verksam': 'olycka_farlig_verksamhet',
    'fartygskollision': 'fartygsolycka',
    'fartygsolyck': 'fartygsolycka',
    'finansiell_störning': 'finansiella_storningar',
    'finanskris': 'ekonomisk_kris',
    'fjärrangrepp': 'fjarrangrepp',
    'fjärrvap': 'fjarrangrepp',
    'fjärrvärmeavbrot': 'fjarrvarmebrott',
    'fjärrvärmebrot': 'fjarrvarmebrott',
    'flodvåg': 'tsunami',
    'flyghaveri': 'flygolycka',
    'flygolyck': 'flygolycka',
    'flyktingkris': 'flyktingstrom',
    'flyktingström': 'flyktingstrom',
    'flyktingströmm': 'flyktingstrom',
    'främm_art': 'invasiva_arter',
    'främm_underrättelsetjänst': 'spionage',
    'föroren_ell_smitt_vatt': 'fororenat_vatten',
    'föroren_smitt_vatt': 'fororenat_vatten',
    'föroren_vatt': 'fororenat_vatten',
    'förorening': 'miljoforening',
    'försurning': 'forsurning',
    'försvun_bruk': 'forsvunnen_person',
    'försvun_person': 'forsvunnen_person',
    'försörjningsstörning': 'storda_leveranskedjor',
    'gasexplosion': 'explosion',
    'gift_ämn': 'kemisk_olycka',
    'giftutsläpp': 'kemisk_olycka',
    'gnss-störning': 'satellitstorningar',
    'gps-störning': 'satellitstorningar',
    'grundvattenbrist': 'grundvattenbrist',
    'grundvattennivå': 'grundvattenbrist',
    'gräsbrand': 'grasbrand',
    'gräsbränd': 'grasbrand',
    'gängkonflik': 'gangkriminalitet',
    'gängkriminalitet': 'gangkriminalitet',
    'gängvåld': 'gangkriminalitet',
    'hackerattack': 'dataintrang',
    'hagel': 'hagel',
    'halk': 'halka',
    'handelsblock': 'blockad',
    'havsförsurning': 'forsurning',
    'havsnivå': 'havsnivahojning',
    'havsnivåhöjning': 'havsnivahojning',
    'hot_demokrati': 'hot_mot_demokrati',
    'hot_demokrati_mänsk_fri-': 'hot_mot_demokrati',
    'hot_demokratin': 'hot_mot_demokrati',
    'hot_mot_demokrati': 'hot_mot_demokrati',
    'hot_mot_demokrati_och_mänsk_fri-_och_rätt': 'hot_mot_demokrati',
    'hot_mot_demokratin': 'hot_mot_demokrati',
    'hot_mot_mänsk_fri-_och_rätt': 'hot_mot_demokrati',
    'hot_mänsk_fri-': 'hot_mot_demokrati',
    'hot_och_våld': 'hot_och_vald',
    'hot_våld': 'hot_och_vald',
    'hybrid_hot': 'hybridhot',
    'hybridhot': 'hybridhot',
    'hög_flöd': 'oversvamning',
    'högvat': 'oversvamning',
    'inbrot': 'vandalism',
    'industriolyck': 'industriolycka',
    'inflation': 'inflation',
    'influensapandemi': 'pandemi',
    'informationspåverkan': 'desinformation',
    'int_flyktingström': 'flyktingstrom',
    'int_flyktingströmm': 'flyktingstrom',
    'invasion': 'strid_svenskt_territorium',
    'invasiv_art': 'invasiva_arter',
    'isbildning': 'isstorm',
    'isbildning_isstorm': 'isstorm',
    'isbildning_och_isstorm': 'isstorm',
    'isstorm': 'isstorm',
    'it-_och_teleavbrot': 'it_teleavbrott',
    'it-_teleavbrot': 'it_teleavbrott',
    'it-avbrot': 'it_teleavbrott',
    'it-bortfall': 'it_teleavbrott',
    'it-sabotag': 'it_sabotage',
    'it/teleavbrot': 'it_teleavbrott',
    'jordbävning': 'jordbaevning',
    'jordskalv': 'jordbaevning',
    'jordskred': 'skred',
    'järnvägsolyck': 'tagolycka',
    'kemikalieolyck': 'industriolycka',
    'kemikaliespill': 'kemikalieutslapp',
    'kemikalieutsläpp': 'kemikalieutslapp',
    'kemisk_olyck': 'kemisk_olycka',
    'kemisk_utsläpp': 'kemisk_olycka',
    'kemiska,_biologiska,_radiologisk_nukleär_hot': 'cbrn_handelse',
    'kemiska,_biologiska,_radiologisk_och_nukleär_hot': 'cbrn_handelse',
    'klimatförändring': 'klimatforandring',
    'kommunikationsavbrot': 'it_teleavbrott',
    'komponentbrist': 'ravarubrist',
    'konflik_i_östersjöområdet': 'militar_konflikt_naromradet',
    'konflik_östersjöområdet': 'militar_konflikt_naromradet',
    'kontaminer_dricksvat': 'fororenat_vatten',
    'kontaminer_vatt': 'fororenat_vatten',
    'kraftavbrot': 'elavbrott',
    'kraftförsörjning': 'elavbrott',
    'krig_i_närområdet': 'militar_konflikt_naromradet',
    'krig_närområdet': 'militar_konflikt_naromradet',
    'kriminalitet': 'organiserad_brottslighet',
    'kärnkraftsolyck': 'karnteknisk_olycka',
    'kärnteknisk_olyck': 'karnteknisk_olycka',
    'köldknäpp': 'extrem_kyla',
    'leveransstörning': 'storda_leveranskedjor',
    'livsmedelsbrist': 'livsmedelsforsorjning',
    'livsmedelsförsörjning': 'livsmedelsforsorjning',
    'logistikavbrot': 'transportavbrott',
    'luftförorening': 'miljoforening',
    'läkemedelsbrist': 'lakemedelsbrist',
    'läkemedelsförsörjning': 'lakemedelsbrist',
    'låg_flöd': 'lagvatten',
    'lågkonjuktur': 'ekonomisk_kris',
    'lågvat': 'lagvatten',
    'markförorening': 'miljoforening',
    'markstrid': 'strid_svenskt_territorium',
    'materialbrist': 'ravarubrist',
    'matförsörjning': 'livsmedelsforsorjning',
    'medicinbrist': 'lakemedelsbrist',
    'medicinteknisk_brist': 'medicinteknisk_brist',
    'mers-cov': 'coronavirus',
    'migrationskris': 'flyktingstrom',
    'militär_konflik_i_närområdet': 'militar_konflikt_naromradet',
    'militär_konflik_närområdet': 'militar_konflikt_naromradet',
    'miljöförorening': 'miljoforening',
    'missilangrepp': 'fjarrangrepp',
    'misstro': 'social_oro',
    'mobilnätsavbrot': 'it_teleavbrott',
    'människohandel': 'manniskohandel',
    'narkotik': 'narkotikabrottslighet',
    'narkotikabrotts': 'narkotikabrottslighet',
    'narkotikahandel': 'narkotikabrottslighet',
    'nätattack': 'cyberattack',
    'nätverksavbrot': 'it_teleavbrott',
    'ockupation': 'strid_svenskt_territorium',
    'oljespill': 'oljeutslapp',
    'oljeutsläpp': 'oljeutslapp',
    'olyck_far_god': 'farligt_godsolycka',
    'olyck_far_verksam': 'olycka_farlig_verksamhet',
    'olyck_med_far_god': 'farligt_godsolycka',
    'olyck_med_nukleär_ämn': 'karnteknisk_olycka',
    'olyck_med_radioaktiv_ämn': 'karnteknisk_olycka',
    'olyck_med_transport_av_far_god': 'farligt_godsolycka',
    'olyck_nukleär_ämn': 'karnteknisk_olycka',
    'olyck_radioaktiv_ämn': 'karnteknisk_olycka',
    'olyck_transport_far_god': 'farligt_godsolycka',
    'olyck_vid_far_verksam': 'olycka_farlig_verksamhet',
    'organiser_brotts': 'organiserad_brottslighet',
    'pandemi': 'pandemi',
    'pdv': 'pagaende_dodligt_vald',
    'personalbortfall': 'personalbortfall',
    'personalbrist': 'personalbortfall',
    'pågåend_död_våld': 'pagaende_dodligt_vald',
    'påverkanskampanj': 'desinformation',
    'påverkansoperation': 'desinformation',
    'radioaktivt_utsläpp': 'radiologisk_olycka',
    'radiologisk_olyck': 'radiologisk_olycka',
    'ransomw': 'ransomware',
    'ras': 'skred',
    'ras_ell_skred': 'skred',
    'ras_och_skred': 'skred',
    'ras_skred': 'skred',
    'recession': 'ekonomisk_kris',
    'regional_konflik': 'militar_konflikt_naromradet',
    'resistent_bakteri': 'antibiotikaresistens',
    'robotangrepp': 'fjarrangrepp',
    'råvarubrist': 'ravarubrist',
    'sabotag': 'sabotage',
    'sakn_person': 'forsvunnen_person',
    'sars-cov': 'coronavirus',
    'satellitstörning': 'satellitstorningar',
    'sjukdomsutbrot': 'smittspridning',
    'sjukvårdsmaterial': 'medicinteknisk_brist',
    'sjunk_grundvattennivå': 'grundvattenbrist',
    'sjöblock': 'blockad',
    'skadegör': 'vandalism',
    'skadeinsek': 'vaxtsjukdom',
    'skjutning': 'gangkriminalitet',
    'skogsbrand': 'skogsbrand',
    'skogsbränd': 'skogsbrand',
    'skred': 'skred',
    'skyddsutrustning': 'medicinteknisk_brist',
    'skyfall': 'oversvamning',
    'slamskred': 'skred',
    'smitt': 'smittspridning',
    'smitt_vatt': 'fororenat_vatten',
    'smittsam_sjukdom': 'smittspridning',
    'smittsamm_sjukdom': 'smittspridning',
    'smittspridning': 'smittspridning',
    'smittämn': 'biologisk_olycka',
    'snöoväd': 'snoovader',
    'social_oro': 'social_oro',
    'solstorm': 'solstorm',
    'spionag': 'spionage',
    'sprängning': 'gangkriminalitet',
    'spårbundn_olyck': 'bussolycka',
    'stig_havsnivå': 'havsnivahojning',
    'stor_snömäng': 'snoovader',
    'storbrand': 'byggnadsbrand',
    'storm': 'storm',
    'stormfällning': 'storm',
    'strid_på_svensk_territorium': 'strid_svenskt_territorium',
    'strid_svensk_territorium': 'strid_svenskt_territorium',
    'strålning': 'radiologisk_olycka',
    'strålningsolyck': 'radiologisk_olycka',
    'strömavbrot': 'elavbrott',
    'störd_leveranskedj': 'storda_leveranskedjor',
    'störning_finansiell_system': 'finansiella_storningar',
    'störning_finansiell_systemet': 'finansiella_storningar',
    'störning_i_det_finansiell_systemet': 'finansiella_storningar',
    'störning_i_finansiell_system': 'finansiella_storningar',
    'störning_i_satellitbaser_navigationssystem': 'satellitstorningar',
    'störning_i_satellitsystem': 'satellitstorningar',
    'störning_satellitbaser_navigationssystem': 'satellitstorningar',
    'störning_satellitsystem': 'satellitstorningar',
    'svampsjukdom': 'vaxtsjukdom',
    'systemfel': 'it_teleavbrott',
    'teleavbrot': 'it_teleavbrott',
    'telebrot': 'it_teleavbrott',
    'telenätavbrot': 'it_teleavbrott',
    'terr': 'terrorism',
    'terrorattent': 'terrorism',
    'terrorhandling': 'terrorism',
    'terrorhot': 'terrorism',
    'terrorism': 'terrorism',
    'tork': 'torka',
    'trafficking': 'manniskohandel',
    'trafikolyck': 'trafikolycka',
    'transport_av_far_god': 'farligt_godsolycka',
    'transport_far_god': 'farligt_godsolycka',
    'transportavbrot': 'transportavbrott',
    'transportstörning': 'transportavbrott',
    'tsunami': 'tsunami',
    'tunnelolyck': 'tunnelolycka',
    'tågolyck': 'tagolycka',
    'underrättelseverksam': 'spionage',
    'upplopp': 'social_oro',
    'utländsk_underrättelseverksam': 'spionage',
    'utpressningsvirus': 'ransomware',
    'utsläpp': 'miljoforening',
    'vandalism': 'vandalism',
    'vansinnesdåd': 'valdsbejakande_extremism',
    'vattenbrist': 'grundvattenbrist',
    'vattenförorening': 'miljoforening',
    'vattenförsörjning': 'vattenforsorjning',
    'vattenläck': 'vattenforsorjning',
    'vilseled_information': 'desinformation',
    'vulkanask': 'vulkanutbrott',
    'vulkanisk_aktivitet': 'vulkanutbrott',
    'vulkanutbrot': 'vulkanutbrott',
    'vägolyck': 'trafikolycka',
    'väpn_angrepp': 'vapnat_angrepp',
    'väpn_angrepp_i_närområdet': 'vapnat_angrepp_naromradet',
    'väpn_angrepp_närområdet': 'vapnat_angrepp_naromradet',
    'värmebölj': 'varmebolja',
    'växtsjukdom': 'vaxtsjukdom',
    'växtskadegör': 'vaxtsjukdom',
    'våldsbejak_extremism': 'valdsbejakande_extremism',
    'våldsbejak_extremism_ell_vansinnesdåd': 'valdsbejakande_extremism',
    'våldsbejak_extremism_vansinnesdåd': 'valdsbejakande_extremism',
    'våldsbrot': 'hot_och_vald',
    'zoono': 'epizooti',
    'zoonos': 'epizooti',
    'översvämning': 'oversvamning',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def translate_term(swedish_term: str) -> str:
    """
    Translate a Swedish term to English.

    Handles both canonical terms and stemmed terms (as used in term_document_matrix).
    Returns original if no translation exists.
    """
    # Direct lookup first
    if swedish_term in RISK_TRANSLATIONS:
        return RISK_TRANSLATIONS[swedish_term]

    # Try resolving stemmed form to canonical
    canonical = STEMMED_TO_CANONICAL.get(swedish_term)
    if canonical and canonical in RISK_TRANSLATIONS:
        return RISK_TRANSLATIONS[canonical]

    return swedish_term


def translate_category(swedish_category: str) -> str:
    """
    Translate a Swedish category to English.
    Returns original if no translation exists.
    """
    return CATEGORY_TRANSLATIONS.get(swedish_category, swedish_category)


def translate_actor(swedish_actor: str) -> str:
    """
    Translate a Swedish actor name to English.
    Returns original if no translation exists.
    """
    return ACTOR_TRANSLATIONS.get(swedish_actor, swedish_actor)


def get_appendix_table() -> str:
    """
    Generate a LaTeX table for the thesis appendix.

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{longtable}{ll}",
        r"\caption{Risk Term Translations (Swedish to English)} \\",
        r"\toprule",
        r"\textbf{Swedish (canonical)} & \textbf{English} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{2}{c}{\tablename\ \thetable{} -- continued} \\",
        r"\toprule",
        r"\textbf{Swedish (canonical)} & \textbf{English} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{2}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for swedish, english in sorted(RISK_TRANSLATIONS.items()):
        # Escape underscores for LaTeX
        swedish_escaped = swedish.replace('_', r'\_')
        lines.append(f"{swedish_escaped} & {english} \\\\")

    lines.append(r"\end{longtable}")
    return '\n'.join(lines)


def get_markdown_table() -> str:
    """
    Generate a Markdown table for documentation.

    Returns:
        Markdown table string
    """
    lines = [
        "| Swedish (canonical) | English |",
        "|---------------------|---------|",
    ]

    for swedish, english in sorted(RISK_TRANSLATIONS.items()):
        lines.append(f"| {swedish} | {english} |")

    return '\n'.join(lines)


if __name__ == '__main__':
    print(f"Risk Translations: {len(RISK_TRANSLATIONS)} terms")
    print(f"Category Translations: {len(CATEGORY_TRANSLATIONS)} categories")
    print()
    print("Sample translations:")
    for term in ['oversvamning', 'terrorism', 'cyberattack', 'pandemi', 'elavbrott']:
        print(f"  {term} → {translate_term(term)}")
