#!/usr/bin/env python3
"""
Risk Diffusion Analysis

Tracks when risk terms first appear across entities and detects
synchronous adoption patterns. Compares diffusion between actor types
(municipalities, prefectures, MCF) to test top-down vs. bottom-up
diffusion hypotheses.

Input:  term_document_matrix.csv (from term_document_matrix.py)
Output: adoption curves, heatmaps, lead-lag analysis, Gini coefficients

Usage:
    python risk_diffusion_analysis.py \\
        --input results/01_bow_analysis/term_matrices/term_document_matrix.csv \\
        --output results/01_bow_analysis/diffusion/

    python risk_diffusion_analysis.py \\
        --input results/01_bow_analysis/term_matrices/term_document_matrix.csv \\
        --output results/01_bow_analysis/diffusion/ \\
        --spike-threshold 0.15 --verbose

Requirements:
    pip install pandas numpy matplotlib seaborn
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# =============================================================================
# CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set1")

METADATA_COLS = ['file', 'actor', 'entity', 'year', 'wave']

ACTOR_TRANSLATIONS = {
    'kommun': 'Municipality',
    'lansstyrelse': 'Prefecture',
    'länsstyrelse': 'Prefecture',  # Handle both spellings
    'MCF': 'MCF',
}

# Key external events for contextual annotation
EXTERNAL_EVENTS = {
    2016: 'MSBFS 2015:5',
    2018: 'Heatwave/fires',
    2020: 'COVID-19',
    2022: 'Ukraine war',
}

# Län (county/prefecture) codes and names
LAN_CODES = {
    1: 'Stockholm', 2: 'Uppsala', 3: 'Södermanland', 4: 'Östergötland',
    5: 'Jönköping', 6: 'Kronoberg', 7: 'Kalmar', 8: 'Gotland',
    9: 'Blekinge', 10: 'Skåne', 11: 'Halland', 12: 'Västra Götaland',
    13: 'Värmland', 14: 'Örebro', 15: 'Västmanland', 16: 'Dalarna',
    17: 'Gävleborg', 18: 'Västernorrland', 19: 'Jämtland',
    20: 'Västerbotten', 21: 'Norrbotten',
}

# Municipality to län mapping (lowercase for matching)
MUNICIPALITY_TO_LAN = {
    # 1 - Stockholm
    'botkyrka': 1, 'danderyd': 1, 'ekerö': 1, 'haninge': 1, 'huddinge': 1,
    'järfälla': 1, 'lidingö': 1, 'nacka': 1, 'norrtälje': 1, 'nykvarn': 1,
    'nynäshamn': 1, 'salem': 1, 'sigtuna': 1, 'sollentuna': 1, 'solna': 1,
    'stockholm': 1, 'sundbyberg': 1, 'södertälje': 1, 'tyresö': 1, 'täby': 1,
    'upplands-bro': 1, 'upplands väsby': 1, 'vallentuna': 1, 'vaxholm': 1,
    'värmdö': 1, 'österåker': 1,
    # 2 - Uppsala
    'enköping': 2, 'heby': 2, 'håbo': 2, 'knivsta': 2, 'tierp': 2,
    'uppsala': 2, 'älvkarleby': 2, 'östhammar': 2,
    # 3 - Södermanland
    'eskilstuna': 3, 'flen': 3, 'gnesta': 3, 'katrineholm': 3, 'nyköping': 3,
    'oxelösund': 3, 'strängnäs': 3, 'trosa': 3, 'vingåker': 3,
    # 4 - Östergötland
    'boxholm': 4, 'finspång': 4, 'kinda': 4, 'linköping': 4, 'mjölby': 4,
    'motala': 4, 'norrköping': 4, 'söderköping': 4, 'vadstena': 4,
    'valdemarsvik': 4, 'ydre': 4, 'åtvidaberg': 4, 'ödeshög': 4,
    # 5 - Jönköping
    'aneby': 5, 'eksjö': 5, 'gislaved': 5, 'gnosjö': 5, 'habo': 5,
    'jönköping': 5, 'mullsjö': 5, 'nässjö': 5, 'sävsjö': 5, 'tranås': 5,
    'vaggeryd': 5, 'vetlanda': 5, 'värnamo': 5,
    # 6 - Kronoberg
    'alvesta': 6, 'lessebo': 6, 'ljungby': 6, 'markaryd': 6, 'tingsryd': 6,
    'uppvidinge': 6, 'växjö': 6, 'älmhult': 6,
    # 7 - Kalmar
    'borgholm': 7, 'emmaboda': 7, 'hultsfred': 7, 'högsby': 7, 'kalmar': 7,
    'mönsterås': 7, 'mörbylånga': 7, 'nybro': 7, 'oskarshamn': 7, 'torsås': 7,
    'vimmerby': 7, 'västervik': 7,
    # 8 - Gotland
    'gotland': 8,
    # 9 - Blekinge
    'karlshamn': 9, 'karlskrona': 9, 'olofström': 9, 'ronneby': 9, 'sölvesborg': 9,
    # 10 - Skåne
    'bjuv': 10, 'bromölla': 10, 'burlöv': 10, 'båstad': 10, 'eslöv': 10,
    'helsingborg': 10, 'hässleholm': 10, 'höganäs': 10, 'hörby': 10, 'höör': 10,
    'klippan': 10, 'kristianstad': 10, 'kävlinge': 10, 'landskrona': 10,
    'lomma': 10, 'lund': 10, 'malmö': 10, 'osby': 10, 'perstorp': 10,
    'simrishamn': 10, 'sjöbo': 10, 'skurup': 10, 'staffanstorp': 10, 'svalöv': 10,
    'svedala': 10, 'tomelilla': 10, 'trelleborg': 10, 'vellinge': 10, 'ystad': 10,
    'åstorp': 10, 'ängelholm': 10, 'örkelljunga': 10, 'östra göinge': 10,
    # 11 - Halland
    'falkenberg': 11, 'halmstad': 11, 'hylte': 11, 'kungsbacka': 11,
    'laholm': 11, 'varberg': 11,
    # 12 - Västra Götaland
    'ale': 12, 'alingsås': 12, 'bengtsfors': 12, 'bollebygd': 12, 'borås': 12,
    'dals-ed': 12, 'essunga': 12, 'falköping': 12, 'färgelanda': 12,
    'grästorp': 12, 'gullspång': 12, 'göteborg': 12, 'götene': 12,
    'herrljunga': 12, 'hjo': 12, 'härryda': 12, 'karlsborg': 12, 'kungälv': 12,
    'lerum': 12, 'lidköping': 12, 'lilla edet': 12, 'lysekil': 12,
    'mariestad': 12, 'mark': 12, 'mellerud': 12, 'munkedal': 12, 'mölndal': 12,
    'orust': 12, 'partille': 12, 'skara': 12, 'skövde': 12, 'sotenäs': 12,
    'stenungsund': 12, 'strömstad': 12, 'svenljunga': 12, 'tanum': 12,
    'tibro': 12, 'tidaholm': 12, 'tjörn': 12, 'tranemo': 12, 'trollhättan': 12,
    'töreboda': 12, 'uddevalla': 12, 'ulricehamn': 12, 'vara': 12,
    'vårgårda': 12, 'vänersborg': 12, 'åmål': 12, 'öckerö': 12,
    # 13 - Värmland
    'arvika': 13, 'eda': 13, 'filipstad': 13, 'forshaga': 13, 'grums': 13,
    'hagfors': 13, 'hammarö': 13, 'karlstad': 13, 'kil': 13, 'kristinehamn': 13,
    'munkfors': 13, 'storfors': 13, 'sunne': 13, 'säffle': 13, 'torsby': 13,
    'årjäng': 13,
    # 14 - Örebro
    'askersund': 14, 'degerfors': 14, 'hallsberg': 14, 'hällefors': 14,
    'karlskoga': 14, 'kumla': 14, 'laxå': 14, 'lekeberg': 14, 'lindesberg': 14,
    'ljusnarsberg': 14, 'nora': 14, 'örebro': 14,
    # 15 - Västmanland
    'arboga': 15, 'fagersta': 15, 'hallstahammar': 15, 'kungsör': 15,
    'köping': 15, 'norberg': 15, 'sala': 15, 'skinnskatteberg': 15,
    'surahammar': 15, 'västerås': 15,
    # 16 - Dalarna
    'avesta': 16, 'borlänge': 16, 'falun': 16, 'gagnef': 16, 'hedemora': 16,
    'leksand': 16, 'ludvika': 16, 'malung-sälen': 16, 'mora': 16, 'orsa': 16,
    'rättvik': 16, 'smedjebacken': 16, 'säter': 16, 'vansbro': 16, 'älvdalen': 16,
    # 17 - Gävleborg
    'bollnäs': 17, 'gävle': 17, 'hofors': 17, 'hudiksvall': 17, 'ljusdal': 17,
    'nordanstig': 17, 'ockelbo': 17, 'ovanåker': 17, 'sandviken': 17,
    'söderhamn': 17,
    # 18 - Västernorrland
    'härnösand': 18, 'kramfors': 18, 'sollefteå': 18, 'sundsvall': 18,
    'timrå': 18, 'ånge': 18, 'örnsköldsvik': 18,
    # 19 - Jämtland
    'berg': 19, 'bräcke': 19, 'härjedalen': 19, 'krokom': 19, 'ragunda': 19,
    'strömsund': 19, 'åre': 19, 'östersund': 19,
    # 20 - Västerbotten
    'bjurholm': 20, 'dorotea': 20, 'lycksele': 20, 'malå': 20, 'nordmaling': 20,
    'norsjö': 20, 'robertsfors': 20, 'skellefteå': 20, 'sorsele': 20,
    'storuman': 20, 'umeå': 20, 'vilhelmina': 20, 'vindeln': 20, 'vännäs': 20,
    'åsele': 20,
    # 21 - Norrbotten
    'arjeplog': 21, 'arvidsjaur': 21, 'boden': 21, 'gällivare': 21,
    'haparanda': 21, 'jokkmokk': 21, 'kalix': 21, 'kiruna': 21, 'luleå': 21,
    'pajala': 21, 'piteå': 21, 'älvsbyn': 21, 'överkalix': 21, 'övertorneå': 21,
}


def get_lan_for_entity(entity: str, actor: str) -> int | None:
    """
    Get the län code for an entity.
    For municipalities, looks up in MUNICIPALITY_TO_LAN.
    For prefectures (länsstyrelse), extracts län from name.
    Returns None for MCF or unmatched entities.
    """
    if actor == 'MCF':
        return None

    # Normalize Unicode to NFC (composed form) for consistent matching
    entity_lower = unicodedata.normalize('NFC', entity.lower().strip())

    # For municipalities, direct lookup
    if actor == 'kommun':
        # Try direct match first
        if entity_lower in MUNICIPALITY_TO_LAN:
            return MUNICIPALITY_TO_LAN[entity_lower]
        # Try without "kommun" suffix
        entity_clean = entity_lower.replace(' kommun', '').replace('kommun', '').strip()
        if entity_clean in MUNICIPALITY_TO_LAN:
            return MUNICIPALITY_TO_LAN[entity_clean]
        # Try removing trailing 'n' (e.g., "Älvkarlebyn" -> "Älvkarleby")
        if entity_lower.endswith('n') and entity_lower[:-1] in MUNICIPALITY_TO_LAN:
            return MUNICIPALITY_TO_LAN[entity_lower[:-1]]
        return None

    # For prefectures, match län name
    if actor in ('lansstyrelse', 'länsstyrelse'):
        for lan_id, lan_name in LAN_CODES.items():
            lan_name_norm = unicodedata.normalize('NFC', lan_name.lower())
            if lan_name_norm in entity_lower or entity_lower in lan_name_norm:
                return lan_id
        return None

    return None


# Import term metadata for grouping by category
sys.path.insert(0, str(Path(__file__).parent))
from risk_dictionary_categories import RISK_DICTIONARY_CATEGORIES as RISK_DICTIONARY


def translate_actor(actor: str) -> str:
    """Translate actor names from Swedish to English."""
    return ACTOR_TRANSLATIONS.get(actor, actor)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare(input_path: Path) -> tuple:
    """Load term-document matrix."""
    df = pd.read_csv(input_path)
    term_cols = [c for c in df.columns if c not in METADATA_COLS]
    print(f"  Loaded {len(df)} documents, {len(term_cols)} terms")
    return df, term_cols


# =============================================================================
# FIRST APPEARANCE
# =============================================================================

def compute_first_appearances(
    df: pd.DataFrame, term_cols: list
) -> pd.DataFrame:
    """
    For each (entity, term), find the earliest year the term is mentioned.

    Parameters
    ----------
    df : pd.DataFrame
        Term-document matrix.
    term_cols : list[str]
        Term column names.

    Returns
    -------
    pd.DataFrame
        Columns: entity, actor, lan, term, first_year, is_left_censored.
        is_left_censored = True if the first appearance is in the
        entity's earliest available document.
        lan = county code (1-21) for municipalities and prefectures, None for MCF.
    """
    records = []

    for entity, group in df.groupby('entity'):
        group = group.sort_values('year')
        actor = group['actor'].iloc[0]
        earliest_year = group['year'].min()
        lan = get_lan_for_entity(entity, actor)

        for term in term_cols:
            # Find first year where count > 0
            present = group[group[term] > 0]
            if len(present) > 0:
                first_year = present['year'].min()
                is_censored = (first_year == earliest_year)
                records.append({
                    'entity': entity,
                    'actor': actor,
                    'lan': lan,
                    'term': term,
                    'first_year': first_year,
                    'is_left_censored': is_censored,
                })

    return pd.DataFrame(records)


# =============================================================================
# ADOPTION CURVES
# =============================================================================

def compute_adoption_curves(
    first_appearances: pd.DataFrame,
    df: pd.DataFrame,
    term_cols: list,
) -> pd.DataFrame:
    """
    For each term (and optionally per actor type), compute cumulative
    adoption fraction over time.

    Returns
    -------
    pd.DataFrame
        Columns: term, actor, year, cumulative_count, total_entities,
        cumulative_fraction.
    """
    all_years = sorted(df['year'].unique())
    records = []

    for actor_filter in [None] + list(df['actor'].unique()):
        if actor_filter is None:
            subset = first_appearances
            total_entities = df['entity'].nunique()
            actor_label = 'all'
        else:
            subset = first_appearances[first_appearances['actor'] == actor_filter]
            total_entities = df[df['actor'] == actor_filter]['entity'].nunique()
            actor_label = actor_filter

        if total_entities == 0:
            continue

        for term in term_cols:
            term_data = subset[subset['term'] == term]
            if len(term_data) == 0:
                continue

            for year in all_years:
                cum_count = (term_data['first_year'] <= year).sum()
                records.append({
                    'term': term,
                    'actor': actor_label,
                    'year': year,
                    'cumulative_count': cum_count,
                    'total_entities': total_entities,
                    'cumulative_fraction': cum_count / total_entities,
                })

    return pd.DataFrame(records)


# =============================================================================
# ADOPTION SPIKES
# =============================================================================

def detect_adoption_spikes(
    first_appearances: pd.DataFrame,
    df: pd.DataFrame,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Detect years with unusually high adoption of a term.

    A spike occurs when ≥ threshold fraction of entities first mention
    a term in the same year.

    Parameters
    ----------
    first_appearances : pd.DataFrame
        First appearance data.
    df : pd.DataFrame
        Full data (for total entity counts).
    threshold : float
        Minimum fraction to flag as a spike.

    Returns
    -------
    pd.DataFrame
        Detected spikes with columns: term, year, new_adopters,
        total_entities, adoption_fraction, is_spike.
    """
    total_entities = df['entity'].nunique()
    records = []

    for term, group in first_appearances.groupby('term'):
        yearly_counts = group.groupby('first_year').size()

        for year, count in yearly_counts.items():
            fraction = count / total_entities
            records.append({
                'term': term,
                'year': year,
                'new_adopters': count,
                'total_entities': total_entities,
                'adoption_fraction': fraction,
                'is_spike': fraction >= threshold,
            })

    spikes_df = pd.DataFrame(records)
    return spikes_df


# =============================================================================
# GINI COEFFICIENT
# =============================================================================

def compute_gini(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of an array.

    Low Gini = uniform/synchronous adoption.
    High Gini = unequal/gradual adoption.

    Parameters
    ----------
    values : np.ndarray
        Non-negative values.

    Returns
    -------
    float
        Gini coefficient in [0, 1].
    """
    values = np.sort(values)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def compute_gini_coefficients(
    first_appearances: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Gini coefficient of first-appearance years for each term.

    Returns
    -------
    pd.DataFrame
        Columns: term, gini, n_entities, mean_year, std_year, iqr_years.
    """
    records = []

    # Collapse pre-2015 adoptions into 2014 bucket
    first_appearances = first_appearances.copy()
    first_appearances.loc[first_appearances['first_year'] < 2015, 'first_year'] = 2014

    for term, group in first_appearances.groupby('term'):
        years = group['first_year'].values.astype(float)
        if len(years) < 2:
            continue

        # Normalize years to start from 0 for meaningful Gini calculation
        years_normalized = years - years.min()

        # Compute IQR (75th percentile - 25th percentile)
        q75, q25 = np.percentile(years, [75, 25])
        iqr = q75 - q25

        records.append({
            'term': term,
            'gini': compute_gini(years_normalized),
            'n_entities': len(years),
            'mean_year': years.mean(),
            'std_year': years.std(),
            'iqr_years': iqr,
        })

    return pd.DataFrame(records).sort_values('gini')


# =============================================================================
# LEAD-LAG ANALYSIS
# =============================================================================

def compute_lead_lag(first_appearances: pd.DataFrame) -> pd.DataFrame:
    """
    For each term, compare median first-appearance year between actor types.

    Returns
    -------
    pd.DataFrame
        Columns: term, median_year_kommun, median_year_lansstyrelse,
        median_year_MCF, lag_kommun_vs_lansstyrelse, lag_kommun_vs_MCF.
        Positive lag = municipalities adopt later (top-down diffusion).
    """
    records = []

    # Normalize actor names and collapse pre-2015 adoptions into 2014 bucket
    first_appearances = first_appearances.copy()
    first_appearances.loc[first_appearances['first_year'] < 2015, 'first_year'] = 2014
    first_appearances['actor'] = first_appearances['actor'].replace('länsstyrelse', 'lansstyrelse')

    for term, group in first_appearances.groupby('term'):
        medians = group.groupby('actor')['first_year'].median()

        row = {'term': term}
        for actor in ['kommun', 'lansstyrelse', 'MCF']:
            col = f'median_year_{actor}'
            row[col] = medians.get(actor, np.nan)
            row[f'n_{actor}'] = len(group[group['actor'] == actor])

        # Compute lags
        if not np.isnan(row.get('median_year_kommun', np.nan)):
            if not np.isnan(row.get('median_year_lansstyrelse', np.nan)):
                row['lag_kommun_vs_lansstyrelse'] = (
                    row['median_year_kommun'] - row['median_year_lansstyrelse']
                )
            if not np.isnan(row.get('median_year_MCF', np.nan)):
                row['lag_kommun_vs_MCF'] = (
                    row['median_year_kommun'] - row['median_year_MCF']
                )

        records.append(row)

    return pd.DataFrame(records)


# =============================================================================
# WITHIN-LÄN ANALYSIS
# =============================================================================

def compute_within_lan_lag(first_appearances: pd.DataFrame) -> pd.DataFrame:
    """
    For each län and term, compare when the prefecture adopted vs municipalities.

    This tests whether municipalities follow their own regional prefecture
    or if adoption patterns are independent.

    Returns
    -------
    pd.DataFrame
        Columns: lan, lan_name, term, prefecture_year, municipality_median_year,
        municipality_mean_year, n_municipalities, lag.
        Positive lag = municipalities adopt after their prefecture.
    """
    records = []

    # Filter to entities with län info and collapse pre-2015
    fa = first_appearances.copy()
    fa = fa[fa['lan'].notna()]
    fa.loc[fa['first_year'] < 2015, 'first_year'] = 2014
    fa['actor'] = fa['actor'].replace('länsstyrelse', 'lansstyrelse')

    for lan_id in sorted(fa['lan'].unique()):
        lan_data = fa[fa['lan'] == lan_id]
        lan_name = LAN_CODES.get(int(lan_id), f'Län {int(lan_id)}')

        # Get prefecture data for this län
        prefecture_data = lan_data[lan_data['actor'] == 'lansstyrelse']
        # Get municipality data for this län
        municipality_data = lan_data[lan_data['actor'] == 'kommun']

        if len(prefecture_data) == 0 or len(municipality_data) == 0:
            continue

        # For each term the prefecture has adopted
        for term in prefecture_data['term'].unique():
            pref_term = prefecture_data[prefecture_data['term'] == term]
            muni_term = municipality_data[municipality_data['term'] == term]

            if len(pref_term) == 0:
                continue

            pref_year = pref_term['first_year'].iloc[0]

            if len(muni_term) > 0:
                muni_median = muni_term['first_year'].median()
                muni_mean = muni_term['first_year'].mean()
                n_muni = len(muni_term)
                lag = muni_median - pref_year
            else:
                muni_median = np.nan
                muni_mean = np.nan
                n_muni = 0
                lag = np.nan

            records.append({
                'lan': int(lan_id),
                'lan_name': lan_name,
                'term': term,
                'prefecture_year': pref_year,
                'municipality_median_year': muni_median,
                'municipality_mean_year': muni_mean,
                'n_municipalities': n_muni,
                'lag': lag,
            })

    return pd.DataFrame(records)


# =============================================================================
# VISUALISATIONS
# =============================================================================

def _get_category_for_term(term: str) -> str:
    """Look up which category a term belongs to."""
    for category, terms in RISK_DICTIONARY.items():
        if term in terms:
            return category
    return 'unknown'


def plot_adoption_curves(
    adoption_curves: pd.DataFrame,
    output_dir: Path,
    min_entities: int = 10,
) -> None:
    """
    Multi-panel adoption curves, one panel per risk category.
    Separate lines per actor type.
    """
    # Get category for each term
    all_terms = adoption_curves['term'].unique()
    term_to_cat = {t: _get_category_for_term(t) for t in all_terms}

    # Filter to terms adopted by enough entities
    all_actor_curves = adoption_curves[adoption_curves['actor'] == 'all']
    max_fractions = all_actor_curves.groupby('term')['cumulative_fraction'].max()
    relevant_terms = max_fractions[max_fractions > 0].index

    # Only actor-specific curves (not 'all'), filter to 2015 onwards
    actor_curves = adoption_curves[adoption_curves['actor'] != 'all']
    actor_curves = actor_curves[actor_curves['term'].isin(relevant_terms)]
    actor_curves = actor_curves[actor_curves['year'] >= 2015]

    categories = sorted(set(term_to_cat.values()) - {'unknown'})
    n_cats = len(categories)
    if n_cats == 0:
        return

    n_cols = 2
    n_rows = (n_cats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    # Map both spellings to same style/color
    actor_styles = {
        'kommun': ('-', 'o'),
        'lansstyrelse': ('--', 's'),
        'länsstyrelse': ('--', 's'),
        'MCF': (':', '^'),
    }
    actor_colors = {
        'kommun': '#e41a1c',
        'lansstyrelse': '#377eb8',
        'länsstyrelse': '#377eb8',
        'MCF': '#4daf4a',
    }

    # Normalize actor names in data
    actor_curves = actor_curves.copy()
    actor_curves['actor'] = actor_curves['actor'].replace('länsstyrelse', 'lansstyrelse')

    for idx, category in enumerate(categories):
        ax = axes[idx // n_cols][idx % n_cols]
        cat_terms = [t for t, c in term_to_cat.items() if c == category and t in relevant_terms]

        # Pick top 5 most widely adopted terms in this category
        term_reach = {}
        for t in cat_terms:
            t_data = all_actor_curves[all_actor_curves['term'] == t]
            term_reach[t] = t_data['cumulative_fraction'].max()
        top_terms = sorted(term_reach, key=term_reach.get, reverse=True)[:5]

        for term in top_terms:
            for actor in ['kommun', 'lansstyrelse', 'MCF']:
                data = actor_curves[
                    (actor_curves['term'] == term) & (actor_curves['actor'] == actor)
                ]
                if len(data) == 0:
                    continue

                ls, marker = actor_styles[actor]
                ax.plot(
                    data['year'], data['cumulative_fraction'],
                    linestyle=ls, marker=marker, markersize=3,
                    color=actor_colors[actor], alpha=0.7,
                    label=f"{term[:20]} ({translate_actor(actor)})",
                )

        # Event annotations
        for year, label in EXTERNAL_EVENTS.items():
            ax.axvline(x=year, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

        ax.set_title(category, fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Year', fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel('Cumulative fraction', fontsize=10)

    # Hide unused subplots
    for idx in range(n_cats, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    # Add a shared legend for actor types (line styles)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#e41a1c', linestyle='-', marker='o', markersize=4, label='Municipality'),
        Line2D([0], [0], color='#377eb8', linestyle='--', marker='s', markersize=4, label='Prefecture'),
        Line2D([0], [0], color='#4daf4a', linestyle=':', marker='^', markersize=4, label='MCF'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
               bbox_to_anchor=(0.98, 1.06), title='Actor type')

    plt.suptitle(
        'Risk term adoption curves by actor type',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'adoption_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'adoption_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: adoption_curves.png/pdf")


def plot_gini_chart(gini_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Two-panel strip plot: Gini coefficients and IQR by risk category.
    """
    if len(gini_df) == 0:
        return

    # Filter to terms with enough entities
    df = gini_df[gini_df['n_entities'] >= 5].copy()

    if len(df) == 0:
        return

    # Add category
    df['category'] = df['term'].apply(_get_category_for_term)
    df = df[df['category'] != 'unknown']

    # Order categories by median Gini
    cat_order = df.groupby('category')['gini'].median().sort_values().index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Gini
    ax1 = axes[0]
    sns.stripplot(
        data=df, x='category', y='gini', order=cat_order,
        hue='category', palette='Set1', alpha=0.7, size=6,
        jitter=0.2, ax=ax1, legend=False,
    )
    medians_gini = df.groupby('category')['gini'].median()
    for i, cat in enumerate(cat_order):
        ax1.hlines(medians_gini[cat], i - 0.3, i + 0.3, colors='black', linewidth=2)
    ax1.set_xlabel('Risk category', fontsize=12)
    ax1.set_ylabel('Gini coefficient', fontsize=12)
    ax1.set_title('Gini coefficient\n(low = synchronous, high = gradual)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for label in ax1.get_xticklabels():
        label.set_ha('right')

    # Right panel: IQR
    ax2 = axes[1]
    sns.stripplot(
        data=df, x='category', y='iqr_years', order=cat_order,
        hue='category', palette='Set1', alpha=0.7, size=6,
        jitter=0.2, ax=ax2, legend=False,
    )
    medians_iqr = df.groupby('category')['iqr_years'].median()
    for i, cat in enumerate(cat_order):
        ax2.hlines(medians_iqr[cat], i - 0.3, i + 0.3, colors='black', linewidth=2)
    ax2.set_xlabel('Risk category', fontsize=12)
    ax2.set_ylabel('IQR (years)', fontsize=12)
    ax2.set_title('Interquartile range\n(years spanned by middle 50% of adopters)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for label in ax2.get_xticklabels():
        label.set_ha('right')

    plt.suptitle('Adoption synchronicity by risk category', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'gini_coefficients.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'gini_coefficients.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: gini_coefficients.png/pdf")


def plot_lead_lag(lead_lag_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Scatter: x = MCF/Prefecture first-appearance year,
    y = municipality first-appearance year.
    """
    for ref_actor, col_x, col_y in [
        ('lansstyrelse', 'median_year_lansstyrelse', 'median_year_kommun'),
        ('MCF', 'median_year_MCF', 'median_year_kommun'),
    ]:
        df = lead_lag_df.dropna(subset=[col_x, col_y])
        if len(df) < 3:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))

        # Color by category
        df = df.copy()
        df['category'] = df['term'].apply(_get_category_for_term)
        cat_colors = dict(zip(
            sorted(df['category'].unique()),
            sns.color_palette("Set1", df['category'].nunique())
        ))
        colors = [cat_colors.get(c, 'gray') for c in df['category']]

        ax.scatter(df[col_x], df[col_y], c=colors, alpha=0.7, s=40)

        # Diagonal line (= simultaneous adoption)
        lim_min = min(df[col_x].min(), df[col_y].min()) - 1
        lim_max = max(df[col_x].max(), df[col_y].max()) + 1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3)

        # Label some points
        for _, row in df.iterrows():
            ax.annotate(
                row['term'][:15], (row[col_x], row[col_y]),
                fontsize=6, alpha=0.7,
                xytext=(3, 3), textcoords='offset points',
            )

        ax.set_xlabel(f'{translate_actor(ref_actor)} median first year', fontsize=12)
        ax.set_ylabel('Municipality median first year', fontsize=12)
        ax.set_title(
            f'Lead-lag: {translate_actor(ref_actor)} vs Municipality\n'
            f'Above diagonal = {translate_actor(ref_actor)} leads',
            fontsize=13, fontweight='bold'
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cat_colors[c], label=c) for c in sorted(cat_colors.keys())
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

        plt.tight_layout()
        fname = f'lead_lag_{ref_actor}'
        plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}.png/pdf")


def plot_within_lan(within_lan_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Visualize within-prefecture adoption patterns:
    Histogram of lag between prefecture and municipality adoption.
    """
    df = within_lan_df.dropna(subset=['lag'])
    if len(df) < 10:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(df['lag'], bins=30, color='#377eb8', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='No lag')
    ax.axvline(x=df['lag'].mean(), color='red', linestyle='-', alpha=0.7,
               label=f'Mean: {df["lag"].mean():+.1f}y')
    ax.axvline(x=df['lag'].median(), color='orange', linestyle='-', alpha=0.7,
               label=f'Median: {df["lag"].median():+.1f}y')

    ax.set_xlabel('Lag (years, positive = municipalities adopt later)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(
        'Within-prefecture diffusion:\nDo municipalities follow their own prefecture?',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'within_prefecture_lag.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'within_prefecture_lag.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: within_prefecture_lag.png/pdf")


# =============================================================================
# REPORT
# =============================================================================

def generate_report(
    first_appearances: pd.DataFrame,
    spikes_df: pd.DataFrame,
    gini_df: pd.DataFrame,
    lead_lag_df: pd.DataFrame,
    within_lan_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate comprehensive text report."""
    report = []
    report.append("=" * 70)
    report.append("RISK DIFFUSION ANALYSIS — REPORT")
    report.append("=" * 70)

    # Summary
    n_entities = first_appearances['entity'].nunique()
    n_terms = first_appearances['term'].nunique()
    report.append(f"\nEntities: {n_entities}")
    report.append(f"Terms with at least one adoption: {n_terms}")

    # Left-censoring
    n_censored = first_appearances['is_left_censored'].sum()
    pct_censored = n_censored / len(first_appearances) * 100
    report.append(f"\nLeft-censored first appearances: {n_censored} ({pct_censored:.1f}%)")
    report.append("  (= term appears in entity's earliest available document)")

    # Spikes
    actual_spikes = spikes_df[spikes_df['is_spike']]
    report.append(f"\nSynchronous adoption spikes detected: {len(actual_spikes)}")
    if len(actual_spikes) > 0:
        for _, spike in actual_spikes.sort_values(
            'adoption_fraction', ascending=False
        ).head(20).iterrows():
            report.append(
                f"  {spike['term']:30s} in {int(spike['year'])}: "
                f"{spike['new_adopters']} adopters ({spike['adoption_fraction']:.1%})"
            )

    # Most synchronous terms (lowest Gini)
    report.append("\nMost synchronous terms (lowest Gini, ≥5 entities):")
    sync = gini_df[gini_df['n_entities'] >= 5].head(15)
    for _, row in sync.iterrows():
        report.append(
            f"  {row['term']:30s}: Gini={row['gini']:.3f}, "
            f"std={row['std_year']:.1f}y, IQR={row['iqr_years']:.1f}y "
            f"(n={int(row['n_entities'])}, mean={row['mean_year']:.0f})"
        )

    # Most gradual terms (highest Gini)
    report.append("\nMost gradual terms (highest Gini, ≥5 entities):")
    gradual = gini_df[gini_df['n_entities'] >= 5].tail(15).iloc[::-1]
    for _, row in gradual.iterrows():
        report.append(
            f"  {row['term']:30s}: Gini={row['gini']:.3f}, "
            f"std={row['std_year']:.1f}y, IQR={row['iqr_years']:.1f}y "
            f"(n={int(row['n_entities'])}, mean={row['mean_year']:.0f})"
        )

    # Lead-lag summary
    lag_col = 'lag_kommun_vs_lansstyrelse'
    if lag_col in lead_lag_df.columns:
        ll = lead_lag_df.dropna(subset=[lag_col])
        if len(ll) > 0:
            mean_lag = ll[lag_col].mean()
            report.append(f"\nLead-lag: Municipality vs Prefecture")
            report.append(f"  Mean lag: {mean_lag:+.1f} years")
            report.append(f"  (positive = municipalities adopt later)")
            top_down = (ll[lag_col] > 0).sum()
            bottom_up = (ll[lag_col] < 0).sum()
            simultaneous = (ll[lag_col] == 0).sum()
            report.append(
                f"  Top-down (prefecture first): {top_down}, "
                f"Bottom-up (municipality first): {bottom_up}, "
                f"Simultaneous: {simultaneous}"
            )

    # Within-prefecture summary
    if len(within_lan_df) > 0:
        wl = within_lan_df.dropna(subset=['lag'])
        if len(wl) > 0:
            report.append(f"\nWithin-prefecture analysis (municipality vs own prefecture):")
            report.append(f"  Comparing when municipalities adopt a term vs when their")
            report.append(f"  own regional prefecture adopted the same term.")
            report.append(f"  N = {len(wl)} (term, prefecture) pairs")
            report.append("")

            # Descriptive statistics
            report.append("  Descriptive statistics for lag (years):")
            report.append(f"    Mean:   {wl['lag'].mean():+.2f}")
            report.append(f"    Median: {wl['lag'].median():+.2f}")
            report.append(f"    Std:    {wl['lag'].std():.2f}")
            report.append(f"    Min:    {wl['lag'].min():+.1f}")
            report.append(f"    Max:    {wl['lag'].max():+.1f}")
            q25, q75 = wl['lag'].quantile([0.25, 0.75])
            report.append(f"    25th percentile: {q25:+.1f}")
            report.append(f"    75th percentile: {q75:+.1f}")
            report.append(f"    IQR:    {q75 - q25:.1f}")
            report.append("")

            # Direction counts
            top_down = (wl['lag'] > 0).sum()
            bottom_up = (wl['lag'] < 0).sum()
            simultaneous = (wl['lag'] == 0).sum()
            report.append("  Direction of diffusion:")
            report.append(
                f"    Prefecture first (top-down): {top_down} ({100*top_down/len(wl):.1f}%)"
            )
            report.append(
                f"    Municipality first (bottom-up): {bottom_up} ({100*bottom_up/len(wl):.1f}%)"
            )
            report.append(
                f"    Simultaneous: {simultaneous} ({100*simultaneous/len(wl):.1f}%)"
            )

            # Per-prefecture summary
            report.append("\n  Per-prefecture mean lag (sorted by lag):")
            lan_stats = wl.groupby('lan_name')['lag'].agg(['mean', 'median', 'std', 'count'])
            lan_stats = lan_stats.sort_values('mean')
            for lan_name, row in lan_stats.iterrows():
                report.append(
                    f"    {lan_name:20s}: mean={row['mean']:+.1f}y, "
                    f"median={row['median']:+.1f}y, std={row['std']:.1f}y (n={int(row['count'])})"
                )

    # Save
    report_text = '\n'.join(report)
    report_path = output_dir / 'diffusion_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Saved: diffusion_report.txt")
    print(f"\n{report_text}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze risk term diffusion across entities and actor types'
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to term_document_matrix.csv'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./results/01_bow_analysis/diffusion'),
        help='Output directory'
    )

    parser.add_argument(
        '--spike-threshold',
        type=float,
        default=0.15,
        help='Fraction threshold for adoption spike detection (default: 0.15)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages'
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RISK DIFFUSION ANALYSIS")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {args.input}")
    df, term_cols = load_and_prepare(args.input)

    # First appearances
    print("\nComputing first appearances...")
    first_appearances = compute_first_appearances(df, term_cols)
    print(f"  {len(first_appearances)} (entity, term) first appearances")
    n_censored = first_appearances['is_left_censored'].sum()
    print(f"  Left-censored: {n_censored} ({n_censored / len(first_appearances) * 100:.1f}%)")

    # Adoption curves
    print("\nComputing adoption curves...")
    adoption_curves = compute_adoption_curves(first_appearances, df, term_cols)
    print(f"  {len(adoption_curves)} curve data points")

    # Spikes
    print(f"\nDetecting adoption spikes (threshold={args.spike_threshold:.0%})...")
    spikes_df = detect_adoption_spikes(
        first_appearances, df, threshold=args.spike_threshold
    )
    n_spikes = spikes_df['is_spike'].sum()
    print(f"  {n_spikes} spikes detected")

    # Gini
    print("\nComputing Gini coefficients...")
    gini_df = compute_gini_coefficients(first_appearances)
    print(f"  {len(gini_df)} terms with Gini scores")

    # Lead-lag
    print("\nComputing lead-lag analysis...")
    lead_lag_df = compute_lead_lag(first_appearances)
    print(f"  {len(lead_lag_df)} terms analyzed")

    # Within-prefecture analysis
    print("\nComputing within-prefecture analysis...")
    within_lan_df = compute_within_lan_lag(first_appearances)
    n_matched = first_appearances['lan'].notna().sum()
    print(f"  {n_matched} first appearances matched to prefecture")
    print(f"  {len(within_lan_df)} (prefecture, term) pairs analyzed")

    # Save data
    print("\nSaving data...")
    first_appearances.to_csv(
        args.output / 'first_appearances.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: first_appearances.csv")

    spikes_df.to_csv(
        args.output / 'adoption_spikes.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: adoption_spikes.csv")

    gini_df.to_csv(
        args.output / 'gini_coefficients.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: gini_coefficients.csv")

    lead_lag_df.to_csv(
        args.output / 'lead_lag.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: lead_lag.csv")

    within_lan_df.to_csv(
        args.output / 'within_prefecture_lag.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: within_prefecture_lag.csv")

    # Visualisations
    print("\nGenerating visualisations...")
    plot_adoption_curves(adoption_curves, args.output)
    plot_gini_chart(gini_df, args.output)
    plot_lead_lag(lead_lag_df, args.output)
    plot_within_lan(within_lan_df, args.output)

    # Report
    print("\nGenerating report...")
    generate_report(
        first_appearances, spikes_df, gini_df, lead_lag_df, within_lan_df, args.output
    )

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {args.output}")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
