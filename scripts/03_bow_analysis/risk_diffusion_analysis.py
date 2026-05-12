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

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries.risk_translations import translate_term

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


# Import centralized dictionary from scripts/dictionaries/
sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries import RISK_TERMS as RISK_DICTIONARY_INDIVIDUAL
from dictionaries.risk_translations import (
    translate_term,
    translate_actor as _translate_actor,
)


def translate_actor(actor: str) -> str:
    """Translate actor names from Swedish to English."""
    return ACTOR_TRANSLATIONS.get(actor, _translate_actor(actor))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare(input_path: Path, min_year: int = 2015) -> tuple:
    """Load term-document matrix and filter to min_year onwards."""
    df = pd.read_csv(input_path)
    n_total = len(df)
    df = df[df['year'] >= min_year]
    term_cols = [c for c in df.columns if c not in METADATA_COLS]
    print(f"  Loaded {n_total} documents, filtered to {len(df)} (>= {min_year}), {len(term_cols)} terms")
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

    for (entity, actor), group in df.groupby(['entity', 'actor']):
        group = group.sort_values('year')
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
# ENTITY COVERAGE (what % of dictionary terms does each entity mention?)
# =============================================================================

def compute_entity_coverage(
    df: pd.DataFrame,
    term_cols: list,
) -> pd.DataFrame:
    """
    For each entity-year, compute what fraction of dictionary terms are mentioned.

    This is the entity-centric view: "how comprehensive is each RSA?"
    Unlike term-centric adoption curves, this isn't dragged down by rare terms.

    Parameters
    ----------
    df : pd.DataFrame
        Term-document matrix with metadata columns.
    term_cols : list
        List of term column names.

    Returns
    -------
    pd.DataFrame
        Columns: entity, actor, year, terms_mentioned, total_terms, coverage_rate.
    """
    records = []
    total_terms = len(term_cols)

    for _, row in df.iterrows():
        terms_mentioned = (row[term_cols] > 0).sum()
        records.append({
            'entity': row['entity'],
            'actor': row['actor'],
            'year': row['year'],
            'terms_mentioned': terms_mentioned,
            'total_terms': total_terms,
            'coverage_rate': terms_mentioned / total_terms if total_terms > 0 else 0,
        })

    return pd.DataFrame(records)


# =============================================================================
# ADOPTION CURVES (term-centric, kept for individual term analysis)
# =============================================================================

def compute_adoption_curves(
    first_appearances: pd.DataFrame,
    df: pd.DataFrame,
    term_cols: list,
) -> pd.DataFrame:
    """
    For each term (and optionally per actor type), compute cumulative
    adoption fraction over time.

    Uses year-varying denominator: only counts entities that have at least
    one document in or before each year. This avoids artificially suppressing
    early adoption rates due to entities that only appear in later waves.

    Returns
    -------
    pd.DataFrame
        Columns: term, actor, year, cumulative_count, observable_entities,
        cumulative_fraction.
    """
    all_years = sorted(df['year'].unique())
    records = []

    # Pre-compute: for each entity, what's their earliest document year?
    entity_first_year = df.groupby('entity')['year'].min()

    for actor_filter in [None] + list(df['actor'].unique()):
        if actor_filter is None:
            subset = first_appearances
            actor_entities = entity_first_year
            actor_label = 'all'
        else:
            subset = first_appearances[first_appearances['actor'] == actor_filter]
            actor_entity_list = df[df['actor'] == actor_filter]['entity'].unique()
            actor_entities = entity_first_year[entity_first_year.index.isin(actor_entity_list)]
            actor_label = actor_filter

        if len(actor_entities) == 0:
            continue

        for term in term_cols:
            term_data = subset[subset['term'] == term]
            if len(term_data) == 0:
                continue

            for year in all_years:
                # Count entities observable by this year (have docs in or before this year)
                observable_entities = (actor_entities <= year).sum()
                if observable_entities == 0:
                    continue

                cum_count = (term_data['first_year'] <= year).sum()
                records.append({
                    'term': term,
                    'actor': actor_label,
                    'year': year,
                    'cumulative_count': cum_count,
                    'observable_entities': observable_entities,
                    'cumulative_fraction': cum_count / observable_entities,
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

    A spike occurs when ≥ threshold fraction of observable entities first
    mention a term in the same year. Uses year-varying denominator to only
    count entities with documents in or before each year.

    Parameters
    ----------
    first_appearances : pd.DataFrame
        First appearance data.
    df : pd.DataFrame
        Full data (for entity counts per year).
    threshold : float
        Minimum fraction to flag as a spike.

    Returns
    -------
    pd.DataFrame
        Detected spikes with columns: term, year, new_adopters,
        observable_entities, adoption_fraction, is_spike.
    """
    # Pre-compute: for each entity, what's their earliest document year?
    entity_first_year = df.groupby('entity')['year'].min()
    all_years = sorted(df['year'].unique())

    # Pre-compute observable entities per year
    observable_by_year = {year: (entity_first_year <= year).sum() for year in all_years}

    records = []

    for term, group in first_appearances.groupby('term'):
        yearly_counts = group.groupby('first_year').size()

        for year, count in yearly_counts.items():
            observable = observable_by_year.get(year, 0)
            if observable == 0:
                continue
            fraction = count / observable
            records.append({
                'term': term,
                'year': year,
                'new_adopters': count,
                'observable_entities': observable,
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

def get_balanced_panel_entities(df: pd.DataFrame) -> set:
    """
    Identify entities with documents in all 3 waves (2015-2018, 2019-2022, 2023+).
    """
    def get_wave(year):
        if year <= 2018:
            return 1
        elif year <= 2022:
            return 2
        else:
            return 3

    df = df.copy()
    df['wave_num'] = df['year'].apply(get_wave)
    entity_waves = df.groupby('entity')['wave_num'].apply(set)
    balanced = entity_waves[entity_waves.apply(lambda x: x == {1, 2, 3})]
    return set(balanced.index)


def compute_lead_lag(
    first_appearances: pd.DataFrame,
    balanced_entities: set = None,
) -> pd.DataFrame:
    """
    For each term, compare median first-appearance year between actor types.

    Parameters
    ----------
    balanced_entities : set, optional
        If provided, restrict to entities in this set (balanced panel).

    Returns
    -------
    pd.DataFrame
        Columns: term, median_year_kommun, median_year_lansstyrelse,
        median_year_MCF, lag_kommun_vs_lansstyrelse, lag_kommun_vs_MCF.
        Positive lag = municipalities adopt later (top-down diffusion).
    """
    records = []

    # Normalize actor names
    first_appearances = first_appearances.copy()
    first_appearances['actor'] = first_appearances['actor'].replace('länsstyrelse', 'lansstyrelse')

    # Filter to balanced panel if specified
    if balanced_entities is not None:
        first_appearances = first_appearances[
            first_appearances['entity'].isin(balanced_entities)
        ]

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

def compute_within_lan_lag(
    first_appearances: pd.DataFrame,
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    For each län and term, compare when the prefecture adopted vs municipalities.

    Only includes municipality adoptions that occur on or after the prefecture's
    earliest document year, to avoid bias from prefectures with incomplete coverage.

    Parameters
    ----------
    first_appearances : pd.DataFrame
        First appearance data.
    df : pd.DataFrame, optional
        Original document data, used to determine each prefecture's earliest doc year.
        If not provided, uses min first_year from first_appearances (less accurate).

    Returns
    -------
    pd.DataFrame
        Columns: lan, lan_name, term, prefecture_year, prefecture_first_doc_year,
        prefecture_left_censored, municipality_median_year, municipality_mean_year,
        n_municipalities, lag.
        Positive lag = municipalities adopt after their prefecture.
        prefecture_left_censored = True if prefecture's first mention is in their
        earliest document (we can't tell if they mentioned it earlier).
    """
    records = []

    # Filter to entities with län info
    fa = first_appearances.copy()
    fa = fa[fa['lan'].notna()]
    fa['actor'] = fa['actor'].replace('länsstyrelse', 'lansstyrelse')

    # Compute each prefecture's earliest document year
    if df is not None:
        df_pref = df[df['actor'].isin(['lansstyrelse', 'länsstyrelse'])].copy()
        pref_first_doc = df_pref.groupby('entity')['year'].min().to_dict()
    else:
        # Fallback: use min first_year from first_appearances
        pref_fa = fa[fa['actor'] == 'lansstyrelse']
        pref_first_doc = pref_fa.groupby('entity')['first_year'].min().to_dict()

    for lan_id in sorted(fa['lan'].unique()):
        lan_data = fa[fa['lan'] == lan_id]
        lan_name = LAN_CODES.get(int(lan_id), f'Län {int(lan_id)}')

        # Get prefecture data for this län
        prefecture_data = lan_data[lan_data['actor'] == 'lansstyrelse']
        # Get municipality data for this län
        municipality_data = lan_data[lan_data['actor'] == 'kommun']

        if len(prefecture_data) == 0 or len(municipality_data) == 0:
            continue

        # Get prefecture's earliest document year
        pref_entity = prefecture_data['entity'].iloc[0]
        pref_first_doc_year = pref_first_doc.get(pref_entity, 2015)

        # For each term the prefecture has adopted
        for term in prefecture_data['term'].unique():
            pref_term = prefecture_data[prefecture_data['term'] == term]
            muni_term = municipality_data[municipality_data['term'] == term]

            if len(pref_term) == 0:
                continue

            pref_year = pref_term['first_year'].iloc[0]
            pref_censored = pref_term['is_left_censored'].iloc[0]

            # Only include municipality adoptions >= prefecture's first doc year
            # This avoids counting municipalities as "earlier" when we simply
            # don't have prefecture data for those years
            muni_term_filtered = muni_term[muni_term['first_year'] >= pref_first_doc_year]

            if len(muni_term_filtered) > 0:
                muni_median = muni_term_filtered['first_year'].median()
                muni_mean = muni_term_filtered['first_year'].mean()
                n_muni = len(muni_term_filtered)
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
                'prefecture_first_doc_year': pref_first_doc_year,
                'prefecture_left_censored': pref_censored,
                'municipality_median_year': muni_median,
                'municipality_mean_year': muni_mean,
                'n_municipalities': n_muni,
                'lag': lag,
            })

    return pd.DataFrame(records)


def compute_municipal_national_lag(first_appearances: pd.DataFrame) -> dict:
    """
    Compare municipality adoptions to the earliest prefecture adoption nationally.

    Returns dict with:
    - n_pairs: number of municipality-term pairs analyzed
    - mean_lag, median_lag: lag to earliest prefecture (any, including left-censored)
    - pct_before: % of municipal adoptions before any prefecture
    - pct_same: % same year
    - pct_after: % after some prefecture
    """
    prefs = first_appearances[first_appearances['actor'] == 'lansstyrelse']
    first_pref = prefs.groupby('term')['first_year'].min().reset_index()
    first_pref.columns = ['term', 'earliest_pref_year']

    munis = first_appearances[first_appearances['actor'] == 'kommun'].copy()
    merged = munis.merge(first_pref, on='term', how='inner')
    merged['lag'] = merged['first_year'] - merged['earliest_pref_year']

    n = len(merged)
    return {
        'n_pairs': n,
        'mean_lag': merged['lag'].mean(),
        'median_lag': merged['lag'].median(),
        'pct_before': (merged['lag'] < 0).sum() / n * 100,
        'pct_same': (merged['lag'] == 0).sum() / n * 100,
        'pct_after': (merged['lag'] > 0).sum() / n * 100,
    }


def compute_prefecture_influence(first_appearances: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which prefectures municipalities follow when adopting risk terms.

    For each municipality-term pair, credits the first (earliest) prefecture to
    adopt the term. This reveals whether municipalities follow their own
    prefecture or copy from national leaders.

    Returns
    -------
    pd.DataFrame
        Columns: prefecture, times_followed, terms_led, influence_ratio, pct_own_region.
        influence_ratio = times_followed / terms_led (higher = more influential per term).
        pct_own_region = % of followers that are from the same region.
    """
    # Only non-left-censored prefectures (valid first adoptions)
    prefs_valid = first_appearances[
        (first_appearances['actor'] == 'lansstyrelse') &
        (first_appearances['is_left_censored'] == False)
    ][['term', 'entity', 'lan', 'first_year']].copy()
    prefs_valid.columns = ['term', 'pref_entity', 'pref_lan', 'pref_year']

    munis = first_appearances[first_appearances['actor'] == 'kommun'].copy()

    results = []
    for _, muni_row in munis.iterrows():
        term = muni_row['term']
        muni_year = muni_row['first_year']
        muni_lan = muni_row['lan']

        # Find prefectures that adopted this term before/same year as municipality
        prefs_before = prefs_valid[
            (prefs_valid['term'] == term) &
            (prefs_valid['pref_year'] <= muni_year)
        ]

        if len(prefs_before) > 0:
            # Find first (earliest) prefecture to adopt this term
            earliest_idx = prefs_before['pref_year'].idxmin()
            earliest = prefs_before.loc[earliest_idx]
            results.append({
                'followed_prefecture': earliest['pref_entity'],
                'followed_lan': earliest['pref_lan'],
                'follower_lan': muni_lan,
                'is_own_region': muni_lan == earliest['pref_lan'],
            })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Aggregate by prefecture
    followed_counts = results_df['followed_prefecture'].value_counts().reset_index()
    followed_counts.columns = ['prefecture', 'times_followed']

    # Own-region percentage
    own_region = results_df.groupby('followed_prefecture')['is_own_region'].mean().reset_index()
    own_region.columns = ['prefecture', 'pct_own_region']

    # Terms led (opportunities)
    terms_led = prefs_valid.groupby('pref_entity')['term'].nunique().reset_index()
    terms_led.columns = ['prefecture', 'terms_led']

    # Merge
    summary = followed_counts.merge(terms_led, on='prefecture', how='left')
    summary = summary.merge(own_region, on='prefecture', how='left')
    summary['influence_ratio'] = (summary['times_followed'] / summary['terms_led']).round(1)
    summary['pct_own_region'] = (summary['pct_own_region'] * 100).round(1)

    # Add overall stats
    total_follows = len(results_df)
    own_prefecture_pct = results_df['is_own_region'].mean() * 100

    return summary.sort_values('influence_ratio', ascending=False), total_follows, own_prefecture_pct


# =============================================================================
# VISUALISATIONS
# =============================================================================

def _get_canonical_risk(term: str) -> str | None:
    """Look up canonical risk name for a term from the individual dictionary."""
    for canonical, variants in RISK_DICTIONARY_INDIVIDUAL.items():
        if term in variants or term == canonical:
            return canonical
    return None


def _get_actor_style() -> tuple[dict, dict]:
    """Return consistent actor styles and colors."""
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
    return actor_styles, actor_colors


def plot_aggregate_adoption_curves(
    entity_coverage: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Aggregate coverage curves: one line per actor type showing mean dictionary
    coverage rate over time.

    Entity-centric metric: "what fraction of dictionary terms does the average
    entity mention?" Entities without documents in a given year are excluded
    (not coded as zero).

    Municipalities: aggregated by wave (3 data points).
    Prefectures: aggregated by year.

    Note: MCF is excluded because n=1 entity makes the average meaningless.
    """
    from matplotlib.lines import Line2D

    # Filter to 2015+ and exclude MCF
    df = entity_coverage[entity_coverage['year'] >= 2015].copy()
    df = df[~df['actor'].isin(['MCF'])]
    df['actor'] = df['actor'].replace('länsstyrelse', 'lansstyrelse')

    if len(df) == 0:
        return

    # Assign waves
    def get_wave(year):
        if year <= 2018:
            return 1
        elif year <= 2022:
            return 2
        else:
            return 3

    def get_wave_midpoint(wave):
        return {1: 2015, 2: 2019, 3: 2023}[wave]

    df['wave'] = df['year'].apply(get_wave)

    actor_styles, actor_colors = _get_actor_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Municipalities: aggregate by wave (3 data points)
    muni = df[df['actor'] == 'kommun']
    if len(muni) > 0:
        muni_agg = muni.groupby('wave').agg(
            mean_coverage=('coverage_rate', 'mean'),
            std_coverage=('coverage_rate', 'std'),
            n_entities=('entity', 'nunique'),
        ).reset_index()
        muni_agg['x'] = muni_agg['wave'].apply(get_wave_midpoint)

        ls, marker = actor_styles['kommun']
        color = actor_colors['kommun']

        ax.plot(
            muni_agg['x'], muni_agg['mean_coverage'],
            linestyle=ls, marker=marker, markersize=8,
            color=color, linewidth=2, label=translate_actor('kommun'),
        )
        ax.fill_between(
            muni_agg['x'],
            muni_agg['mean_coverage'] - muni_agg['std_coverage'],
            muni_agg['mean_coverage'] + muni_agg['std_coverage'],
            color=color, alpha=0.15,
        )

    # Prefectures: aggregate by 2-year periods (2019→2018, 2023→2022)
    pref = df[df['actor'] == 'lansstyrelse'].copy()
    if len(pref) > 0:
        # Group sparse years with prior year
        pref['year_grouped'] = pref['year'].replace({2019: 2018, 2023: 2022})

        pref_agg = pref.groupby('year_grouped').agg(
            mean_coverage=('coverage_rate', 'mean'),
            std_coverage=('coverage_rate', 'std'),
            n_entities=('entity', 'nunique'),
        ).reset_index()

        ls, marker = actor_styles['lansstyrelse']
        color = actor_colors['lansstyrelse']

        ax.plot(
            pref_agg['year_grouped'], pref_agg['mean_coverage'],
            linestyle=ls, marker=marker, markersize=5,
            color=color, linewidth=2, label=translate_actor('lansstyrelse'),
        )
        ax.fill_between(
            pref_agg['year_grouped'],
            pref_agg['mean_coverage'] - pref_agg['std_coverage'],
            pref_agg['mean_coverage'] + pref_agg['std_coverage'],
            color=color, alpha=0.15,
        )

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Mean dictionary coverage rate', fontsize=12)
    ax.set_ylim(0, None)  # Start at 0, auto-scale upper bound
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title='Actor type', loc='lower right')
    ax.set_title('Risk dictionary coverage by actor type', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'adoption_curves_aggregate.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'adoption_curves_aggregate.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: adoption_curves_aggregate.png/pdf")


def plot_individual_adoption_curves(
    adoption_curves: pd.DataFrame,
    output_dir: Path,
    gini_df: pd.DataFrame = None,
    first_appearances: pd.DataFrame = None,
    selection: str = 'mixed',
) -> None:
    """
    Small multiples: one panel per selected individual risk terms,
    showing adoption curves by actor type.

    Parameters
    ----------
    selection : str
        'top_adopted' - top 9 by max adoption (original)
        'mixed' - top 3 synchronous + top 3 gradual + top 3 recent
    """
    from matplotlib.lines import Line2D

    # Filter to actor-specific curves and 2015+
    actor_curves = adoption_curves[adoption_curves['actor'] != 'all'].copy()
    actor_curves = actor_curves[actor_curves['year'] >= 2015]
    actor_curves['actor'] = actor_curves['actor'].replace('länsstyrelse', 'lansstyrelse')

    # Map terms to canonical risk names
    actor_curves['canonical'] = actor_curves['term'].apply(_get_canonical_risk)
    actor_curves = actor_curves[actor_curves['canonical'].notna()]

    if len(actor_curves) == 0:
        print("  Warning: No terms matched to individual risk dictionary")
        return

    # Aggregate by canonical risk (in case multiple variants exist)
    agg = actor_curves.groupby(['canonical', 'actor', 'year'])['cumulative_fraction'].mean().reset_index()

    # Select risks based on selection mode
    if selection == 'theoretical':
        # Theoretically motivated selection
        # Legitimacy/blame (orange)
        legitimacy = ['social oro', 'pågående dödligt våld']
        # Top-down diffusion (blue)
        top_down = ['desinformation', 'flyktingström', 'väpnat angrepp']
        # Bottom-up diffusion (green)
        bottom_up = ['ekonomisk kris', 'försvunnen person']
        # Post-2022 security event (red)
        security = ['strid på svenskt territorium', 'cyberattack', 'ransomware']

        top_risks = legitimacy + top_down + bottom_up + security
        subtitle = 'Legitimacy (orange) | Top-down (blue) | Bottom-up (green) | Post-2022 security (red)'
        risk_colors = {r: '#ff7f00' for r in legitimacy}  # orange
        risk_colors.update({r: '#377eb8' for r in top_down})  # blue
        risk_colors.update({r: '#4daf4a' for r in bottom_up})  # green
        risk_colors.update({r: '#e41a1c' for r in security})  # red

    elif selection == 'mixed' and gini_df is not None and first_appearances is not None:
        # Map gini terms to canonical
        gini = gini_df.copy()
        gini['canonical'] = gini['term'].apply(_get_canonical_risk)
        gini = gini[gini['canonical'].notna()]
        gini = gini.groupby('canonical').agg({'gini': 'mean', 'n_entities': 'sum'}).reset_index()
        gini = gini[gini['n_entities'] >= 10]  # Minimum entities for meaningful Gini

        # Top 3 synchronous (lowest Gini)
        synchronous = gini.nsmallest(3, 'gini')['canonical'].tolist()

        # Top 3 gradual (highest Gini)
        gradual = gini.nlargest(3, 'gini')['canonical'].tolist()

        # Top 3 largest actor gap (municipality vs prefecture timing difference)
        fa = first_appearances.copy()
        fa['canonical'] = fa['term'].apply(_get_canonical_risk)
        fa = fa[fa['canonical'].notna()]
        # Compute median first_year per actor per canonical risk
        actor_timing = fa.groupby(['canonical', 'actor'])['first_year'].median().unstack(fill_value=np.nan)
        # Normalize actor names
        if 'länsstyrelse' in actor_timing.columns and 'lansstyrelse' not in actor_timing.columns:
            actor_timing = actor_timing.rename(columns={'länsstyrelse': 'lansstyrelse'})
        # Compute gap: municipality - prefecture (positive = municipalities lag)
        if 'kommun' in actor_timing.columns and 'lansstyrelse' in actor_timing.columns:
            actor_timing['gap'] = abs(actor_timing['kommun'] - actor_timing['lansstyrelse'])
            actor_timing = actor_timing.dropna(subset=['gap'])
            # Exclude those already selected
            already_selected = set(synchronous + gradual)
            gap_candidates = actor_timing[~actor_timing.index.isin(already_selected)]
            actor_gap = gap_candidates.nlargest(3, 'gap').index.tolist()
        else:
            actor_gap = []

        top_risks = synchronous + gradual + actor_gap
        subtitle = 'Top 3 synchronous (blue) + Top 3 gradual (red) + Top 3 actor gap (green)'
        # Color-code titles
        risk_colors = {r: '#377eb8' for r in synchronous}
        risk_colors.update({r: '#e41a1c' for r in gradual})
        risk_colors.update({r: '#4daf4a' for r in actor_gap})
    else:
        # Fallback: top N by max adoption
        all_curves = adoption_curves[adoption_curves['actor'] == 'all'].copy()
        all_curves['canonical'] = all_curves['term'].apply(_get_canonical_risk)
        all_curves = all_curves[all_curves['canonical'].notna()]
        max_adoption = all_curves.groupby('canonical')['cumulative_fraction'].max().sort_values(ascending=False)
        top_risks = max_adoption.head(9).index.tolist()
        subtitle = 'Top 9 by maximum adoption'
        risk_colors = {}

    if len(top_risks) == 0:
        return

    actor_styles, actor_colors = _get_actor_style()

    n_cols = 3
    n_rows = (len(top_risks) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    for idx, risk in enumerate(top_risks):
        ax = axes[idx // n_cols][idx % n_cols]

        for actor in ['kommun', 'lansstyrelse', 'MCF']:
            data = agg[(agg['canonical'] == risk) & (agg['actor'] == actor)].sort_values('year')
            if len(data) == 0:
                continue

            ls, marker = actor_styles[actor]
            ax.plot(
                data['year'], data['cumulative_fraction'],
                linestyle=ls, marker=marker, markersize=4,
                color=actor_colors[actor], linewidth=1.5,
            )

        # Event annotations
        for year, label in EXTERNAL_EVENTS.items():
            ax.axvline(x=year, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

        # Color-code title based on selection category
        title_color = risk_colors.get(risk, 'black')
        ax.set_title(translate_term(risk), fontsize=11, fontweight='bold', color=title_color)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Year', fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel('Cumulative fraction', fontsize=10)

    # Hide unused subplots
    for idx in range(len(top_risks), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color='#e41a1c', linestyle='-', marker='o', markersize=4, label='Municipality'),
        Line2D([0], [0], color='#377eb8', linestyle='--', marker='s', markersize=4, label='Prefecture'),
        Line2D([0], [0], color='#4daf4a', linestyle=':', marker='^', markersize=4, label='MSB'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
               bbox_to_anchor=(0.98, 1.02), title='Actor type')

    plt.suptitle(
        f'Individual risk adoption curves\n({subtitle})',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'adoption_curves_individual.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'adoption_curves_individual.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: adoption_curves_individual.png/pdf")


def plot_gini_chart(gini_df: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    """
    Two-panel chart showing Gini coefficients and IQR for top individual risks.
    Left: bar chart of top N risks by Gini (most synchronous to most gradual)
    Right: scatter of Gini vs IQR with risk labels
    """
    if len(gini_df) == 0:
        return

    # Filter to terms with enough entities
    df = gini_df[gini_df['n_entities'] >= 5].copy()

    if len(df) == 0:
        return

    # Map to canonical risk names
    df['canonical'] = df['term'].apply(_get_canonical_risk)
    df = df[df['canonical'].notna()]

    if len(df) == 0:
        print("  Warning: No terms matched to individual risk dictionary for Gini chart")
        return

    # Aggregate by canonical risk (take mean if multiple variants)
    agg = df.groupby('canonical').agg({
        'gini': 'mean',
        'iqr_years': 'mean',
        'n_entities': 'sum',
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left panel: Top N risks by Gini (bar chart)
    ax1 = axes[0]
    top_gini = agg.nlargest(top_n, 'n_entities').sort_values('gini')
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_gini)))
    ax1.barh(top_gini['canonical'], top_gini['gini'], color=colors)
    ax1.set_xlabel('Gini coefficient', fontsize=12)
    ax1.set_title('Adoption synchronicity\n(low = synchronous, high = gradual)', fontsize=12, fontweight='bold')
    ax1.axvline(x=agg['gini'].median(), color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax1.text(agg['gini'].median() + 0.01, 0.5, 'median', fontsize=9, va='bottom', color='gray')

    # Right panel: Gini vs IQR scatter
    ax2 = axes[1]
    scatter = ax2.scatter(
        agg['gini'], agg['iqr_years'],
        s=agg['n_entities'] * 2, alpha=0.6, c=agg['gini'],
        cmap='RdYlGn_r', edgecolors='black', linewidth=0.5
    )
    # Label top risks by n_entities
    for _, row in agg.nlargest(10, 'n_entities').iterrows():
        ax2.annotate(
            row['canonical'], (row['gini'], row['iqr_years']),
            fontsize=8, alpha=0.8,
            xytext=(5, 5), textcoords='offset points',
        )
    ax2.set_xlabel('Gini coefficient', fontsize=12)
    ax2.set_ylabel('IQR (years)', fontsize=12)
    ax2.set_title('Gini vs. adoption timespan\n(size = entity count)', fontsize=12, fontweight='bold')

    plt.suptitle('Adoption synchronicity by individual risk', fontsize=14, fontweight='bold', y=1.02)
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

        fig, ax = plt.subplots(figsize=(12, 12))

        # Map to canonical risk names and aggregate variants
        df = df.copy()
        df['canonical'] = df['term'].apply(_get_canonical_risk)
        df = df[df['canonical'].notna()]  # Filter to known risks

        # Aggregate by canonical risk (mean of medians across variants)
        # This prevents duplicate labels for the same risk concept
        agg_cols = [col_x, col_y]
        n_cols = [c for c in df.columns if c.startswith('n_')]
        df = df.groupby('canonical', as_index=False).agg({
            col_x: 'mean',
            col_y: 'mean',
            **{c: 'sum' for c in n_cols},
            'term': 'first',  # Keep one term for reference
        })
        df['lag'] = df[col_y] - df[col_x]

        # Color by lag magnitude (blue = leads, red = lags)
        scatter = ax.scatter(df[col_x], df[col_y], c=df['lag'], alpha=0.7, s=40,
                             cmap='RdYlBu_r')

        # Diagonal line (= simultaneous adoption)
        lim_min = min(df[col_x].min(), df[col_y].min()) - 1
        lim_max = max(df[col_x].max(), df[col_y].max()) + 1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3)

        # Label points: max 5 per (x,y) coordinate, prioritized by corpus adoption
        df = df.copy()

        # Group by coordinate and keep top 5 most common terms per position
        max_labels_per_pos = 5
        df['coord'] = list(zip(df[col_x].round(0), df[col_y].round(0)))

        # Get term adoption counts (total entities that adopted each term)
        adoption_cols = [c for c in df.columns if c.startswith('n_') and c != 'n_entities']
        if adoption_cols:
            df['total_adoption'] = df[adoption_cols].sum(axis=1)
            adoption_col = 'total_adoption'
        else:
            # Fallback: use inverse of absolute lag (prefer terms near diagonal)
            df['adoption_rank'] = -abs(df[col_y] - df[col_x])
            adoption_col = 'adoption_rank'

        # For each coordinate, keep top N by adoption
        labels_to_show = []
        for coord, group in df.groupby('coord'):
            top_n = group.nlargest(max_labels_per_pos, adoption_col)
            labels_to_show.extend(top_n.index.tolist())

        df_to_label = df.loc[labels_to_show].copy()

        # Add small jitter to label positions for readability
        np.random.seed(42)
        jitter_x = np.random.uniform(-0.3, 0.3, len(df_to_label))
        jitter_y = np.random.uniform(-0.3, 0.3, len(df_to_label))

        try:
            from adjustText import adjust_text
            texts = []
            for i, (_, row) in enumerate(df_to_label.iterrows()):
                # Use canonical name if available, otherwise original term
                raw_label = row['canonical'] if pd.notna(row.get('canonical')) else row['term']
                label = translate_term(raw_label)
                texts.append(ax.text(
                    row[col_x] + jitter_x[i],
                    row[col_y] + jitter_y[i],
                    label[:20],
                    fontsize=7, alpha=0.85,
                ))
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.4),
                expand_points=(1.5, 1.5),
                force_text=(0.4, 0.4),
                lim=300,
            )
        except ImportError:
            # Fallback: simple jittered labels
            for i, (_, row) in enumerate(df_to_label.iterrows()):
                raw_label = row['canonical'] if pd.notna(row.get('canonical')) else row['term']
                label = translate_term(raw_label)
                ax.annotate(
                    label[:20],
                    (row[col_x], row[col_y]),
                    fontsize=7, alpha=0.85,
                    xytext=(5 + jitter_x[i]*10, 5 + jitter_y[i]*10),
                    textcoords='offset points',
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

        # Add colorbar for lag magnitude
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Lag (years)', fontsize=10)

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

    Only uses valid comparisons where prefecture is NOT left-censored.
    """
    # Filter to valid comparisons only (prefecture not left-censored)
    df = within_lan_df[
        (within_lan_df['prefecture_left_censored'] == False) &
        (within_lan_df['lag'].notna())
    ].copy()

    if len(df) < 10:
        return

    total_pairs = len(within_lan_df)
    valid_pairs = len(df)

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
        f'Within-prefecture diffusion (valid pairs only)\n'
        f'n={valid_pairs} of {total_pairs} pairs (prefecture not left-censored)',
        fontsize=12, fontweight='bold'
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
    national_lag_stats: dict = None,
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

    # Within-prefecture summary (valid pairs only - prefecture not left-censored)
    if len(within_lan_df) > 0:
        total_pairs = len(within_lan_df)
        wl = within_lan_df[
            (within_lan_df['prefecture_left_censored'] == False) &
            (within_lan_df['lag'].notna())
        ].copy()
        if len(wl) > 0:
            report.append(f"\nWithin-prefecture analysis (municipality vs own prefecture):")
            report.append(f"  Comparing when municipalities adopt a term vs when their")
            report.append(f"  own regional prefecture adopted the same term.")
            report.append(f"  Valid pairs: {len(wl)} of {total_pairs} (prefecture NOT left-censored)")
            report.append(f"  Left-censored pairs excluded: {total_pairs - len(wl)} ({100*(total_pairs-len(wl))/total_pairs:.1f}%)")
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
            report.append("\n  Per-prefecture mean lag (sorted by lag, valid pairs only):")
            lan_stats = wl.groupby('lan_name')['lag'].agg(['mean', 'median', 'std', 'count'])
            lan_stats = lan_stats.sort_values('mean')
            for lan_name, row in lan_stats.iterrows():
                report.append(
                    f"    {lan_name:20s}: mean={row['mean']:+.1f}y, "
                    f"median={row['median']:+.1f}y, std={row['std']:.1f}y (n={int(row['count'])})"
                )

    # Municipal lag to national leaders
    if national_lag_stats:
        report.append(f"\nMunicipal lag to NATIONAL leaders (any prefecture):")
        report.append(f"  Comparing each municipality adoption to the earliest prefecture")
        report.append(f"  adoption of the same term nationally (including left-censored).")
        report.append(f"  N = {national_lag_stats['n_pairs']} municipality-term pairs")
        report.append("")
        report.append(f"  Mean lag:   {national_lag_stats['mean_lag']:+.2f} years")
        report.append(f"  Median lag: {national_lag_stats['median_lag']:+.2f} years")
        report.append("")
        report.append("  Direction:")
        report.append(f"    Municipality BEFORE any prefecture: {national_lag_stats['pct_before']:.1f}%")
        report.append(f"    Same year as earliest prefecture:   {national_lag_stats['pct_same']:.1f}%")
        report.append(f"    Municipality AFTER some prefecture: {national_lag_stats['pct_after']:.1f}%")

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
        '--min-year',
        type=int,
        default=2015,
        help='Minimum year to include in analysis (default: 2015)'
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
    df_full, term_cols = load_and_prepare(args.input, min_year=args.min_year)

    # Identify balanced panel municipalities (docs in all 3 waves)
    print("\nIdentifying balanced panel...")
    balanced_entities = get_balanced_panel_entities(df_full)
    n_total_entities = df_full['entity'].nunique()

    # For most analyses: use balanced panel municipalities + all prefectures + MCF
    # This avoids bias in municipality timing while keeping all prefecture data
    balanced_munis = df_full[(df_full['actor'] == 'kommun') &
                             (df_full['entity'].isin(balanced_entities))]['entity'].unique()
    all_prefectures = df_full[df_full['actor'] == 'lansstyrelse']['entity'].unique()
    all_mcf = df_full[df_full['actor'] == 'MCF']['entity'].unique()

    analysis_entities = set(balanced_munis) | set(all_prefectures) | set(all_mcf)
    df = df_full[df_full['entity'].isin(analysis_entities)]

    print(f"  Balanced panel municipalities: {len(balanced_munis)}/{df_full[df_full['actor']=='kommun']['entity'].nunique()}")
    print(f"  All prefectures: {len(all_prefectures)}")
    print(f"  Analysis entities: {len(analysis_entities)}")
    print(f"  {len(df)} documents")

    # First appearances (all analysis entities)
    print("\nComputing first appearances...")
    first_appearances = compute_first_appearances(df, term_cols)
    print(f"  {len(first_appearances)} (entity, term) first appearances")
    n_censored = first_appearances['is_left_censored'].sum()
    print(f"  Left-censored: {n_censored} ({n_censored / len(first_appearances) * 100:.1f}%)")

    # Entity coverage: what % of dictionary terms does each entity mention?
    # Uses all entities with data (not balanced panel) - entities without docs
    # in a given year are simply excluded, not coded as zero
    print("\nComputing entity coverage (all entities)...")
    entity_coverage = compute_entity_coverage(df_full, term_cols)
    print(f"  {len(entity_coverage)} entity-year observations")
    print(f"  Mean coverage: {entity_coverage['coverage_rate'].mean():.1%}")

    # Adoption curves for individual term analysis (balanced panel)
    print("\nComputing adoption curves (balanced panel only)...")
    df_balanced = df_full[df_full['entity'].isin(balanced_entities)]
    fa_balanced = first_appearances[first_appearances['entity'].isin(balanced_entities)]
    adoption_curves = compute_adoption_curves(fa_balanced, df_balanced, term_cols)
    print(f"  {len(adoption_curves)} curve data points")

    # Spikes (balanced panel)
    print(f"\nDetecting adoption spikes (threshold={args.spike_threshold:.0%})...")
    spikes_df = detect_adoption_spikes(fa_balanced, df_balanced, threshold=args.spike_threshold)
    n_spikes = spikes_df['is_spike'].sum()
    print(f"  {n_spikes} spikes detected")

    # Gini (balanced panel)
    print("\nComputing Gini coefficients (balanced panel)...")
    gini_df = compute_gini_coefficients(fa_balanced)
    print(f"  {len(gini_df)} terms with Gini scores")

    # Lead-lag (balanced panel)
    print("\nComputing lead-lag analysis (balanced panel)...")
    lead_lag_df = compute_lead_lag(fa_balanced)
    print(f"  {len(lead_lag_df)} terms analyzed")

    # Within-prefecture analysis (all prefectures, balanced municipalities)
    # Only count municipality adoptions >= prefecture's first doc year
    print("\nComputing within-prefecture analysis...")
    print("  (all prefectures, balanced-panel municipalities, aligned by prefecture coverage)")
    within_lan_df = compute_within_lan_lag(first_appearances, df=df_full)
    n_matched = first_appearances['lan'].notna().sum()
    print(f"  {n_matched} first appearances matched to prefecture")
    print(f"  {len(within_lan_df)} (prefecture, term) pairs analyzed")

    # Prefecture influence analysis (which prefectures do municipalities follow?)
    print("\nComputing prefecture influence analysis...")
    influence_result = compute_prefecture_influence(first_appearances)
    if influence_result:
        influence_df, total_follows, own_pref_pct = influence_result
        print(f"  {total_follows} municipality-term pairs with prior prefecture adoption")
        print(f"  Municipalities follow own prefecture: {own_pref_pct:.1f}%")
        print(f"  Most influential: {influence_df['prefecture'].iloc[0]} ({influence_df['influence_ratio'].iloc[0]}x)")
    else:
        influence_df = pd.DataFrame()

    # Municipal lag to national leaders
    print("\nComputing municipal lag to national leaders...")
    national_lag_stats = compute_municipal_national_lag(first_appearances)
    print(f"  {national_lag_stats['n_pairs']} municipality-term pairs")
    print(f"  Mean lag to earliest prefecture: {national_lag_stats['mean_lag']:.2f} years")
    print(f"  Municipality before any prefecture: {national_lag_stats['pct_before']:.1f}%")
    print(f"  Municipality after some prefecture: {national_lag_stats['pct_after']:.1f}%")

    # Create output subdirectories
    national_dir = args.output / 'national'
    within_dir = args.output / 'within_prefecture'
    national_dir.mkdir(parents=True, exist_ok=True)
    within_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    print("\nSaving data...")

    # Base data (root)
    first_appearances.to_csv(
        args.output / 'first_appearances.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: first_appearances.csv")

    # National diffusion data
    spikes_df.to_csv(
        national_dir / 'adoption_spikes.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: national/adoption_spikes.csv")

    gini_df.to_csv(
        national_dir / 'gini_coefficients.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: national/gini_coefficients.csv")

    lead_lag_df.to_csv(
        national_dir / 'lead_lag.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: national/lead_lag.csv")

    if len(influence_df) > 0:
        influence_df.to_csv(
            national_dir / 'prefecture_influence.csv', index=False, encoding='utf-8'
        )
        print(f"  Saved: national/prefecture_influence.csv")

    # Within-prefecture data
    within_lan_df.to_csv(
        within_dir / 'within_prefecture_lag.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: within_prefecture/within_prefecture_lag.csv")

    # Visualisations
    print("\nGenerating visualisations...")
    plot_aggregate_adoption_curves(entity_coverage, national_dir)
    plot_individual_adoption_curves(
        adoption_curves, national_dir,
        gini_df=gini_df, first_appearances=first_appearances,
        selection='top_adopted'
    )
    plot_gini_chart(gini_df, national_dir)
    plot_lead_lag(lead_lag_df, national_dir)
    plot_within_lan(within_lan_df, within_dir)

    # Report
    print("\nGenerating report...")
    generate_report(
        first_appearances, spikes_df, gini_df, lead_lag_df, within_lan_df, args.output,
        national_lag_stats=national_lag_stats
    )

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {args.output}")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
