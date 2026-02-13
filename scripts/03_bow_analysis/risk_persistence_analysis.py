#!/usr/bin/env python3
"""
Risk Persistence Analysis

Measures whether risk terms persist in RSA documents over time.
For entities (municipalities, prefectures, MCF) with ≥2 documents in
different waves, tracks which terms appear, disappear, or are newly
adopted between consecutive waves.

Waves:
    Wave 0: pre-2015
    Wave 1: 2015-2018
    Wave 2: 2019-2022
    Wave 3: >= 2023

Includes actor-type comparisons throughout.

Input:  term_document_matrix.csv (from term_document_matrix.py)
Output: persistence heatmaps, dropout/adoption rankings, Jaccard distributions

Usage:
    python risk_persistence_analysis.py \\
        --input results/term_document_matrix/term_document_matrix.csv \\
        --output results/persistence/

    python risk_persistence_analysis.py \\
        --input results/term_document_matrix/term_document_matrix.csv \\
        --output results/persistence/ \\
        --min-entities 5 --verbose

Requirements:
    pip install pandas numpy matplotlib seaborn
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set1")

METADATA_COLS = ['file', 'actor', 'entity', 'year', 'wave']

ACTOR_TRANSLATIONS = {
    'kommun': 'Municipality',
    'länsstyrelse': 'Prefecture',
    'MCF': 'MCF',
}


def translate_actor(actor: str) -> str:
    """Translate actor names from Swedish to English."""
    return ACTOR_TRANSLATIONS.get(actor, actor)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare(input_path: Path) -> tuple:
    """
    Load term-document matrix and separate metadata from term columns.

    Returns
    -------
    tuple of (pd.DataFrame, list[str])
        (full dataframe, list of term column names)
    """
    df = pd.read_csv(input_path)
    term_cols = [c for c in df.columns if c not in METADATA_COLS]
    print(f"  Loaded {len(df)} documents, {len(term_cols)} terms")
    return df, term_cols


def build_panel(df: pd.DataFrame, term_cols: list) -> pd.DataFrame:
    """
    Build a longitudinal panel: keep only entities with ≥2 documents
    in different waves, sorted by entity and wave.

    Parameters
    ----------
    df : pd.DataFrame
        Full term-document matrix.
    term_cols : list[str]
        Term column names.

    Returns
    -------
    pd.DataFrame
        Filtered and sorted panel.
    """
    # Drop duplicates: if same entity has multiple docs in same wave, keep last
    df_sorted = df.sort_values(['entity', 'wave', 'year'])
    df_dedup = df_sorted.drop_duplicates(subset=['entity', 'wave'], keep='last')

    # Keep entities with ≥2 distinct waves
    wave_counts = df_dedup.groupby('entity')['wave'].nunique()
    multi_wave = wave_counts[wave_counts >= 2].index
    panel = df_dedup[df_dedup['entity'].isin(multi_wave)].copy()
    panel = panel.sort_values(['entity', 'wave']).reset_index(drop=True)

    n_entities = panel['entity'].nunique()
    actor_breakdown = panel.groupby('actor')['entity'].nunique().to_dict()
    translated = {translate_actor(k): v for k, v in actor_breakdown.items()}
    print(f"  Panel: {len(panel)} docs from {n_entities} entities with ≥2 waves")
    print(f"  By actor: {translated}")

    return panel


# =============================================================================
# PERSISTENCE COMPUTATION
# =============================================================================

def compute_transitions(panel: pd.DataFrame, term_cols: list) -> pd.DataFrame:
    """
    For each consecutive document pair within an entity, compute
    per-term transitions: persist, dropout, adopt, stable_absent.

    Transitions are computed between consecutive waves, not years.

    Returns
    -------
    pd.DataFrame
        One row per (entity, wave_from, wave_to, term) with columns:
        entity, actor, wave_from, wave_to, term, present_from, present_to,
        transition (persist/dropout/adopt/stable_absent).
    """
    records = []

    for entity, group in panel.groupby('entity'):
        group = group.sort_values('wave')
        actor = group['actor'].iloc[0]
        docs = list(group.iterrows())

        for i in range(len(docs) - 1):
            _, doc_t = docs[i]
            _, doc_t1 = docs[i + 1]

            wave_from = doc_t['wave']
            wave_to = doc_t1['wave']

            for term in term_cols:
                present_t = int(doc_t[term] > 0)
                present_t1 = int(doc_t1[term] > 0)

                if present_t and present_t1:
                    transition = 'persist'
                elif present_t and not present_t1:
                    transition = 'dropout'
                elif not present_t and present_t1:
                    transition = 'adopt'
                else:
                    transition = 'stable_absent'

                records.append({
                    'entity': entity,
                    'actor': actor,
                    'wave_from': wave_from,
                    'wave_to': wave_to,
                    'wave_pair': f"W{int(wave_from)}→W{int(wave_to)}",
                    'term': term,
                    'present_from': present_t,
                    'present_to': present_t1,
                    'transition': transition,
                })

    return pd.DataFrame(records)


def compute_direct_wave_transitions(
    panel: pd.DataFrame, term_cols: list, wave_from: int, wave_to: int
) -> pd.DataFrame:
    """
    Compute direct transitions between two specific waves for all entities
    that have documents in BOTH waves, regardless of intermediate waves.

    This allows comparing W1→W3 for all municipalities, not just those
    that skipped W2.

    Parameters
    ----------
    panel : pd.DataFrame
        Longitudinal panel data.
    term_cols : list
        List of term column names.
    wave_from : int
        Starting wave (e.g., 1).
    wave_to : int
        Ending wave (e.g., 3).

    Returns
    -------
    pd.DataFrame
        Transition records for entities with docs in both waves.
    """
    records = []

    for entity, group in panel.groupby('entity'):
        waves_present = set(group['wave'].unique())

        # Only include entities that have BOTH waves
        if wave_from not in waves_present or wave_to not in waves_present:
            continue

        actor = group['actor'].iloc[0]

        # Get the document for each wave (latest if multiple)
        doc_from = group[group['wave'] == wave_from].sort_values('year').iloc[-1]
        doc_to = group[group['wave'] == wave_to].sort_values('year').iloc[-1]

        for term in term_cols:
            present_t = int(doc_from[term] > 0)
            present_t1 = int(doc_to[term] > 0)

            if present_t and present_t1:
                transition = 'persist'
            elif present_t and not present_t1:
                transition = 'dropout'
            elif not present_t and present_t1:
                transition = 'adopt'
            else:
                transition = 'stable_absent'

            records.append({
                'entity': entity,
                'actor': actor,
                'wave_from': wave_from,
                'wave_to': wave_to,
                'wave_pair': f"W{wave_from}→W{wave_to}",
                'term': term,
                'present_from': present_t,
                'present_to': present_t1,
                'transition': transition,
            })

    return pd.DataFrame(records)


def compute_year_transitions(df: pd.DataFrame, term_cols: list) -> pd.DataFrame:
    """
    Compute year-by-year transitions for entities with multiple documents.

    Unlike wave-based transitions, this compares consecutive YEARS.
    Useful for länsstyrelsen and MCF where wave categorisation is less meaningful.

    Parameters
    ----------
    df : pd.DataFrame
        Full term-document matrix (not panel-filtered).
    term_cols : list
        List of term column names.

    Returns
    -------
    pd.DataFrame
        Transition records with year_pair instead of wave_pair.
    """
    records = []

    for entity, group in df.groupby('entity'):
        if len(group) < 2:
            continue

        group = group.sort_values('year')
        actor = group['actor'].iloc[0]
        docs = list(group.iterrows())

        for i in range(len(docs) - 1):
            _, doc_t = docs[i]
            _, doc_t1 = docs[i + 1]

            year_from = int(doc_t['year'])
            year_to = int(doc_t1['year'])

            for term in term_cols:
                present_t = int(doc_t[term] > 0)
                present_t1 = int(doc_t1[term] > 0)

                if present_t and present_t1:
                    transition = 'persist'
                elif present_t and not present_t1:
                    transition = 'dropout'
                elif not present_t and present_t1:
                    transition = 'adopt'
                else:
                    transition = 'stable_absent'

                records.append({
                    'entity': entity,
                    'actor': actor,
                    'year_from': year_from,
                    'year_to': year_to,
                    'year_pair': f"{year_from}→{year_to}",
                    'term': term,
                    'present_from': present_t,
                    'present_to': present_t1,
                    'transition': transition,
                })

    return pd.DataFrame(records)


def compute_jaccard(panel: pd.DataFrame, term_cols: list) -> pd.DataFrame:
    """
    Compute Jaccard similarity between consecutive documents
    for each entity (by wave).

    Returns
    -------
    pd.DataFrame
        One row per (entity, wave_from, wave_to) with Jaccard score.
    """
    records = []

    for entity, group in panel.groupby('entity'):
        group = group.sort_values('wave')
        actor = group['actor'].iloc[0]
        docs = list(group.iterrows())

        for i in range(len(docs) - 1):
            _, doc_t = docs[i]
            _, doc_t1 = docs[i + 1]

            set_t = set(t for t in term_cols if doc_t[t] > 0)
            set_t1 = set(t for t in term_cols if doc_t1[t] > 0)

            union = set_t | set_t1
            intersection = set_t & set_t1

            jaccard = len(intersection) / len(union) if union else 0.0

            records.append({
                'entity': entity,
                'actor': actor,
                'wave_from': doc_t['wave'],
                'wave_to': doc_t1['wave'],
                'wave_pair': f"W{int(doc_t['wave'])}→W{int(doc_t1['wave'])}",
                'n_terms_t': len(set_t),
                'n_terms_t1': len(set_t1),
                'n_intersection': len(intersection),
                'n_union': len(union),
                'jaccard': jaccard,
            })

    return pd.DataFrame(records)


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_persistence_by_term(
    transitions: pd.DataFrame, min_entities: int = 5
) -> pd.DataFrame:
    """
    Compute persistence rate per term (aggregated across all wave transitions).

    Persistence rate = persist / (persist + dropout), i.e., fraction of
    terms present in wave T that remain in wave T+1.

    Parameters
    ----------
    transitions : pd.DataFrame
        Transition records.
    min_entities : int
        Minimum number of entity-pairs where the term was present in T.

    Returns
    -------
    pd.DataFrame
        Aggregated persistence rates.
    """
    # Filter to terms that were present in doc T (exclude stable_absent and adopt)
    present_in_t = transitions[transitions['present_from'] == 1].copy()

    # Group by term
    grouped = present_in_t.groupby('term')['transition'].value_counts().unstack(fill_value=0)
    for col in ['persist', 'dropout']:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped['n_entities_t0'] = grouped['persist'] + grouped['dropout']
    grouped['n_entities_persist'] = grouped['persist']
    grouped['n_entities_dropout'] = grouped['dropout']
    grouped['persistence_rate'] = grouped['persist'] / grouped['n_entities_t0']
    grouped['flag_low_n'] = grouped['n_entities_t0'] < 3

    # Filter by min_entities
    result = grouped[grouped['n_entities_t0'] >= min_entities].copy()

    # Reorder columns for clarity
    result = result[['n_entities_t0', 'n_entities_persist', 'n_entities_dropout',
                     'persistence_rate', 'flag_low_n']]

    return result.sort_values('persistence_rate', ascending=False)


def aggregate_by_actor_and_wave_pair(
    transitions: pd.DataFrame, min_entities: int = 3
) -> pd.DataFrame:
    """
    Compute persistence rate per term, grouped by actor and wave transition.

    Returns
    -------
    pd.DataFrame
        Rows with: actor, wave_pair, term, persistence_rate, n_entities.
    """
    present_in_t = transitions[transitions['present_from'] == 1].copy()

    records = []
    for (actor, wave_pair, term), group in present_in_t.groupby(
        ['actor', 'wave_pair', 'term']
    ):
        n_persist = (group['transition'] == 'persist').sum()
        n_dropout = (group['transition'] == 'dropout').sum()
        total = n_persist + n_dropout

        # Always include the record, but flag if low N
        flag_low_n = (total < 3)  # Flag if fewer than 3 entities

        records.append({
            'actor': actor,
            'wave_pair': wave_pair,
            'term': term,
            'persistence_rate': n_persist / total if total > 0 else 0,
            'n_entities_t0': total,
            'n_entities_persist': n_persist,
            'n_entities_dropout': n_dropout,
            'flag_low_n': flag_low_n,
        })

    # Filter by min_entities if specified
    result = pd.DataFrame(records)
    if min_entities > 1:
        result = result[result['n_entities_t0'] >= min_entities]

    return result


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_persistence_heatmap(
    transitions: pd.DataFrame,
    output_dir: Path,
    min_entities: int = 10,
    actor_filter: str = None,
    suffix: str = '',
) -> None:
    """
    Heatmap: rows = terms, columns = wave transitions,
    cell colour = persistence rate.

    Only includes consecutive wave transitions (W0→W1, W1→W2, W2→W3).
    Skip-wave transitions (e.g., W1→W3 for entities missing W2) are excluded.
    """
    # Only include consecutive wave transitions
    CONSECUTIVE_PAIRS = ['W0→W1', 'W1→W2', 'W2→W3']

    df = transitions.copy()
    df = df[df['wave_pair'].isin(CONSECUTIVE_PAIRS)]

    if actor_filter:
        df = df[df['actor'] == actor_filter]

    present_in_t = df[df['present_from'] == 1]

    # Compute persistence rate per term per wave transition
    pivot_data = []
    for (term, wave_pair), group in present_in_t.groupby(['term', 'wave_pair']):
        n_persist = (group['transition'] == 'persist').sum()
        total = len(group)
        if total >= 3:
            pivot_data.append({
                'term': term,
                'wave_pair': wave_pair,
                'persistence_rate': n_persist / total,
                'n': total,
            })

    if not pivot_data:
        print(f"  No data for persistence heatmap{suffix}")
        return

    pivot_df = pd.DataFrame(pivot_data)

    # Filter to terms with enough observations
    term_counts = pivot_df.groupby('term')['n'].sum()
    frequent_terms = term_counts[term_counts >= min_entities].index
    pivot_df = pivot_df[pivot_df['term'].isin(frequent_terms)]

    if len(pivot_df) == 0:
        print(f"  No terms meet min_entities threshold for heatmap{suffix}")
        return

    # Pivot for heatmap
    heatmap_data = pivot_df.pivot_table(
        index='term', columns='wave_pair', values='persistence_rate'
    )

    # Sort by mean persistence rate
    heatmap_data = heatmap_data.loc[
        heatmap_data.mean(axis=1).sort_values(ascending=False).index
    ]

    # Plot
    fig_height = max(6, len(heatmap_data) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    sns.heatmap(
        heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Persistence rate'}
    )

    title = 'Term persistence rate by wave'
    if actor_filter:
        title += f' ({translate_actor(actor_filter)})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Wave transition', fontsize=12)
    ax.set_ylabel('Risk term', fontsize=12)

    plt.tight_layout()
    fname = f'persistence_heatmap{suffix}'
    plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.png/pdf")


def plot_year_persistence_heatmap(
    year_transitions: pd.DataFrame,
    output_dir: Path,
    min_entities: int = 1,
    actor_filter: str = None,
    suffix: str = '',
) -> None:
    """
    Heatmap for year-by-year transitions (for länsstyrelsen/MCF).

    Since there are fewer entities, uses lower thresholds.
    """
    df = year_transitions.copy()

    if actor_filter:
        df = df[df['actor'] == actor_filter]

    if len(df) == 0:
        print(f"  No data for year persistence heatmap{suffix}")
        return

    present_in_t = df[df['present_from'] == 1]

    if len(present_in_t) == 0:
        print(f"  No presence data for year persistence heatmap{suffix}")
        return

    # Compute persistence rate per term per year transition
    pivot_data = []
    for (term, year_pair), group in present_in_t.groupby(['term', 'year_pair']):
        n_persist = (group['transition'] == 'persist').sum()
        total = len(group)
        if total >= min_entities:
            pivot_data.append({
                'term': term,
                'year_pair': year_pair,
                'persistence_rate': n_persist / total,
                'n': total,
            })

    if not pivot_data:
        print(f"  No data for year persistence heatmap{suffix}")
        return

    pivot_df = pd.DataFrame(pivot_data)

    # Filter to terms with enough observations (at least 2 year-pairs)
    term_counts = pivot_df.groupby('term').size()
    frequent_terms = term_counts[term_counts >= 2].index
    pivot_df = pivot_df[pivot_df['term'].isin(frequent_terms)]

    if len(pivot_df) == 0:
        print(f"  No terms meet threshold for year heatmap{suffix}")
        return

    # Pivot for heatmap
    heatmap_data = pivot_df.pivot_table(
        index='term', columns='year_pair', values='persistence_rate'
    )

    # Sort columns chronologically
    def year_pair_sort_key(yp):
        try:
            return int(yp.split('→')[0])
        except:
            return 0
    sorted_cols = sorted(heatmap_data.columns, key=year_pair_sort_key)
    heatmap_data = heatmap_data[sorted_cols]

    # Sort rows by mean persistence rate
    heatmap_data = heatmap_data.loc[
        heatmap_data.mean(axis=1).sort_values(ascending=False).index
    ]

    # Plot
    fig_width = max(10, len(heatmap_data.columns) * 0.8)
    fig_height = max(6, len(heatmap_data) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Persistence rate'}
    )

    title = 'Term persistence rate by year'
    if actor_filter:
        title += f' ({translate_actor(actor_filter)})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Year transition', fontsize=12)
    ax.set_ylabel('Risk term', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    fname = f'persistence_heatmap_year{suffix}'
    plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.png/pdf")


def plot_direct_wave_heatmap(
    transitions: pd.DataFrame,
    output_dir: Path,
    wave_pair_label: str,
    min_entities: int = 3,
    actor_filter: str = None,
    suffix: str = '',
) -> None:
    """
    Heatmap for a specific direct wave comparison (e.g., W1→W3 for all municipalities).
    """
    df = transitions.copy()

    if actor_filter:
        df = df[df['actor'] == actor_filter]

    present_in_t = df[df['present_from'] == 1]

    # Compute persistence rate per term
    pivot_data = []
    for term, group in present_in_t.groupby('term'):
        n_persist = (group['transition'] == 'persist').sum()
        total = len(group)
        if total >= min_entities:
            pivot_data.append({
                'term': term,
                'persistence_rate': n_persist / total,
                'n': total,
            })

    if not pivot_data:
        print(f"  No data for direct wave heatmap{suffix}")
        return

    pivot_df = pd.DataFrame(pivot_data)

    # Sort by persistence rate
    pivot_df = pivot_df.sort_values('persistence_rate', ascending=False)

    # Create single-column heatmap data
    heatmap_data = pivot_df.set_index('term')[['persistence_rate']]
    heatmap_data.columns = [wave_pair_label]

    # Plot
    fig_height = max(6, len(heatmap_data) * 0.3)
    fig, ax = plt.subplots(figsize=(6, fig_height))

    sns.heatmap(
        heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Persistence rate'}
    )

    title = f'Term persistence rate ({wave_pair_label})'
    if actor_filter:
        title += f' — {translate_actor(actor_filter)}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('Risk term', fontsize=12)

    plt.tight_layout()
    fname = f'persistence_heatmap{suffix}'
    plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.png/pdf")


def plot_dropout_adoption_ranking(
    transitions: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Bar charts showing terms most frequently dropped and adopted.
    """
    for transition_type, label in [('dropout', 'Dropout'), ('adopt', 'Adoption')]:
        subset = transitions[transitions['transition'] == transition_type]

        if len(subset) == 0:
            continue

        # Count by term and actor
        counts = subset.groupby(['term', 'actor']).size().reset_index(name='count')

        # Get top terms overall
        term_totals = counts.groupby('term')['count'].sum().nlargest(top_n)
        top_terms = term_totals.index.tolist()

        counts_top = counts[counts['term'].isin(top_terms)]

        # Pivot for stacked bar
        pivot = counts_top.pivot_table(
            index='term', columns='actor', values='count', fill_value=0
        )
        pivot = pivot.loc[top_terms]  # Maintain sort order
        pivot.columns = [translate_actor(c) for c in pivot.columns]

        fig, ax = plt.subplots(figsize=(10, 8))
        pivot.plot(kind='barh', stacked=True, ax=ax, alpha=0.8)

        ax.set_xlabel(f'Number of {label.lower()} events', fontsize=12)
        ax.set_ylabel('Risk term', fontsize=12)
        ax.set_title(
            f'Top {top_n} terms by {label.lower()} frequency',
            fontsize=14, fontweight='bold'
        )
        ax.legend(title='Actor')
        ax.invert_yaxis()

        plt.tight_layout()
        fname = f'{transition_type}_ranking'
        plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}.png/pdf")


def plot_jaccard_by_actor(jaccard_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Boxplot of Jaccard similarity scores grouped by actor type.
    """
    if len(jaccard_df) == 0:
        return

    df = jaccard_df.copy()
    df['actor_en'] = df['actor'].map(translate_actor)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        data=df, x='actor_en', y='jaccard',
        ax=ax, palette='Set1', width=0.5,
    )
    sns.stripplot(
        data=df, x='actor_en', y='jaccard',
        ax=ax, color='black', alpha=0.3, size=4, jitter=True,
    )

    ax.set_xlabel('Actor type', fontsize=12)
    ax.set_ylabel('Jaccard similarity', fontsize=12)
    ax.set_title(
        'Risk term overlap between consecutive RSAs by actor type',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'jaccard_by_actor.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'jaccard_by_actor.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: jaccard_by_actor.png/pdf")


def plot_actor_persistence_comparison(
    transitions: pd.DataFrame, output_dir: Path
) -> None:
    """
    Grouped bar chart: mean persistence rate per actor type per wave transition.

    Only includes consecutive wave transitions (W0→W1, W1→W2, W2→W3).
    """
    CONSECUTIVE_PAIRS = ['W0→W1', 'W1→W2', 'W2→W3']

    df = transitions[transitions['wave_pair'].isin(CONSECUTIVE_PAIRS)].copy()
    present_in_t = df[df['present_from'] == 1].copy()

    # Compute persistence rate per entity-wave transition, then average by actor
    entity_rates = []
    for (entity, wave_pair), group in present_in_t.groupby(['entity', 'wave_pair']):
        n_persist = (group['transition'] == 'persist').sum()
        total = len(group)
        entity_rates.append({
            'entity': entity,
            'actor': group['actor'].iloc[0],
            'wave_pair': wave_pair,
            'persistence_rate': n_persist / total if total > 0 else 0,
        })

    rates_df = pd.DataFrame(entity_rates)
    rates_df['actor_en'] = rates_df['actor'].map(translate_actor)

    if len(rates_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=rates_df, x='wave_pair', y='persistence_rate',
        hue='actor_en', ax=ax, palette='Set1', alpha=0.8,
    )

    ax.set_xlabel('Wave transition', fontsize=12)
    ax.set_ylabel('Mean persistence rate', fontsize=12)
    ax.set_title(
        'Mean persistence rate by actor type and wave',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylim(0, 1)
    ax.legend(title='Actor')

    plt.tight_layout()
    plt.savefig(output_dir / 'persistence_by_actor.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'persistence_by_actor.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: persistence_by_actor.png/pdf")


# =============================================================================
# REPORT
# =============================================================================

def generate_report(
    panel: pd.DataFrame,
    transitions: pd.DataFrame,
    jaccard_df: pd.DataFrame,
    persistence_by_term: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate a comprehensive text report."""
    report = []
    report.append("=" * 70)
    report.append("RISK PERSISTENCE ANALYSIS — REPORT")
    report.append("=" * 70)

    # Panel summary
    report.append(f"\nPanel: {panel['entity'].nunique()} entities, {len(panel)} documents")
    for actor in sorted(panel['actor'].unique()):
        n_ent = panel[panel['actor'] == actor]['entity'].nunique()
        n_doc = len(panel[panel['actor'] == actor])
        report.append(f"  {translate_actor(actor)}: {n_ent} entities, {n_doc} documents")

    # Overall persistence
    present_in_t = transitions[transitions['present_from'] == 1]
    n_persist = (present_in_t['transition'] == 'persist').sum()
    n_dropout = (present_in_t['transition'] == 'dropout').sum()
    total = n_persist + n_dropout
    if total > 0:
        report.append(f"\nOverall persistence rate: {n_persist / total:.1%}")
        report.append(f"  Persist: {n_persist}, Dropout: {n_dropout}")

    # By actor
    report.append("\nPersistence rate by actor type:")
    for actor in sorted(transitions['actor'].unique()):
        actor_data = present_in_t[present_in_t['actor'] == actor]
        ap = (actor_data['transition'] == 'persist').sum()
        ad = (actor_data['transition'] == 'dropout').sum()
        at = ap + ad
        if at > 0:
            report.append(f"  {translate_actor(actor)}: {ap / at:.1%} (n={at})")

    # Jaccard summary
    report.append("\nJaccard similarity by actor:")
    for actor in sorted(jaccard_df['actor'].unique()):
        actor_j = jaccard_df[jaccard_df['actor'] == actor]['jaccard']
        report.append(
            f"  {translate_actor(actor)}: "
            f"mean={actor_j.mean():.3f}, median={actor_j.median():.3f}, "
            f"std={actor_j.std():.3f} (n={len(actor_j)})"
        )

    # Top persistent terms
    report.append("\nTop 20 most persistent terms:")
    top_persist = persistence_by_term.head(20)
    for term, row in top_persist.iterrows():
        report.append(
            f"  {term:30s}: {row['persistence_rate']:.1%} "
            f"(persisted {int(row['n_entities_persist'])}/{int(row['n_entities_t0'])} entities)"
        )

    # Top dropout terms
    report.append("\nTop 20 most frequently dropped terms:")
    dropout_ranked = persistence_by_term.sort_values('persistence_rate').head(20)
    for term, row in dropout_ranked.iterrows():
        report.append(
            f"  {term:30s}: {1 - row['persistence_rate']:.1%} dropout rate "
            f"({int(row['n_entities_dropout'])}/{int(row['n_entities_t0'])} entities)"
        )

    # Save
    report_text = '\n'.join(report)
    report_path = output_dir / 'persistence_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Saved: persistence_report.txt")
    print(f"\n{report_text}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze risk term persistence across consecutive RSA documents'
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
        default=Path('./results/persistence'),
        help='Output directory for figures and data'
    )

    parser.add_argument(
        '--min-entities',
        type=int,
        default=10,
        help='Minimum entity-pairs for a term to appear in heatmap (default: 10)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages'
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RISK PERSISTENCE ANALYSIS")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {args.input}")
    df, term_cols = load_and_prepare(args.input)

    # Build panel
    print("\nBuilding longitudinal panel...")
    panel = build_panel(df, term_cols)

    # Compute transitions
    print("\nComputing term transitions...")
    transitions = compute_transitions(panel, term_cols)
    print(f"  {len(transitions)} transition records")

    transition_counts = transitions['transition'].value_counts()
    for t, c in transition_counts.items():
        print(f"    {t}: {c}")

    # Compute Jaccard
    print("\nComputing Jaccard similarity...")
    jaccard_df = compute_jaccard(panel, term_cols)
    print(f"  {len(jaccard_df)} entity-pair comparisons")
    print(f"  Mean Jaccard: {jaccard_df['jaccard'].mean():.3f}")

    # Aggregate persistence by term
    print("\nAggregating persistence rates...")
    persistence_by_term = aggregate_persistence_by_term(
        transitions, min_entities=args.min_entities
    )
    print(f"  {len(persistence_by_term)} terms above threshold")

    # Save data
    print("\nSaving data...")
    transitions.to_csv(
        args.output / 'persistence_transitions.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: persistence_transitions.csv")

    persistence_by_term.to_csv(
        args.output / 'persistence_by_term.csv', encoding='utf-8'
    )
    print(f"  Saved: persistence_by_term.csv")

    jaccard_df.to_csv(
        args.output / 'jaccard_scores.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: jaccard_scores.csv")

    # Visualisations
    print("\nGenerating visualisations...")

    # Combined heatmap (all actors, wave-based)
    plot_persistence_heatmap(
        transitions, args.output, min_entities=args.min_entities
    )

    # Municipality wave-based heatmap (W0→W1, W1→W2, W2→W3)
    kommun_transitions = transitions[transitions['actor'] == 'kommun']
    plot_persistence_heatmap(
        kommun_transitions, args.output,
        min_entities=max(3, args.min_entities // 3),
        suffix='_kommun',
    )

    # Municipality W1→W3 direct comparison (all municipalities with both waves)
    print("\nComputing W1→W3 direct transitions for municipalities...")
    kommun_panel = panel[panel['actor'] == 'kommun']
    w1_w3_transitions = compute_direct_wave_transitions(
        kommun_panel, term_cols, wave_from=1, wave_to=3
    )
    if len(w1_w3_transitions) > 0:
        n_entities = w1_w3_transitions['entity'].nunique()
        print(f"  {n_entities} municipalities with both W1 and W3 documents")
        plot_direct_wave_heatmap(
            w1_w3_transitions, args.output,
            wave_pair_label='W1→W3',
            min_entities=3,
            suffix='_kommun_w1_w3',
        )
        # Save W1→W3 transitions
        w1_w3_transitions.to_csv(
            args.output / 'persistence_transitions_kommun_w1_w3.csv',
            index=False, encoding='utf-8'
        )
        print(f"  Saved: persistence_transitions_kommun_w1_w3.csv")
    else:
        print("  No municipalities with both W1 and W3 documents")

    # Year-by-year transitions for länsstyrelsen and MCF
    print("\nComputing year-by-year transitions for länsstyrelsen and MCF...")
    for actor in ['länsstyrelse', 'MCF']:
        actor_df = df[df['actor'] == actor]
        if len(actor_df) < 2:
            print(f"  Skipping {actor}: not enough documents")
            continue

        year_trans = compute_year_transitions(actor_df, term_cols)
        if len(year_trans) == 0:
            print(f"  Skipping {actor}: no year transitions computed")
            continue

        n_entities = year_trans['entity'].nunique()
        n_pairs = year_trans['year_pair'].nunique()
        print(f"  {translate_actor(actor)}: {n_entities} entities, {n_pairs} year-pairs")

        plot_year_persistence_heatmap(
            year_trans, args.output,
            min_entities=1,
            suffix=f'_{actor}',
        )

        # Save year transitions
        year_trans.to_csv(
            args.output / f'persistence_transitions_year_{actor}.csv',
            index=False, encoding='utf-8'
        )

    # Dropout and adoption rankings
    plot_dropout_adoption_ranking(transitions, args.output)

    # Jaccard by actor
    plot_jaccard_by_actor(jaccard_df, args.output)

    # Actor persistence comparison
    plot_actor_persistence_comparison(transitions, args.output)

    # Report
    print("\nGenerating report...")
    generate_report(
        panel, transitions, jaccard_df, persistence_by_term, args.output
    )

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {args.output}")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
