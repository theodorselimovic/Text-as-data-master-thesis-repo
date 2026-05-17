#!/usr/bin/env python3
"""
Risk Persistence Analysis

Measures whether risk terms persist in RSA documents over time.
For entities (municipalities, prefectures, MCF) with ≥2 documents in
different waves, tracks which terms appear, disappear, or are newly
adopted between consecutive waves.

Waves (starting from 2015):
    Wave 1: 2015-2018
    Wave 2: 2019-2022
    Wave 3: 2023+

Note: Pre-2015 data (wave 0) is excluded from analysis.

Includes actor-type comparisons throughout.

Input:  term_document_matrix.csv (from term_document_matrix.py)
Output: persistence heatmaps, dropout/adoption rankings, Jaccard distributions

Usage:
    python risk_persistence_analysis.py \\
        --input results/01_bow_analysis/term_matrices/term_document_matrix.csv \\
        --output results/01_bow_analysis/persistence/

    python risk_persistence_analysis.py \\
        --input results/01_bow_analysis/term_matrices/term_document_matrix.csv \\
        --output results/01_bow_analysis/persistence/ \\
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

# Import translations
sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries.risk_translations import (
    translate_term,
    translate_actor as _translate_actor,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set1")

METADATA_COLS = ['file', 'actor', 'entity', 'year', 'wave']

ACTOR_TRANSLATIONS = {
    'kommun': 'Municipality',
    'lansstyrelse': 'Prefecture',
    'MCF': 'MCF',
}


def translate_actor(actor: str) -> str:
    """Translate actor names from Swedish to English."""
    return ACTOR_TRANSLATIONS.get(actor, _translate_actor(actor))


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
    Useful for prefectures and MCF where wave categorisation is less meaningful.

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
    for each entity (by wave). Used for municipalities.

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
                'period_pair': f"W{int(doc_t['wave'])}→W{int(doc_t1['wave'])}",
                'n_terms_t': len(set_t),
                'n_terms_t1': len(set_t1),
                'n_intersection': len(intersection),
                'n_union': len(union),
                'jaccard': jaccard,
            })

    return pd.DataFrame(records)


def compute_jaccard_yearly(df: pd.DataFrame, term_cols: list) -> pd.DataFrame:
    """
    Compute Jaccard similarity between consecutive documents
    for each entity (by year). Used for prefectures and MCF.

    Returns
    -------
    pd.DataFrame
        One row per (entity, year_from, year_to) with Jaccard score.
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

            set_t = set(t for t in term_cols if doc_t[t] > 0)
            set_t1 = set(t for t in term_cols if doc_t1[t] > 0)

            union = set_t | set_t1
            intersection = set_t & set_t1

            jaccard = len(intersection) / len(union) if union else 0.0

            year_from = int(doc_t['year'])
            year_to = int(doc_t1['year'])

            records.append({
                'entity': entity,
                'actor': actor,
                'year_from': year_from,
                'year_to': year_to,
                'period_pair': f"{year_from}→{year_to}",
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
# CHARACTER PERSISTENCE
# =============================================================================

def load_character_matrix(input_path: Path) -> tuple:
    """
    Load character document matrix (parallel to term_document_matrix).

    Returns (DataFrame, list of term columns).
    """
    char_path = input_path.parent / 'character_document_matrix.csv'
    if not char_path.exists():
        print(f"  Character matrix not found: {char_path}")
        return None, []

    df = pd.read_csv(char_path)
    term_cols = [c for c in df.columns if c not in METADATA_COLS]
    print(f"  Loaded character matrix: {len(df)} documents, {len(term_cols)} terms")
    return df, term_cols


def compute_character_by_wave(
    char_df: pd.DataFrame,
    term_cols: list,
) -> pd.DataFrame:
    """
    Compute average characters per document per risk per wave.

    Only counts documents that actually mention the risk (chars > 0).
    This measures depth of coverage when discussed, not diluted by non-mentions.

    Returns long-format DataFrame: wave, actor, term, mean_chars, n_docs_with_risk.
    """
    records = []

    for (wave, actor), group in char_df.groupby(['wave', 'actor']):
        n_docs_total = len(group)

        for term in term_cols:
            term_chars = group[term]
            docs_with_risk = term_chars > 0
            n_docs_with_risk = docs_with_risk.sum()
            total_chars = term_chars.sum()

            mean_chars = total_chars / n_docs_with_risk if n_docs_with_risk > 0 else 0

            records.append({
                'wave': int(wave),
                'actor': actor,
                'term': term,
                'total_chars': total_chars,
                'n_docs_with_risk': n_docs_with_risk,
                'n_docs_total': n_docs_total,
                'mean_chars_per_doc': mean_chars,
            })

    return pd.DataFrame(records)


def compute_character_deltas(char_by_wave: pd.DataFrame) -> pd.DataFrame:
    """
    Compute wave-to-wave change in character coverage per risk.

    Returns DataFrame with delta and percent change between waves.
    """
    records = []

    for (actor, term), group in char_by_wave.groupby(['actor', 'term']):
        group = group.sort_values('wave')
        rows = list(group.iterrows())

        for i in range(len(rows) - 1):
            _, row_from = rows[i]
            _, row_to = rows[i + 1]

            w_from, w_to = row_from['wave'], row_to['wave']
            c_from, c_to = row_from['mean_chars_per_doc'], row_to['mean_chars_per_doc']
            n_from, n_to = row_from['n_docs_with_risk'], row_to['n_docs_with_risk']

            delta = c_to - c_from
            pct_change = (delta / c_from * 100) if c_from > 0 else (100 if c_to > 0 else 0)

            records.append({
                'actor': actor,
                'term': term,
                'wave_from': w_from,
                'wave_to': w_to,
                'wave_pair': f"W{w_from}→W{w_to}",
                'n_docs_from': n_from,
                'n_docs_to': n_to,
                'chars_from': c_from,
                'chars_to': c_to,
                'delta_chars': delta,
                'pct_change': pct_change,
            })

    return pd.DataFrame(records)


def plot_character_trends(
    char_by_wave: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """
    Line plot showing character coverage trends per risk across waves.
    Only shows Municipality and Prefecture (MCF excluded due to small N).
    """
    WAVE_LABELS = {1: '2015-18', 2: '2019-22', 3: '2023+'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    actors = ['kommun', 'lansstyrelse']

    for ax, actor in zip(axes, actors):
        actor_data = char_by_wave[char_by_wave['actor'] == actor]
        if len(actor_data) == 0:
            ax.set_title(f'{translate_actor(actor)} (no data)')
            continue

        # Get top N risks by average character coverage
        top_risks = (
            actor_data.groupby('term')['mean_chars_per_doc']
            .mean()
            .nlargest(top_n)
            .index.tolist()
        )

        for term in top_risks:
            term_data = actor_data[actor_data['term'] == term].sort_values('wave')
            ax.plot(
                term_data['wave'],
                term_data['mean_chars_per_doc'],
                marker='o',
                label=translate_term(term),
                linewidth=2,
                markersize=6,
            )

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([WAVE_LABELS.get(w, str(w)) for w in [1, 2, 3]])
        ax.set_xlabel('Wave', fontsize=12)
        ax.set_title(translate_actor(actor), fontsize=14, fontweight='bold')

        if actor == 'kommun':
            ax.set_ylabel('Avg characters per document', fontsize=12)
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Character coverage per risk by wave', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'character_trends.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'character_trends.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: character_trends.png/pdf")


def plot_character_heatmap(
    char_deltas: pd.DataFrame,
    output_dir: Path,
    actor: str = 'kommun',
) -> None:
    """
    Heatmap showing percent change in character coverage between waves.
    Shows ALL risks, not just top N.
    """
    actor_data = char_deltas[char_deltas['actor'] == actor].copy()
    if len(actor_data) == 0:
        print(f"  No character delta data for {actor}")
        return

    # Filter to risks with any coverage (exclude always-zero risks)
    risks_with_coverage = actor_data.groupby('term').apply(
        lambda g: (g['chars_from'].sum() + g['chars_to'].sum()) > 0
    )
    active_risks = risks_with_coverage[risks_with_coverage].index.tolist()
    actor_data = actor_data[actor_data['term'].isin(active_risks)]

    pivot = actor_data.pivot_table(
        index='term', columns='wave_pair', values='pct_change'
    )

    pivot.index = [translate_term(t) for t in pivot.index]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, max(10, len(pivot) * 0.3)))
    sns.heatmap(
        pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
        linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Percent change (%)'}
    )

    ax.set_title(f'Character coverage change by wave ({translate_actor(actor)}) - {len(pivot)} risks',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Wave transition', fontsize=12)
    ax.set_ylabel('Risk term', fontsize=12)

    plt.tight_layout()
    fname = f'character_change_heatmap_{actor}'
    plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.png/pdf ({len(pivot)} risks)")


def plot_character_change_histogram(
    char_deltas: pd.DataFrame,
    output_dir: Path,
    min_docs: int = 5,
    entity_persistence: pd.DataFrame = None,
) -> None:
    """
    Histogram showing the distribution of absolute character changes.
    Top row: Municipality wave-on-wave changes.
    Bottom row: Prefecture year-on-year changes (from entity-level data).
    """
    # Municipality data (wave-on-wave aggregated)
    kommun_deltas = char_deltas[char_deltas['actor'] == 'kommun'].copy()
    kommun_deltas['abs_change'] = kommun_deltas['chars_to'] - kommun_deltas['chars_from']
    wave_pairs = sorted(kommun_deltas['wave_pair'].unique())

    # Prefecture data (year-on-year from entity persistence)
    has_prefecture = (entity_persistence is not None and
                      'lansstyrelse' in entity_persistence['actor'].values)

    if has_prefecture:
        pref_data = entity_persistence[
            (entity_persistence['actor'] == 'lansstyrelse') &
            (entity_persistence['chars_from'] > 0)
        ].copy()
        pref_data['abs_change'] = pref_data['chars_to'] - pref_data['chars_from']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(1, len(wave_pairs), figsize=(6 * len(wave_pairs), 5))
        if len(wave_pairs) == 1:
            axes = [axes]
        axes = [axes] if len(wave_pairs) == 1 else axes

    # Top row: Municipality wave-on-wave
    for col, wave_pair in enumerate(wave_pairs):
        ax = axes[0, col] if has_prefecture else axes[col]
        subset = kommun_deltas[
            (kommun_deltas['wave_pair'] == wave_pair) &
            (kommun_deltas['n_docs_from'] >= min_docs)
        ]

        if len(subset) == 0:
            ax.set_title(f'Municipality {wave_pair} (no data)')
            ax.set_visible(False)
            continue

        abs_changes = subset['abs_change']
        ax.hist(abs_changes, bins=15, color='#e41a1c', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=2, label='No change')
        ax.axvline(abs_changes.mean(), color='green', linestyle='-', linewidth=2,
                  label=f'Mean: {abs_changes.mean():,.0f}')
        ax.set_xlabel('Character change')
        ax.set_ylabel('Number of risks')
        ax.set_title(f'Municipality {wave_pair} (n={len(subset)})')
        ax.legend(fontsize=8)

    # Bottom row: Prefecture year-on-year
    if has_prefecture:
        # Aggregate by term to get mean change per risk
        pref_by_term = pref_data.groupby('term').agg(
            mean_abs_change=('abs_change', 'mean'),
            n_transitions=('abs_change', 'count'),
        ).reset_index()

        ax = axes[1, 0]
        abs_changes = pref_by_term['mean_abs_change']
        ax.hist(abs_changes, bins=15, color='#377eb8', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=2, label='No change')
        ax.axvline(abs_changes.mean(), color='green', linestyle='-', linewidth=2,
                  label=f'Mean: {abs_changes.mean():,.0f}')
        ax.set_xlabel('Mean character change per risk')
        ax.set_ylabel('Number of risks')
        ax.set_title(f'Prefecture year-on-year (n={len(pref_by_term)} risks)')
        ax.legend(fontsize=8)

        # Hide unused subplot
        axes[1, 1].set_visible(False)

    plt.suptitle(f'Distribution of character coverage changes\n(Municipality: ≥{min_docs} docs; Prefecture: year-on-year)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'character_change_histogram.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'character_change_histogram.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: character_change_histogram.png/pdf")


def compute_entity_character_persistence(
    char_df: pd.DataFrame,
    term_cols: list,
) -> pd.DataFrame:
    """
    Compute character persistence at the entity level.

    For municipalities: compares across waves (latest doc per wave).
    For prefectures: compares across years (latest doc per year).

    Returns DataFrame with one row per (entity, term, transition).
    """
    records = []

    for (entity, actor), group in char_df.groupby(['entity', 'actor']):
        if actor == 'kommun':
            # For municipalities: keep latest doc per wave, compare across waves
            group = group.sort_values(['wave', 'year'])
            group = group.drop_duplicates(subset='wave', keep='last')
            if len(group) < 2:
                continue
            group = group.sort_values('wave')
            period_col = 'wave'
        else:
            # For prefectures/MCF: compare across years
            group = group.sort_values('year')
            group = group.drop_duplicates(subset='year', keep='last')
            if len(group) < 2:
                continue
            period_col = 'year'

        docs = list(group.iterrows())

        for i in range(len(docs) - 1):
            _, doc_from = docs[i]
            _, doc_to = docs[i + 1]

            period_from = int(doc_from[period_col])
            period_to = int(doc_to[period_col])

            # Skip if same period (shouldn't happen after dedup, but safety check)
            if period_from == period_to:
                continue

            if actor == 'kommun':
                period_pair = f"W{period_from}→W{period_to}"
            else:
                period_pair = f"{period_from}→{period_to}"

            for term in term_cols:
                chars_from = doc_from[term]
                chars_to = doc_to[term]

                # Only include if risk was mentioned in at least one doc
                if chars_from == 0 and chars_to == 0:
                    continue

                if chars_from > 0:
                    pct_change = (chars_to - chars_from) / chars_from * 100
                else:
                    pct_change = 100.0  # New adoption

                records.append({
                    'entity': entity,
                    'actor': actor,
                    'period_from': period_from,
                    'period_to': period_to,
                    'period_pair': period_pair,
                    'term': term,
                    'chars_from': chars_from,
                    'chars_to': chars_to,
                    'pct_change': pct_change,
                })

    return pd.DataFrame(records)


def aggregate_entity_character_persistence(
    entity_persistence: pd.DataFrame,
    min_entities: int = 5,
) -> pd.DataFrame:
    """
    Aggregate entity-level character persistence to risk level.

    Computes both percent change and absolute character change.
    """
    records = []

    for (actor, term), group in entity_persistence.groupby(['actor', 'term']):
        # Filter to entities where risk was present in the "from" doc
        present_from = group[group['chars_from'] > 0]

        if len(present_from) < min_entities:
            continue

        # Absolute change = chars_to - chars_from
        abs_change = present_from['chars_to'] - present_from['chars_from']

        records.append({
            'actor': actor,
            'term': term,
            'n_entities': len(present_from),
            'mean_pct_change': present_from['pct_change'].mean(),
            'median_pct_change': present_from['pct_change'].median(),
            'mean_abs_change': abs_change.mean(),
            'median_abs_change': abs_change.median(),
            'std_abs_change': abs_change.std(),
        })

    result = pd.DataFrame(records)
    if len(result) > 0:
        result = result.sort_values(['actor', 'mean_abs_change'], ascending=[True, False])
    return result


def plot_entity_character_persistence(
    agg_persistence: pd.DataFrame,
    output_dir: Path,
    top_n: int = 25,
) -> None:
    """
    Bar chart showing mean absolute character change per risk across entities.
    """
    actors = ['kommun', 'lansstyrelse']
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    for ax, actor in zip(axes, actors):
        subset = agg_persistence[agg_persistence['actor'] == actor].copy()
        if len(subset) == 0:
            ax.set_visible(False)
            continue

        # Get top growing and shrinking by mean absolute change
        top = subset.nlargest(top_n, 'mean_abs_change')
        bottom = subset.nsmallest(top_n, 'mean_abs_change')
        combined = pd.concat([top, bottom]).drop_duplicates()
        combined = combined.sort_values('mean_abs_change', ascending=True)
        combined['term_en'] = combined['term'].apply(translate_term)

        colors = ['#e41a1c' if x < 0 else '#4daf4a' for x in combined['mean_abs_change']]

        ax.barh(combined['term_en'], combined['mean_abs_change'], color=colors)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('Mean character change')
        ax.set_title(f'{translate_actor(actor)}', fontsize=12, fontweight='bold')

        # Annotate with n
        for i, (_, row) in enumerate(combined.iterrows()):
            ax.annotate(f"n={int(row['n_entities'])}",
                       xy=(row['mean_abs_change'], i),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=8, va='center')

    plt.suptitle('Character change by risk\n(mean across entities)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'character_entity_ranking.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'character_entity_ranking.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: character_entity_ranking.png/pdf")


def plot_entity_character_histogram(
    entity_persistence: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Histogram showing distribution of entity-level absolute character changes.
    Municipalities (wave-on-wave) and Prefectures (year-on-year).
    """
    # Filter to transitions where risk was present in "from" doc
    present_from = entity_persistence[entity_persistence['chars_from'] > 0].copy()
    present_from['abs_change'] = present_from['chars_to'] - present_from['chars_from']

    actors = ['kommun', 'lansstyrelse']
    actor_labels = {
        'kommun': 'Municipality (wave-on-wave)',
        'lansstyrelse': 'Prefecture (year-on-year)',
    }
    actor_colors = {'kommun': '#e41a1c', 'lansstyrelse': '#377eb8'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, actor in zip(axes, actors):
        subset = present_from[present_from['actor'] == actor]
        if len(subset) == 0:
            ax.set_visible(False)
            continue

        abs_changes = subset['abs_change']

        ax.hist(abs_changes, bins=30, color=actor_colors[actor], alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=2, label='No change')
        ax.axvline(abs_changes.mean(), color='green', linestyle='-', linewidth=2,
                  label=f'Mean: {abs_changes.mean():,.0f}')

        ax.set_xlabel('Character change')
        ax.set_ylabel('Count (entity-risk transitions)')
        ax.set_title(f'{actor_labels[actor]} (n={len(subset):,})')
        ax.legend(fontsize=9)

    plt.suptitle('Distribution of entity-level character changes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'character_entity_histogram.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'character_entity_histogram.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: character_entity_histogram.png/pdf")


def plot_top_growing_risks(
    char_deltas: pd.DataFrame,
    output_dir: Path,
    min_docs: int = 5,
    top_n: int = 20,
) -> None:
    """
    Bar chart showing top growing and shrinking risks by absolute character change.
    """
    # Filter to kommun and risks with sufficient baseline
    kommun = char_deltas[
        (char_deltas['actor'] == 'kommun') &
        (char_deltas['n_docs_from'] >= min_docs)
    ].copy()
    kommun['abs_change'] = kommun['chars_to'] - kommun['chars_from']

    if len(kommun) == 0:
        print("  No data for top growing risks")
        return

    # Create figure with two rows (one per wave transition)
    wave_pairs = sorted(kommun['wave_pair'].unique())
    fig, axes = plt.subplots(len(wave_pairs), 1, figsize=(12, 6 * len(wave_pairs)))
    if len(wave_pairs) == 1:
        axes = [axes]

    for ax, wave_pair in zip(axes, wave_pairs):
        subset = kommun[kommun['wave_pair'] == wave_pair].copy()
        subset['term_en'] = subset['term'].apply(translate_term)

        # Get top growing and top shrinking by absolute change
        top_growing = subset.nlargest(top_n, 'abs_change')
        top_shrinking = subset.nsmallest(top_n, 'abs_change')

        # Combine and sort
        combined = pd.concat([top_growing, top_shrinking]).drop_duplicates()
        combined = combined.sort_values('abs_change', ascending=True)

        # Color by direction
        colors = ['#e41a1c' if x < 0 else '#4daf4a' for x in combined['abs_change']]

        ax.barh(combined['term_en'], combined['abs_change'], color=colors)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('Character change')
        ax.set_ylabel('Risk')
        ax.set_title(f'{wave_pair}: Top growing (green) and shrinking (red) risks',
                     fontsize=12, fontweight='bold')

        # Add doc counts as annotations
        for i, (_, row) in enumerate(combined.iterrows()):
            ax.annotate(f"n={int(row['n_docs_from'])}→{int(row['n_docs_to'])}",
                       xy=(row['abs_change'], i),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=8, va='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'character_top_changes.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'character_top_changes.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: character_top_changes.png/pdf")

    # Also save ranked CSV with absolute change
    for wave_pair in wave_pairs:
        subset = kommun[kommun['wave_pair'] == wave_pair].copy()
        subset = subset.sort_values('abs_change', ascending=False)
        subset['term_en'] = subset['term'].apply(translate_term)
        subset = subset[['term', 'term_en', 'n_docs_from', 'n_docs_to',
                        'chars_from', 'chars_to', 'abs_change', 'pct_change']]
        subset.to_csv(output_dir / f'character_ranking_{wave_pair.replace("→", "_to_")}.csv',
                     index=False)
    print(f"  Saved: character_ranking_*.csv")


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
    CONSECUTIVE_PAIRS = ['W1→W2', 'W2→W3']

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

    # Translate index to English
    heatmap_data.index = [translate_term(t) for t in heatmap_data.index]

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
    Heatmap for year-by-year transitions (for prefectures/MCF).

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

    # Translate index to English
    heatmap_data.index = [translate_term(t) for t in heatmap_data.index]

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
    # Translate index to English
    heatmap_data.index = [translate_term(t) for t in heatmap_data.index]

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
        # Translate index to English
        pivot.index = [translate_term(t) for t in pivot.index]

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
    Horizontal boxplot of Jaccard similarity scores grouped by actor type.
    Actor on y-axis, Jaccard on x-axis.

    Note: Municipalities use wave-based comparison, prefectures and MCF use
    year-based comparison (combined in jaccard_df with 'period_pair' column).
    """
    if len(jaccard_df) == 0:
        return

    df = jaccard_df.copy()
    df['actor_en'] = df['actor'].map(translate_actor)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df, y='actor_en', x='jaccard',
        ax=ax, palette='Set1', width=0.5,
        orient='h',
        showfliers=False,
        whis=[0, 100],  # Extend whiskers to full data range
    )
    sns.stripplot(
        data=df, y='actor_en', x='jaccard',
        ax=ax, color='black', alpha=0.4, size=5, jitter=True,
        orient='h',
    )

    ax.set_ylabel('Actor type', fontsize=12)
    ax.set_xlabel('Jaccard similarity', fontsize=12)
    ax.set_title(
        'Risk term overlap between consecutive RSAs by actor type\n'
        '(municipalities: wave-based, prefectures/MCF: year-based)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'jaccard_by_actor.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'jaccard_by_actor.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: jaccard_by_actor.png/pdf")


def plot_actor_persistence_comparison(
    kommun_transitions: pd.DataFrame,
    yearly_transitions: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Horizontal boxplot: persistence rate distribution per actor type.

    Municipalities use wave-based transitions, prefectures/MCF use year-based.
    Shows actual distribution rather than just means.
    """
    entity_rates = []

    # Process municipality wave-based transitions
    if len(kommun_transitions) > 0:
        CONSECUTIVE_PAIRS = ['W1→W2', 'W2→W3']
        df = kommun_transitions[kommun_transitions['wave_pair'].isin(CONSECUTIVE_PAIRS)].copy()
        present_in_t = df[df['present_from'] == 1].copy()

        for (entity, wave_pair), group in present_in_t.groupby(['entity', 'wave_pair']):
            n_persist = (group['transition'] == 'persist').sum()
            total = len(group)
            entity_rates.append({
                'entity': entity,
                'actor': group['actor'].iloc[0],
                'period': wave_pair,
                'persistence_rate': n_persist / total if total > 0 else 0,
            })

    # Process yearly transitions for prefectures/MCF
    if len(yearly_transitions) > 0:
        present_in_t = yearly_transitions[yearly_transitions['present_from'] == 1].copy()

        for (entity, year_pair), group in present_in_t.groupby(['entity', 'year_pair']):
            n_persist = (group['transition'] == 'persist').sum()
            total = len(group)
            entity_rates.append({
                'entity': entity,
                'actor': group['actor'].iloc[0],
                'period': year_pair,
                'persistence_rate': n_persist / total if total > 0 else 0,
            })

    if not entity_rates:
        return

    rates_df = pd.DataFrame(entity_rates)
    rates_df['actor_en'] = rates_df['actor'].map(translate_actor)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=rates_df, y='actor_en', x='persistence_rate',
        ax=ax, palette='Set1', width=0.5,
        orient='h',
        showfliers=False,
        whis=[0, 100],  # Extend whiskers to full data range
    )
    sns.stripplot(
        data=rates_df, y='actor_en', x='persistence_rate',
        ax=ax, color='black', alpha=0.4, size=5, jitter=True,
        orient='h',
    )

    ax.set_ylabel('Actor type', fontsize=12)
    ax.set_xlabel('Persistence rate', fontsize=12)
    ax.set_title(
        'Persistence rate distribution by actor type\n'
        '(municipalities: wave-based, prefectures/MCF: year-based)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(-0.05, 1.05)

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
        default=Path('./results/01_bow_analysis/persistence'),
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

    # Create subfolders for mention and character persistence
    mention_output = args.output / 'mention_persistence'
    char_output = args.output / 'character_persistence'
    mention_output.mkdir(parents=True, exist_ok=True)
    char_output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RISK PERSISTENCE ANALYSIS")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {args.input}")
    df, term_cols = load_and_prepare(args.input)

    # Filter out pre-2015 data (wave 0) - we only analyze from 2015 onwards
    pre_filter_count = len(df)
    df = df[df['wave'] >= 1].copy()
    print(f"  Filtered to wave >= 1 (2015+): {pre_filter_count} -> {len(df)} documents")

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

    # Compute Jaccard - wave-based for municipalities, year-based for others
    print("\nComputing Jaccard similarity...")

    # Municipalities: wave-based
    kommun_panel = panel[panel['actor'] == 'kommun']
    jaccard_kommun = compute_jaccard(kommun_panel, term_cols)
    print(f"  Municipalities (wave-based): {len(jaccard_kommun)} comparisons")

    # Prefectures and MCF: year-based
    jaccard_yearly_list = []
    for actor in ['lansstyrelse', 'MCF']:
        actor_df = df[df['actor'] == actor]
        if len(actor_df) >= 2:
            actor_jaccard = compute_jaccard_yearly(actor_df, term_cols)
            if len(actor_jaccard) > 0:
                jaccard_yearly_list.append(actor_jaccard)
                print(f"  {translate_actor(actor)} (year-based): {len(actor_jaccard)} comparisons")

    # Combine all Jaccard scores
    jaccard_parts = [jaccard_kommun] + jaccard_yearly_list
    jaccard_df = pd.concat([j for j in jaccard_parts if len(j) > 0], ignore_index=True)
    print(f"  Total: {len(jaccard_df)} entity-pair comparisons")
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
        mention_output / 'persistence_transitions.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: mention_persistence/persistence_transitions.csv")

    persistence_by_term.to_csv(
        mention_output / 'persistence_by_term.csv', encoding='utf-8'
    )
    print(f"  Saved: mention_persistence/persistence_by_term.csv")

    jaccard_df.to_csv(
        mention_output / 'jaccard_scores.csv', index=False, encoding='utf-8'
    )
    print(f"  Saved: mention_persistence/jaccard_scores.csv")

    # Visualisations
    print("\nGenerating visualisations...")

    # Combined heatmap (all actors, wave-based)
    plot_persistence_heatmap(
        transitions, mention_output, min_entities=args.min_entities
    )

    # Municipality wave-based heatmap (W0→W1, W1→W2, W2→W3)
    kommun_transitions = transitions[transitions['actor'] == 'kommun']
    plot_persistence_heatmap(
        kommun_transitions, mention_output,
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
            w1_w3_transitions, mention_output,
            wave_pair_label='W1→W3',
            min_entities=3,
            suffix='_kommun_w1_w3',
        )
        # Save W1→W3 transitions
        w1_w3_transitions.to_csv(
            mention_output / 'persistence_transitions_kommun_w1_w3.csv',
            index=False, encoding='utf-8'
        )
        print(f"  Saved: mention_persistence/persistence_transitions_kommun_w1_w3.csv")
    else:
        print("  No municipalities with both W1 and W3 documents")

    # Year-by-year transitions for prefectures and MCF
    print("\nComputing year-by-year transitions for prefectures and MCF...")
    yearly_transitions_list = []
    for actor in ['lansstyrelse', 'MCF']:
        actor_df = df[df['actor'] == actor]
        if len(actor_df) < 2:
            print(f"  Skipping {actor}: not enough documents")
            continue

        year_trans = compute_year_transitions(actor_df, term_cols)
        if len(year_trans) == 0:
            print(f"  Skipping {actor}: no year transitions computed")
            continue

        yearly_transitions_list.append(year_trans)
        n_entities = year_trans['entity'].nunique()
        n_pairs = year_trans['year_pair'].nunique()
        print(f"  {translate_actor(actor)}: {n_entities} entities, {n_pairs} year-pairs")

        plot_year_persistence_heatmap(
            year_trans, mention_output,
            min_entities=1,
            suffix=f'_{actor}',
        )

        # Save year transitions
        year_trans.to_csv(
            mention_output / f'persistence_transitions_year_{actor}.csv',
            index=False, encoding='utf-8'
        )

    # Combine yearly transitions for comparison plot
    yearly_transitions = pd.concat(yearly_transitions_list, ignore_index=True) if yearly_transitions_list else pd.DataFrame()

    # Dropout and adoption rankings
    plot_dropout_adoption_ranking(transitions, mention_output)

    # Jaccard by actor
    plot_jaccard_by_actor(jaccard_df, mention_output)

    # Actor persistence comparison (municipalities wave-based, others year-based)
    kommun_transitions = transitions[transitions['actor'] == 'kommun']
    plot_actor_persistence_comparison(kommun_transitions, yearly_transitions, mention_output)

    # Character persistence analysis
    print("\n" + "=" * 60)
    print("CHARACTER PERSISTENCE ANALYSIS")
    print("=" * 60)

    char_df, char_term_cols = load_character_matrix(args.input)
    if char_df is not None and len(char_df) > 0:
        # Create subfolders
        wave_output = char_output / 'wave_comparison'
        entity_output = char_output / 'entity_comparison'
        wave_output.mkdir(parents=True, exist_ok=True)
        entity_output.mkdir(parents=True, exist_ok=True)

        # Filter to 2015+
        char_df = char_df[char_df['wave'] >= 1].copy()

        # === WAVE COMPARISON ===
        print("\nComputing wave-level character coverage...")
        char_by_wave = compute_character_by_wave(char_df, char_term_cols)
        print(f"  {len(char_by_wave)} (wave, actor, term) combinations")

        char_deltas = compute_character_deltas(char_by_wave)
        print(f"  {len(char_deltas)} delta records")

        char_by_wave.to_csv(wave_output / 'character_by_wave.csv', index=False, encoding='utf-8')
        char_deltas.to_csv(wave_output / 'character_deltas.csv', index=False, encoding='utf-8')
        print(f"  Saved to wave_comparison/")

        # === ENTITY COMPARISON (compute first for histogram) ===
        print("\nComputing entity-level character persistence...")
        entity_char_persistence = compute_entity_character_persistence(char_df, char_term_cols)
        print(f"  {len(entity_char_persistence)} entity-term transitions")

        agg_char_persistence = aggregate_entity_character_persistence(entity_char_persistence, min_entities=5)
        print(f"  {len(agg_char_persistence)} risks with ≥5 entities")

        entity_char_persistence.to_csv(entity_output / 'character_entity_transitions.csv', index=False, encoding='utf-8')
        agg_char_persistence.to_csv(entity_output / 'character_entity_aggregated.csv', index=False, encoding='utf-8')
        print(f"  Saved to entity_comparison/")

        # === WAVE COMPARISON VISUALIZATIONS ===
        print("\nGenerating wave comparison visualizations...")
        plot_character_trends(char_by_wave, wave_output, top_n=15)
        plot_character_heatmap(char_deltas, wave_output, actor='kommun')
        plot_character_change_histogram(char_deltas, wave_output, entity_persistence=entity_char_persistence)
        plot_top_growing_risks(char_deltas, wave_output)

        # === ENTITY COMPARISON VISUALIZATIONS ===
        print("\nGenerating entity comparison visualizations...")
        plot_entity_character_persistence(agg_char_persistence, entity_output)
        plot_entity_character_histogram(entity_char_persistence, entity_output)
    else:
        print("  Skipping character analysis: no character matrix found")

    # Report
    print("\nGenerating report...")
    generate_report(
        panel, transitions, jaccard_df, persistence_by_term, mention_output
    )

    print(f"\n{'=' * 60}")
    print(f"Mention persistence saved to: {mention_output}")
    print(f"Character persistence saved to: {char_output}")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
