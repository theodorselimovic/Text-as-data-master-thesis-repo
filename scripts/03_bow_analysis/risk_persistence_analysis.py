#!/usr/bin/env python3
"""
Risk Persistence Analysis

Measures whether risk terms persist in RSA documents over time.
For entities (municipalities, prefectures, MCF) with ≥2 documents in
different years, tracks which terms appear, disappear, or are newly
adopted between consecutive documents.

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
    in different years, sorted by entity and year.

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
    # Drop duplicates: if same entity has multiple docs in same year, keep last
    df_sorted = df.sort_values(['entity', 'year'])
    df_dedup = df_sorted.drop_duplicates(subset=['entity', 'year'], keep='last')

    # Keep entities with ≥2 distinct years
    year_counts = df_dedup.groupby('entity')['year'].nunique()
    multi_year = year_counts[year_counts >= 2].index
    panel = df_dedup[df_dedup['entity'].isin(multi_year)].copy()
    panel = panel.sort_values(['entity', 'year']).reset_index(drop=True)

    n_entities = panel['entity'].nunique()
    actor_breakdown = panel.groupby('actor')['entity'].nunique().to_dict()
    translated = {translate_actor(k): v for k, v in actor_breakdown.items()}
    print(f"  Panel: {len(panel)} docs from {n_entities} entities with ≥2 time points")
    print(f"  By actor: {translated}")

    return panel


# =============================================================================
# PERSISTENCE COMPUTATION
# =============================================================================

def compute_transitions(panel: pd.DataFrame, term_cols: list) -> pd.DataFrame:
    """
    For each consecutive document pair within an entity, compute
    per-term transitions: persist, dropout, adopt, stable_absent.

    Returns
    -------
    pd.DataFrame
        One row per (entity, year_from, year_to, term) with columns:
        entity, actor, year_from, year_to, term, present_from, present_to,
        transition (persist/dropout/adopt/stable_absent).
    """
    records = []

    for entity, group in panel.groupby('entity'):
        group = group.sort_values('year')
        actor = group['actor'].iloc[0]
        docs = list(group.iterrows())

        for i in range(len(docs) - 1):
            _, doc_t = docs[i]
            _, doc_t1 = docs[i + 1]

            year_from = doc_t['year']
            year_to = doc_t1['year']

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
                    'year_pair': f"{int(year_from)}→{int(year_to)}",
                    'term': term,
                    'present_from': present_t,
                    'present_to': present_t1,
                    'transition': transition,
                })

    return pd.DataFrame(records)


def compute_jaccard(panel: pd.DataFrame, term_cols: list) -> pd.DataFrame:
    """
    Compute Jaccard similarity between consecutive documents
    for each entity.

    Returns
    -------
    pd.DataFrame
        One row per (entity, year_from, year_to) with Jaccard score.
    """
    records = []

    for entity, group in panel.groupby('entity'):
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

            records.append({
                'entity': entity,
                'actor': actor,
                'year_from': doc_t['year'],
                'year_to': doc_t1['year'],
                'year_pair': f"{int(doc_t['year'])}→{int(doc_t1['year'])}",
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
    Compute persistence rate per term, optionally grouped by actor and year-pair.

    Persistence rate = persist / (persist + dropout), i.e., fraction of
    terms present in T that remain in T+1.

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


def aggregate_by_actor_and_year_pair(
    transitions: pd.DataFrame, min_entities: int = 3
) -> pd.DataFrame:
    """
    Compute persistence rate per term, grouped by actor and year-pair.

    Returns
    -------
    pd.DataFrame
        Rows with: actor, year_pair, term, persistence_rate, n_entities.
    """
    present_in_t = transitions[transitions['present_from'] == 1].copy()

    records = []
    for (actor, year_pair, term), group in present_in_t.groupby(
        ['actor', 'year_pair', 'term']
    ):
        n_persist = (group['transition'] == 'persist').sum()
        n_dropout = (group['transition'] == 'dropout').sum()
        total = n_persist + n_dropout

        # Always include the record, but flag if low N
        flag_low_n = (total < 3)  # Flag if fewer than 3 entities

        records.append({
            'actor': actor,
            'year_pair': year_pair,
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
    Heatmap: rows = terms, columns = year-pair transitions,
    cell colour = persistence rate.
    """
    df = transitions.copy()
    if actor_filter:
        df = df[df['actor'] == actor_filter]

    present_in_t = df[df['present_from'] == 1]

    # Compute persistence rate per term per year-pair
    pivot_data = []
    for (term, year_pair), group in present_in_t.groupby(['term', 'year_pair']):
        n_persist = (group['transition'] == 'persist').sum()
        total = len(group)
        if total >= 3:
            pivot_data.append({
                'term': term,
                'year_pair': year_pair,
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
        index='term', columns='year_pair', values='persistence_rate'
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

    title = 'Term persistence rate by year-pair'
    if actor_filter:
        title += f' ({translate_actor(actor_filter)})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Year-pair transition', fontsize=12)
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
    Grouped bar chart: mean persistence rate per actor type per year-pair.
    """
    present_in_t = transitions[transitions['present_from'] == 1].copy()

    # Compute persistence rate per entity-pair, then average by actor
    entity_rates = []
    for (entity, year_pair), group in present_in_t.groupby(['entity', 'year_pair']):
        n_persist = (group['transition'] == 'persist').sum()
        total = len(group)
        entity_rates.append({
            'entity': entity,
            'actor': group['actor'].iloc[0],
            'year_pair': year_pair,
            'persistence_rate': n_persist / total if total > 0 else 0,
        })

    rates_df = pd.DataFrame(entity_rates)
    rates_df['actor_en'] = rates_df['actor'].map(translate_actor)

    if len(rates_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=rates_df, x='year_pair', y='persistence_rate',
        hue='actor_en', ax=ax, palette='Set1', alpha=0.8,
    )

    ax.set_xlabel('Year-pair transition', fontsize=12)
    ax.set_ylabel('Mean persistence rate', fontsize=12)
    ax.set_title(
        'Mean persistence rate by actor type and period',
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

    # Combined heatmap
    plot_persistence_heatmap(
        transitions, args.output, min_entities=args.min_entities
    )

    # Per-actor heatmaps
    for actor in sorted(panel['actor'].unique()):
        plot_persistence_heatmap(
            transitions, args.output,
            min_entities=max(3, args.min_entities // 3),
            actor_filter=actor,
            suffix=f'_{actor}',
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
