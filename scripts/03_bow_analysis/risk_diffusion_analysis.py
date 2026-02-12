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
        --input results/term_document_matrix/term_document_matrix.csv \\
        --output results/diffusion/

    python risk_diffusion_analysis.py \\
        --input results/term_document_matrix/term_document_matrix.csv \\
        --output results/diffusion/ \\
        --spike-threshold 0.15 --verbose

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
from matplotlib.ticker import MaxNLocator

# =============================================================================
# CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set1")

METADATA_COLS = ['file', 'actor', 'entity', 'year']

ACTOR_TRANSLATIONS = {
    'kommun': 'Municipality',
    'länsstyrelse': 'Prefecture',
    'MCF': 'MCF',
}

# Key external events for contextual annotation
EXTERNAL_EVENTS = {
    2016: 'MSBFS 2015:5',
    2018: 'Heatwave/fires',
    2020: 'COVID-19',
    2022: 'Ukraine war',
}

# Import term metadata for grouping by category
sys.path.insert(0, str(Path(__file__).parent))
from risk_context_analysis import RISK_DICTIONARY


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
        Columns: entity, actor, term, first_year, is_left_censored.
        is_left_censored = True if the first appearance is in the
        entity's earliest available document.
    """
    records = []

    for entity, group in df.groupby('entity'):
        group = group.sort_values('year')
        actor = group['actor'].iloc[0]
        earliest_year = group['year'].min()

        for term in term_cols:
            # Find first year where count > 0
            present = group[group[term] > 0]
            if len(present) > 0:
                first_year = present['year'].min()
                is_censored = (first_year == earliest_year)
                records.append({
                    'entity': entity,
                    'actor': actor,
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
        Columns: term, gini, n_entities, mean_year, std_year.
    """
    records = []

    for term, group in first_appearances.groupby('term'):
        years = group['first_year'].values.astype(float)
        if len(years) < 2:
            continue

        records.append({
            'term': term,
            'gini': compute_gini(years),
            'n_entities': len(years),
            'mean_year': years.mean(),
            'std_year': years.std(),
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

    for term, group in first_appearances.groupby('term'):
        medians = group.groupby('actor')['first_year'].median()

        row = {'term': term}
        for actor in ['kommun', 'länsstyrelse', 'MCF']:
            col = f'median_year_{actor}'
            row[col] = medians.get(actor, np.nan)
            row[f'n_{actor}'] = len(group[group['actor'] == actor])

        # Compute lags
        if not np.isnan(row.get('median_year_kommun', np.nan)):
            if not np.isnan(row.get('median_year_länsstyrelse', np.nan)):
                row['lag_kommun_vs_lansstyrelse'] = (
                    row['median_year_kommun'] - row['median_year_länsstyrelse']
                )
            if not np.isnan(row.get('median_year_MCF', np.nan)):
                row['lag_kommun_vs_MCF'] = (
                    row['median_year_kommun'] - row['median_year_MCF']
                )

        records.append(row)

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

    # Only actor-specific curves (not 'all')
    actor_curves = adoption_curves[adoption_curves['actor'] != 'all']
    actor_curves = actor_curves[actor_curves['term'].isin(relevant_terms)]

    categories = sorted(set(term_to_cat.values()) - {'unknown'})
    n_cats = len(categories)
    if n_cats == 0:
        return

    n_cols = 2
    n_rows = (n_cats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    actor_styles = {
        'kommun': ('-', 'o'),
        'länsstyrelse': ('--', 's'),
        'MCF': (':', '^'),
    }
    actor_colors = {
        'kommun': '#e41a1c',
        'länsstyrelse': '#377eb8',
        'MCF': '#4daf4a',
    }

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
            for actor in ['kommun', 'länsstyrelse', 'MCF']:
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

    plt.suptitle(
        'Risk term adoption curves by actor type',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'adoption_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'adoption_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: adoption_curves.png/pdf")


def plot_adoption_heatmap(
    first_appearances: pd.DataFrame,
    output_dir: Path,
    min_adopters: int = 3,
) -> None:
    """
    Heatmap: rows = terms, columns = years, cell = new adopters.
    """
    pivot = first_appearances.groupby(
        ['term', 'first_year']
    ).size().reset_index(name='new_adopters')

    # Filter to terms with enough total adopters
    term_totals = pivot.groupby('term')['new_adopters'].sum()
    relevant_terms = term_totals[term_totals >= min_adopters].index
    pivot = pivot[pivot['term'].isin(relevant_terms)]

    if len(pivot) == 0:
        return

    heatmap_data = pivot.pivot_table(
        index='term', columns='first_year', values='new_adopters', fill_value=0
    )

    # Sort by total adopters
    heatmap_data = heatmap_data.loc[
        heatmap_data.sum(axis=1).sort_values(ascending=False).index
    ]

    # Limit to top 40 terms for readability
    heatmap_data = heatmap_data.head(40)

    fig_height = max(8, len(heatmap_data) * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    sns.heatmap(
        heatmap_data, cmap='YlOrRd', linewidths=0.3, ax=ax,
        cbar_kws={'label': 'New adopters'},
    )

    ax.set_title(
        'New adopters per term per year',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Risk term', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'adoption_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'adoption_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: adoption_heatmap.png/pdf")


def plot_gini_chart(gini_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Bar chart of Gini coefficients, ranked.
    Low Gini = synchronous, high Gini = gradual.
    """
    if len(gini_df) == 0:
        return

    # Filter to terms with enough entities
    df = gini_df[gini_df['n_entities'] >= 5].copy()
    df = df.sort_values('gini')

    if len(df) == 0:
        return

    # Color by category
    df['category'] = df['term'].apply(_get_category_for_term)
    cat_colors = dict(zip(
        sorted(df['category'].unique()),
        sns.color_palette("Set1", df['category'].nunique())
    ))

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.25)))

    colors = [cat_colors.get(c, 'gray') for c in df['category']]
    ax.barh(range(len(df)), df['gini'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['term'], fontsize=8)
    ax.set_xlabel('Gini coefficient', fontsize=12)
    ax.set_title(
        'Adoption synchronicity (low Gini = synchronous, high = gradual)',
        fontsize=14, fontweight='bold'
    )
    ax.invert_yaxis()

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cat_colors[c], label=c) for c in sorted(cat_colors.keys())
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'gini_coefficients.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'gini_coefficients.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: gini_coefficients.png/pdf")


def plot_lead_lag(lead_lag_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Scatter: x = MCF/länsstyrelse first-appearance year,
    y = municipality first-appearance year.
    """
    for ref_actor, col_x, col_y in [
        ('länsstyrelse', 'median_year_länsstyrelse', 'median_year_kommun'),
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


# =============================================================================
# REPORT
# =============================================================================

def generate_report(
    first_appearances: pd.DataFrame,
    spikes_df: pd.DataFrame,
    gini_df: pd.DataFrame,
    lead_lag_df: pd.DataFrame,
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
            f"  {row['term']:30s}: Gini={row['gini']:.3f} "
            f"(n={int(row['n_entities'])}, mean year={row['mean_year']:.0f})"
        )

    # Most gradual terms (highest Gini)
    report.append("\nMost gradual terms (highest Gini, ≥5 entities):")
    gradual = gini_df[gini_df['n_entities'] >= 5].tail(15).iloc[::-1]
    for _, row in gradual.iterrows():
        report.append(
            f"  {row['term']:30s}: Gini={row['gini']:.3f} "
            f"(n={int(row['n_entities'])}, mean year={row['mean_year']:.0f})"
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
        default=Path('./results/diffusion'),
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

    # Visualisations
    print("\nGenerating visualisations...")
    plot_adoption_curves(adoption_curves, args.output)
    plot_adoption_heatmap(first_appearances, args.output)
    plot_gini_chart(gini_df, args.output)
    plot_lead_lag(lead_lag_df, args.output)

    # Report
    print("\nGenerating report...")
    generate_report(
        first_appearances, spikes_df, gini_df, lead_lag_df, args.output
    )

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {args.output}")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
