#!/usr/bin/env python3
"""
Actor Similarity Analysis (Panel Data Approach)

Measures within-group and between-group similarity for Swedish RSA documents
across actor types (Municipality, Prefecture, MSB) using a three-level
variance decomposition inspired by panel data regression.

Three levels:
    1. Within-entity: Same entity across waves (temporal stability)
    2. Between-entity, within-actor: Different entities, same actor type (homogeneity)
    3. Between-actor: Different actor types (distinctiveness)

Also runs PERMANOVA to test whether actor groups differ significantly.

Input:  category_document_matrix.csv (from term_document_matrix.py)
Output: similarity metrics, heatmaps, bar charts, temporal trends

Usage:
    python actor_similarity_analysis.py \\
        --input results/01_bow_analysis/term_matrices/category_document_matrix.csv \\
        --output results/01_bow_analysis/similarity/

    python actor_similarity_analysis.py \\
        --input results/01_bow_analysis/term_matrices/category_document_matrix.csv \\
        --output results/01_bow_analysis/similarity/ \\
        --waves 1 2 3 --verbose

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
"""

import argparse
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

# =============================================================================
# CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set1")

METADATA_COLS = ['file', 'actor', 'entity', 'year', 'wave', 'total_risk_mentions']

ACTOR_TRANSLATIONS = {
    'kommun': 'Municipality',
    'lansstyrelse': 'Prefecture',
    'länsstyrelse': 'Prefecture',
    'MCF': 'MSB',
}

ACTOR_COLORS = {
    'kommun': '#e41a1c',
    'lansstyrelse': '#377eb8',
    'länsstyrelse': '#377eb8',
    'MCF': '#4daf4a',
}

WAVE_RANGES = {
    1: '2015-2018',
    2: '2019-2022',
    3: '≥2023',
}


def translate_actor(actor: str) -> str:
    """Translate actor names from Swedish to English."""
    return ACTOR_TRANSLATIONS.get(actor, actor)


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_normalize(input_path: Path, waves: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """
    Load category-document matrix, filter to specified waves, normalize to proportions.

    Returns
    -------
    tuple of (pd.DataFrame, list[str])
        (normalized dataframe, risk_column_names)
    """
    df = pd.read_csv(input_path)
    risk_cols = [c for c in df.columns if c.startswith('risk_') and c != 'total_risk_mentions']

    print(f"  Loaded {len(df)} documents, {len(risk_cols)} risk terms")

    # Filter to waves
    df = df[df['wave'].isin(waves)].copy()
    print(f"  After wave filter: {len(df)} documents")

    # Remove zero-mention rows
    row_totals = df[risk_cols].sum(axis=1)
    zero_mask = row_totals == 0
    if zero_mask.any():
        print(f"  Removing {zero_mask.sum()} zero-mention documents")
        df = df[~zero_mask].copy()
        row_totals = df[risk_cols].sum(axis=1)

    # Normalize to proportions
    df[risk_cols] = df[risk_cols].div(row_totals, axis=0)

    print(f"  Final: {len(df)} documents")
    actor_counts = df['actor'].value_counts().to_dict()
    print(f"  By actor: {', '.join(f'{translate_actor(k)}: {v}' for k, v in actor_counts.items())}")

    return df.reset_index(drop=True), risk_cols


# =============================================================================
# SIMILARITY DECOMPOSITION
# =============================================================================

def compute_pairwise_similarity(df: pd.DataFrame, risk_cols: list[str]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Returns
    -------
    np.ndarray
        n x n similarity matrix
    """
    X = df[risk_cols].values
    return cosine_similarity(X)


def decompose_similarity(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
) -> dict:
    """
    Decompose pairwise similarities into three levels:
    1. Within-entity (same entity across waves)
    2. Between-entity, within-actor (different entities, same actor type)
    3. Between-actor (different actor types)

    Returns
    -------
    dict with similarity values for each level
    """
    n = len(df)
    entities = df['entity'].values
    actors = df['actor'].values

    within_entity = []
    between_entity_within_actor = []
    between_actor = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]

            if entities[i] == entities[j]:
                # Same entity (different waves)
                within_entity.append(sim)
            elif actors[i] == actors[j]:
                # Different entity, same actor type
                between_entity_within_actor.append(sim)
            else:
                # Different actor types
                between_actor.append(sim)

    return {
        'within_entity': within_entity,
        'between_entity_within_actor': between_entity_within_actor,
        'between_actor': between_actor,
    }


def compute_similarity_stats(decomposition: dict) -> dict:
    """Compute summary statistics for each similarity level."""
    stats = {}

    for level, values in decomposition.items():
        if len(values) > 0:
            stats[f'{level}_mean'] = np.mean(values)
            stats[f'{level}_std'] = np.std(values)
            stats[f'{level}_n'] = len(values)
        else:
            stats[f'{level}_mean'] = np.nan
            stats[f'{level}_std'] = np.nan
            stats[f'{level}_n'] = 0

    # Compute homogeneity ratio (within-actor / between-actor)
    within_actor_mean = stats.get('between_entity_within_actor_mean', np.nan)
    between_actor_mean = stats.get('between_actor_mean', np.nan)

    if not np.isnan(within_actor_mean) and not np.isnan(between_actor_mean) and between_actor_mean > 0:
        stats['homogeneity_ratio'] = within_actor_mean / between_actor_mean
    else:
        stats['homogeneity_ratio'] = np.nan

    return stats


def compute_actor_pair_similarities(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Compute mean similarity for each actor pair.

    Returns
    -------
    pd.DataFrame with columns: actor1, actor2, mean_similarity, n_pairs
    """
    actors = df['actor'].values
    unique_actors = sorted(df['actor'].unique())

    records = []
    for a1 in unique_actors:
        for a2 in unique_actors:
            mask1 = actors == a1
            mask2 = actors == a2

            # Get similarities between all pairs
            sims = []
            idx1 = np.where(mask1)[0]
            idx2 = np.where(mask2)[0]

            for i in idx1:
                for j in idx2:
                    if i < j:
                        sims.append(sim_matrix[i, j])
                    elif i > j:
                        sims.append(sim_matrix[j, i])

            if len(sims) > 0:
                records.append({
                    'actor1': translate_actor(a1),
                    'actor2': translate_actor(a2),
                    'mean_similarity': np.mean(sims),
                    'n_pairs': len(sims),
                })

    return pd.DataFrame(records)


# =============================================================================
# PERMANOVA
# =============================================================================

def run_permanova(
    df: pd.DataFrame,
    risk_cols: list[str],
    n_permutations: int = 999,
) -> dict:
    """
    Permutational multivariate ANOVA on distance matrix.

    Tests whether actor-type centroids differ significantly.

    Returns
    -------
    dict with pseudo_f, p_value, r_squared
    """
    X = df[risk_cols].values
    groups = df['actor'].values

    # Compute distance matrix (1 - cosine similarity)
    dist_matrix = 1 - cosine_similarity(X)
    np.fill_diagonal(dist_matrix, 0)

    # Compute observed F statistic
    observed_f, ss_between, ss_total = _permanova_f(dist_matrix, groups)

    # Permutation test
    n_greater = 0
    for _ in range(n_permutations):
        perm_groups = np.random.permutation(groups)
        perm_f, _, _ = _permanova_f(dist_matrix, perm_groups)
        if perm_f >= observed_f:
            n_greater += 1

    p_value = (n_greater + 1) / (n_permutations + 1)
    r_squared = ss_between / ss_total if ss_total > 0 else 0

    return {
        'pseudo_f': observed_f,
        'p_value': p_value,
        'r_squared': r_squared,
        'n_permutations': n_permutations,
    }


def _permanova_f(dist_matrix: np.ndarray, groups: np.ndarray) -> tuple:
    """
    Compute pseudo-F statistic for PERMANOVA.

    F = (SS_between / df_between) / (SS_within / df_within)

    Returns (F, SS_between, SS_total)
    """
    n = len(groups)
    unique_groups = np.unique(groups)
    k = len(unique_groups)

    # Total sum of squared distances
    ss_total = 0.5 * np.sum(dist_matrix ** 2) / n

    # Within-group sum of squares
    ss_within = 0
    for g in unique_groups:
        mask = groups == g
        n_g = mask.sum()
        if n_g > 1:
            group_dist = dist_matrix[np.ix_(mask, mask)]
            ss_within += 0.5 * np.sum(group_dist ** 2) / n_g

    ss_between = ss_total - ss_within

    # Degrees of freedom
    df_between = k - 1
    df_within = n - k

    if df_within > 0 and ss_within > 0:
        f_stat = (ss_between / df_between) / (ss_within / df_within)
    else:
        f_stat = 0

    return f_stat, ss_between, ss_total


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_similarity_heatmap(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    output_dir: Path,
    suffix: str = '',
) -> None:
    """Plot pairwise similarity matrix sorted by actor type."""
    # Sort by actor, then entity
    sort_idx = df.sort_values(['actor', 'entity']).index.values
    sorted_sim = sim_matrix[np.ix_(sort_idx, sort_idx)]

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        sorted_sim, cmap='RdYlBu_r', vmin=0, vmax=1,
        square=True, ax=ax,
        cbar_kws={'label': 'Cosine similarity'},
    )

    # Add actor boundaries
    sorted_df = df.iloc[sort_idx]
    actor_changes = np.where(sorted_df['actor'].values[:-1] != sorted_df['actor'].values[1:])[0] + 1

    for idx in actor_changes:
        ax.axhline(y=idx, color='black', linewidth=2)
        ax.axvline(x=idx, color='black', linewidth=2)

    ax.set_title('Pairwise cosine similarity (sorted by actor type)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Document')
    ax.set_ylabel('Document')

    plt.tight_layout()
    fname = f'similarity_heatmap{suffix}'
    plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.png/pdf")


def plot_variance_decomposition(
    stats: dict,
    output_dir: Path,
    suffix: str = '',
) -> None:
    """Bar chart comparing the three similarity levels."""
    levels = ['within_entity', 'between_entity_within_actor', 'between_actor']
    labels = ['Within-entity\n(temporal stability)', 'Between-entity\nwithin-actor\n(homogeneity)', 'Between-actor\n(distinctiveness)']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']

    means = [stats.get(f'{l}_mean', 0) for l in levels]
    stds = [stats.get(f'{l}_std', 0) for l in levels]
    ns = [stats.get(f'{l}_n', 0) for l in levels]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)

    # Add n labels
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'n={n:,}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Mean cosine similarity', fontsize=12)
    ax.set_title('Three-level similarity decomposition', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)

    # Add homogeneity ratio annotation
    ratio = stats.get('homogeneity_ratio', np.nan)
    if not np.isnan(ratio):
        ax.annotate(f'Homogeneity ratio: {ratio:.2f}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fname = f'variance_decomposition{suffix}'
    plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.png/pdf")


def plot_actor_similarity_matrix(
    actor_pairs: pd.DataFrame,
    output_dir: Path,
    suffix: str = '',
) -> None:
    """Heatmap of mean similarity between actor pairs."""
    pivot = actor_pairs.pivot(index='actor1', columns='actor2', values='mean_similarity')

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        pivot, annot=True, fmt='.3f', cmap='RdYlBu_r',
        vmin=0, vmax=1, square=True, ax=ax,
        cbar_kws={'label': 'Mean cosine similarity'},
    )

    ax.set_title('Mean similarity between actor types', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    fname = f'actor_similarity_matrix{suffix}'
    plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.png/pdf")


def plot_similarity_over_time(
    wave_stats: list[dict],
    output_dir: Path,
) -> None:
    """Line plot showing similarity trends across waves."""
    if len(wave_stats) < 2:
        return

    waves = [s['wave'] for s in wave_stats]
    levels = ['within_entity', 'between_entity_within_actor', 'between_actor']
    labels = ['Within-entity', 'Between-entity, within-actor', 'Between-actor']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']

    fig, ax = plt.subplots(figsize=(10, 6))

    for level, label, color in zip(levels, labels, colors):
        means = [s.get(f'{level}_mean', np.nan) for s in wave_stats]
        ax.plot(waves, means, 'o-', label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel('Wave', fontsize=12)
    ax.set_ylabel('Mean cosine similarity', fontsize=12)
    ax.set_title('Similarity trends over time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.set_xticks(waves)
    ax.set_xticklabels([f"Wave {w}\n({WAVE_RANGES.get(w, '')})" for w in waves])
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_over_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'similarity_over_time.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: similarity_over_time.png/pdf")


def plot_permanova_over_time(
    wave_stats: list[dict],
    output_dir: Path,
) -> None:
    """Bar chart of PERMANOVA R² over time."""
    if len(wave_stats) < 2:
        return

    waves = [s['wave'] for s in wave_stats]
    r2s = [s.get('permanova_r_squared', 0) for s in wave_stats]
    p_vals = [s.get('permanova_p_value', 1) for s in wave_stats]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar([f"Wave {w}" for w in waves], r2s, color='#8da0cb', edgecolor='black', alpha=0.8)

    # Add significance markers
    for bar, p in zip(bars, p_vals):
        marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                marker, ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('R² (variance explained by actor type)', fontsize=12)
    ax.set_title('PERMANOVA: Actor effect over time', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(r2s) * 1.3 if max(r2s) > 0 else 0.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'permanova_over_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'permanova_over_time.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: permanova_over_time.png/pdf")


# =============================================================================
# REPORT
# =============================================================================

def generate_report(
    all_stats: list[dict],
    output_dir: Path,
) -> None:
    """Generate comprehensive text report."""
    report = []
    report.append("=" * 70)
    report.append("ACTOR SIMILARITY ANALYSIS — REPORT")
    report.append("=" * 70)
    report.append("\nThree-level variance decomposition (panel data approach):")
    report.append("  1. Within-entity: Same entity across waves (temporal stability)")
    report.append("  2. Between-entity, within-actor: Different entities, same actor (homogeneity)")
    report.append("  3. Between-actor: Different actor types (distinctiveness)")

    for stats in all_stats:
        wave = stats.get('wave', 'pooled')
        report.append(f"\n{'=' * 50}")
        report.append(f"WAVE {wave}" if wave != 'pooled' else "POOLED (ALL WAVES)")
        report.append(f"{'=' * 50}")

        report.append(f"\nDocuments: {stats.get('n_documents', 'N/A')}")

        report.append("\nSimilarity decomposition:")
        for level in ['within_entity', 'between_entity_within_actor', 'between_actor']:
            mean = stats.get(f'{level}_mean', np.nan)
            std = stats.get(f'{level}_std', np.nan)
            n = stats.get(f'{level}_n', 0)
            if not np.isnan(mean):
                report.append(f"  {level}: {mean:.3f} ± {std:.3f} (n={n:,})")

        ratio = stats.get('homogeneity_ratio', np.nan)
        if not np.isnan(ratio):
            report.append(f"\nHomogeneity ratio (within-actor / between-actor): {ratio:.3f}")
            if ratio > 1:
                report.append("  → Actors are more similar internally than externally")
            else:
                report.append("  → Actors are NOT more similar internally than externally")

        # PERMANOVA
        f_stat = stats.get('permanova_pseudo_f', np.nan)
        if not np.isnan(f_stat):
            report.append(f"\nPERMANOVA:")
            report.append(f"  pseudo-F = {f_stat:.2f}")
            report.append(f"  p-value = {stats.get('permanova_p_value', np.nan):.4f}")
            report.append(f"  R² = {stats.get('permanova_r_squared', np.nan):.3f}")

            if stats.get('permanova_p_value', 1) < 0.05:
                report.append("  → Actor types differ SIGNIFICANTLY")
            else:
                report.append("  → Actor types do NOT differ significantly")

    report_text = '\n'.join(report)
    report_path = output_dir / 'similarity_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Saved: similarity_report.txt")
    print(f"\n{report_text}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze within-group and between-group similarity for RSA documents'
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to category_document_matrix.csv'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./results/01_bow_analysis/similarity'),
        help='Output directory'
    )

    parser.add_argument(
        '--waves',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='Wave numbers to analyze (default: 1 2 3)'
    )

    parser.add_argument(
        '--n-permutations',
        type=int,
        default=999,
        help='Number of permutations for PERMANOVA (default: 999)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages'
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACTOR SIMILARITY ANALYSIS")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {args.input}")
    df, risk_cols = load_and_normalize(args.input, args.waves)

    all_stats = []
    wave_stats = []

    # Pooled analysis (all waves combined)
    print(f"\n{'=' * 40}")
    print("POOLED ANALYSIS (ALL WAVES)")
    print(f"{'=' * 40}")

    sim_matrix = compute_pairwise_similarity(df, risk_cols)
    decomposition = decompose_similarity(df, sim_matrix)
    stats = compute_similarity_stats(decomposition)
    stats['n_documents'] = len(df)
    stats['wave'] = 'pooled'

    # PERMANOVA
    print("  Running PERMANOVA...")
    permanova = run_permanova(df, risk_cols, args.n_permutations)
    stats.update({f'permanova_{k}': v for k, v in permanova.items()})
    print(f"  pseudo-F = {permanova['pseudo_f']:.2f}, p = {permanova['p_value']:.4f}, R² = {permanova['r_squared']:.3f}")

    all_stats.append(stats)

    # Actor pair similarities
    actor_pairs = compute_actor_pair_similarities(df, sim_matrix)

    # Visualizations for pooled
    print("  Generating visualizations...")
    plot_similarity_heatmap(df, sim_matrix, args.output)
    plot_variance_decomposition(stats, args.output)
    plot_actor_similarity_matrix(actor_pairs, args.output)

    # Per-wave analysis
    for wave in sorted(args.waves):
        print(f"\n{'=' * 40}")
        print(f"WAVE {wave} ({WAVE_RANGES.get(wave, '')})")
        print(f"{'=' * 40}")

        df_wave = df[df['wave'] == wave].copy()
        if len(df_wave) < 10:
            print(f"  Too few documents ({len(df_wave)}), skipping")
            continue

        print(f"  {len(df_wave)} documents")

        sim_matrix_wave = compute_pairwise_similarity(df_wave, risk_cols)
        decomposition_wave = decompose_similarity(df_wave, sim_matrix_wave)
        stats_wave = compute_similarity_stats(decomposition_wave)
        stats_wave['n_documents'] = len(df_wave)
        stats_wave['wave'] = wave

        # PERMANOVA
        print("  Running PERMANOVA...")
        permanova_wave = run_permanova(df_wave, risk_cols, args.n_permutations)
        stats_wave.update({f'permanova_{k}': v for k, v in permanova_wave.items()})
        print(f"  pseudo-F = {permanova_wave['pseudo_f']:.2f}, p = {permanova_wave['p_value']:.4f}, R² = {permanova_wave['r_squared']:.3f}")

        all_stats.append(stats_wave)
        wave_stats.append(stats_wave)

        # Per-wave visualizations
        plot_variance_decomposition(stats_wave, args.output, suffix=f'_wave{wave}')

    # Temporal trends
    if len(wave_stats) >= 2:
        print(f"\n{'=' * 40}")
        print("TEMPORAL TRENDS")
        print(f"{'=' * 40}")
        plot_similarity_over_time(wave_stats, args.output)
        plot_permanova_over_time(wave_stats, args.output)

    # Save metrics CSV
    metrics_df = pd.DataFrame(all_stats)
    metrics_df.to_csv(args.output / 'similarity_metrics.csv', index=False)
    print(f"\n  Saved: similarity_metrics.csv")

    # Save actor pairs
    actor_pairs.to_csv(args.output / 'actor_pair_similarities.csv', index=False)
    print(f"  Saved: actor_pair_similarities.csv")

    # Report
    print(f"\nGenerating report...")
    generate_report(all_stats, args.output)

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {args.output}")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
