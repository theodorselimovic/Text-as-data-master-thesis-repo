#!/usr/bin/env python3
"""
Risk Clustering Analysis (Per Time Period)

Clusters entities (municipalities, prefectures, MCF) by their risk
profiles, run separately for each collection wave (~2015, ~2019, ~2023).
Uses k-means and hierarchical clustering on proportion-normalised
8-category risk profiles.

Includes actor-type markers in visualisations and cross-period
transition tracking.

Input:  category_document_matrix.csv (from term_document_matrix.py)
Output: elbow plots, dendrograms, PCA scatter, centroid heatmaps,
        cluster assignments, transition matrices

Usage:
    python risk_clustering_analysis.py \\
        --input results/term_document_matrix/category_document_matrix.csv \\
        --output results/municipality_clustering/

    python risk_clustering_analysis.py \\
        --input results/term_document_matrix/category_document_matrix.csv \\
        --output results/municipality_clustering/ \\
        --waves 2015 2019 2023 --window 2 --verbose

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# =============================================================================
# CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set1")

METADATA_COLS = ['file', 'actor', 'entity', 'year', 'total_risk_mentions']

ACTOR_TRANSLATIONS = {
    'kommun': 'Municipality',
    'länsstyrelse': 'Prefecture',
    'MCF': 'MCF',
}

ACTOR_MARKERS = {
    'kommun': 'o',
    'länsstyrelse': 's',
    'MCF': '^',
}


def translate_actor(actor: str) -> str:
    """Translate actor names from Swedish to English."""
    return ACTOR_TRANSLATIONS.get(actor, actor)


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_category_matrix(input_path: Path) -> tuple:
    """
    Load category-document matrix and identify risk columns.

    Returns
    -------
    tuple of (pd.DataFrame, list[str])
        (dataframe, risk_category_column_names)
    """
    df = pd.read_csv(input_path)
    risk_cols = [c for c in df.columns if c.startswith('risk_') and c != 'total_risk_mentions']
    print(f"  Loaded {len(df)} documents, {len(risk_cols)} risk categories")
    print(f"  Categories: {[c.replace('risk_', '') for c in risk_cols]}")
    return df, risk_cols


def filter_to_wave(
    df: pd.DataFrame,
    wave_year: int,
    window: int = 2,
) -> pd.DataFrame:
    """
    Filter to documents within a time window around the wave year.
    If an entity has multiple docs in the window, keep the one
    closest to the wave year.

    Parameters
    ----------
    df : pd.DataFrame
        Full category-document matrix.
    wave_year : int
        Central year of the collection wave.
    window : int
        Years on each side of wave_year to include.

    Returns
    -------
    pd.DataFrame
        Filtered data, one row per entity.
    """
    year_min = wave_year - window
    year_max = wave_year + window

    in_window = df[(df['year'] >= year_min) & (df['year'] <= year_max)].copy()

    if len(in_window) == 0:
        return in_window

    # For each entity, keep the doc closest to wave_year
    in_window['year_dist'] = (in_window['year'] - wave_year).abs()
    in_window = in_window.sort_values(['entity', 'year_dist'])
    result = in_window.drop_duplicates(subset='entity', keep='first')
    result = result.drop(columns='year_dist')

    return result.reset_index(drop=True)


def normalise_to_proportions(
    df: pd.DataFrame, risk_cols: list
) -> pd.DataFrame:
    """
    Normalise risk counts to proportions (row sums to 1).
    Excludes rows with zero total mentions.

    Returns
    -------
    pd.DataFrame
        Copy with normalised risk columns. Rows with zero mentions removed.
    """
    result = df.copy()
    row_totals = result[risk_cols].sum(axis=1)

    # Remove zero-mention rows
    zero_mask = row_totals == 0
    if zero_mask.any():
        n_removed = zero_mask.sum()
        entities_removed = result[zero_mask]['entity'].tolist()
        print(f"    Removed {n_removed} zero-mention entities: {entities_removed}")
        result = result[~zero_mask].copy()
        row_totals = result[risk_cols].sum(axis=1)

    result[risk_cols] = result[risk_cols].div(row_totals, axis=0)

    return result.reset_index(drop=True)


# =============================================================================
# CLUSTERING
# =============================================================================

def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> dict:
    """
    Run k-means for a range of k values and compute evaluation metrics.

    Returns
    -------
    dict with keys:
        'k_range': list of k values
        'inertias': list of inertias
        'silhouettes': list of silhouette scores
        'calinski': list of Calinski-Harabasz scores
        'best_k': k with highest silhouette score
    """
    inertias = []
    silhouettes = []
    calinski = []

    # Cap k_range at n_samples - 1
    max_k = min(max(k_range), len(X) - 1)
    actual_range = range(min(k_range), max_k + 1)

    for k in actual_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        calinski.append(calinski_harabasz_score(X, labels))

    best_k = list(actual_range)[np.argmax(silhouettes)]

    return {
        'k_range': list(actual_range),
        'inertias': inertias,
        'silhouettes': silhouettes,
        'calinski': calinski,
        'best_k': best_k,
    }


def run_clustering(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> dict:
    """
    Run both k-means and hierarchical clustering.

    Returns
    -------
    dict with keys:
        'kmeans_labels': cluster labels from k-means
        'kmeans_centroids': cluster centroids
        'linkage_matrix': scipy linkage matrix
        'hierarchical_labels': labels from cutting dendrogram at k
    """
    # K-means
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans_labels = km.fit_predict(X)

    # Hierarchical (Ward)
    Z = linkage(X, method='ward')
    hier_labels = fcluster(Z, t=k, criterion='maxclust') - 1  # 0-indexed

    return {
        'kmeans_labels': kmeans_labels,
        'kmeans_centroids': km.cluster_centers_,
        'linkage_matrix': Z,
        'hierarchical_labels': hier_labels,
    }


def characterise_clusters(
    wave_df: pd.DataFrame,
    labels: np.ndarray,
    risk_cols: list,
) -> pd.DataFrame:
    """
    Compute cluster centroids and distinctive features.

    Returns
    -------
    pd.DataFrame
        Rows = clusters, columns = risk categories + metadata.
    """
    df = wave_df.copy()
    df['cluster'] = labels

    # Corpus mean
    corpus_mean = df[risk_cols].mean()

    records = []
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        centroid = cluster_data[risk_cols].mean()

        # Z-score vs corpus
        z_scores = (centroid - corpus_mean) / df[risk_cols].std()

        row = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'n_kommun': (cluster_data['actor'] == 'kommun').sum(),
            'n_lansstyrelse': (cluster_data['actor'] == 'länsstyrelse').sum(),
            'n_MCF': (cluster_data['actor'] == 'MCF').sum(),
        }

        for col in risk_cols:
            cat = col.replace('risk_', '')
            row[f'centroid_{cat}'] = centroid[col]
            row[f'zscore_{cat}'] = z_scores[col]

        # Distinctive feature: category with highest z-score
        z_dict = {col.replace('risk_', ''): z_scores[col] for col in risk_cols}
        row['distinctive'] = max(z_dict, key=z_dict.get)

        records.append(row)

    return pd.DataFrame(records)


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_elbow(metrics: dict, wave_year: int, output_dir: Path) -> None:
    """Elbow plot with inertia and silhouette score."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = '#e41a1c'
    color2 = '#377eb8'

    ax1.plot(metrics['k_range'], metrics['inertias'], 'o-', color=color1, linewidth=2)
    ax1.set_xlabel('Number of clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(metrics['k_range'], metrics['silhouettes'], 's--', color=color2, linewidth=2)
    ax2.set_ylabel('Silhouette score', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Mark best k
    best_idx = metrics['k_range'].index(metrics['best_k'])
    ax2.axvline(x=metrics['best_k'], color='gray', linestyle=':', alpha=0.5)
    ax2.annotate(
        f"best k={metrics['best_k']}", (metrics['best_k'], metrics['silhouettes'][best_idx]),
        fontsize=10, fontweight='bold',
        xytext=(10, 10), textcoords='offset points',
    )

    ax1.set_title(f'Cluster evaluation — wave {wave_year}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_dir / f'elbow_{wave_year}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'elbow_{wave_year}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: elbow_{wave_year}.png/pdf")


def plot_dendrogram_fig(
    Z: np.ndarray,
    wave_df: pd.DataFrame,
    k: int,
    wave_year: int,
    output_dir: Path,
) -> None:
    """Dendrogram with actor type in labels."""
    # Build labels: entity (actor)
    labels = [
        f"{row['entity'][:15]} ({translate_actor(row['actor'])[0]})"
        for _, row in wave_df.iterrows()
    ]

    fig_height = max(8, len(labels) * 0.15)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    dendrogram(
        Z, labels=labels, orientation='right',
        color_threshold=Z[-(k - 1), 2] if k > 1 else 0,
        leaf_font_size=7, ax=ax,
    )

    ax.set_title(
        f'Hierarchical clustering — wave {wave_year} (k={k})',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Distance (Ward)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / f'dendrogram_{wave_year}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'dendrogram_{wave_year}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: dendrogram_{wave_year}.png/pdf")


def plot_pca_scatter(
    X: np.ndarray,
    wave_df: pd.DataFrame,
    labels: np.ndarray,
    wave_year: int,
    output_dir: Path,
) -> None:
    """PCA 2D scatter: colour by cluster, marker by actor type."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a qualitative colormap for clusters
    n_clusters = len(set(labels))
    cluster_colors = sns.color_palette("Set2", n_clusters)

    for actor in sorted(wave_df['actor'].unique()):
        mask = wave_df['actor'].values == actor
        marker = ACTOR_MARKERS.get(actor, 'o')
        for cluster_id in range(n_clusters):
            cluster_mask = mask & (labels == cluster_id)
            if cluster_mask.any():
                ax.scatter(
                    X_2d[cluster_mask, 0], X_2d[cluster_mask, 1],
                    c=[cluster_colors[cluster_id]],
                    marker=marker, s=60, alpha=0.7,
                    label=f'C{cluster_id} - {translate_actor(actor)}',
                    edgecolors='white', linewidth=0.5,
                )

    # Label a subset (every Nth for readability)
    n_labels = min(30, len(wave_df))
    label_indices = np.linspace(0, len(wave_df) - 1, n_labels, dtype=int)
    for i in label_indices:
        ax.annotate(
            wave_df.iloc[i]['entity'][:12],
            (X_2d[i, 0], X_2d[i, 1]),
            fontsize=5, alpha=0.6,
            xytext=(3, 3), textcoords='offset points',
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title(
        f'PCA scatter — wave {wave_year} (k={n_clusters})',
        fontsize=14, fontweight='bold'
    )

    # Simplified legend (one entry per cluster + one per actor shape)
    handles, labels_legend = ax.get_legend_handles_labels()
    # Deduplicate
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels_legend):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / f'pca_scatter_{wave_year}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'pca_scatter_{wave_year}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: pca_scatter_{wave_year}.png/pdf")


def plot_centroid_heatmap(
    cluster_info: pd.DataFrame,
    wave_year: int,
    output_dir: Path,
) -> None:
    """Heatmap of cluster centroids (risk profile per cluster)."""
    centroid_cols = [c for c in cluster_info.columns if c.startswith('centroid_')]
    cats = [c.replace('centroid_', '') for c in centroid_cols]

    heatmap_data = cluster_info[centroid_cols].copy()
    heatmap_data.columns = cats
    heatmap_data.index = [
        f"Cluster {int(row['cluster'])} (n={int(row['size'])})"
        for _, row in cluster_info.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(12, max(4, len(heatmap_data) * 1.5)))

    sns.heatmap(
        heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
        linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Proportion of risk mentions'},
    )

    ax.set_title(
        f'Cluster risk profiles — wave {wave_year}',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_dir / f'centroid_heatmap_{wave_year}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'centroid_heatmap_{wave_year}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: centroid_heatmap_{wave_year}.png/pdf")


def plot_actor_distribution(
    cluster_info: pd.DataFrame,
    wave_year: int,
    output_dir: Path,
) -> None:
    """Stacked bar: actor composition per cluster."""
    actor_cols = ['n_kommun', 'n_lansstyrelse', 'n_MCF']
    labels = ['Municipality', 'Prefecture', 'MCF']

    data = cluster_info[actor_cols].copy()
    data.columns = labels
    data.index = [f"Cluster {int(c)}" for c in cluster_info['cluster']]

    fig, ax = plt.subplots(figsize=(8, 5))
    data.plot(kind='bar', stacked=True, ax=ax, alpha=0.8, color=sns.color_palette("Set1", 3))

    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Number of entities', fontsize=12)
    ax.set_title(
        f'Actor composition per cluster — wave {wave_year}',
        fontsize=14, fontweight='bold'
    )
    ax.legend(title='Actor type')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / f'actor_distribution_{wave_year}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'actor_distribution_{wave_year}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: actor_distribution_{wave_year}.png/pdf")


# =============================================================================
# CROSS-PERIOD TRANSITIONS
# =============================================================================

def compute_transitions(
    all_assignments: pd.DataFrame,
    waves: list,
) -> pd.DataFrame:
    """
    Track how entities move between clusters across consecutive waves.

    Returns
    -------
    pd.DataFrame
        Columns: entity, actor, wave_from, wave_to, cluster_from, cluster_to.
    """
    records = []

    for i in range(len(waves) - 1):
        wave_from = waves[i]
        wave_to = waves[i + 1]

        df_from = all_assignments[all_assignments['wave'] == wave_from]
        df_to = all_assignments[all_assignments['wave'] == wave_to]

        # Entities present in both waves
        common = set(df_from['entity']) & set(df_to['entity'])

        for entity in common:
            row_from = df_from[df_from['entity'] == entity].iloc[0]
            row_to = df_to[df_to['entity'] == entity].iloc[0]
            records.append({
                'entity': entity,
                'actor': row_from['actor'],
                'wave_from': wave_from,
                'wave_to': wave_to,
                'cluster_from': row_from['cluster'],
                'cluster_to': row_to['cluster'],
                'changed': row_from['cluster'] != row_to['cluster'],
            })

    return pd.DataFrame(records)


def plot_transition_matrix(
    transitions_df: pd.DataFrame,
    wave_from: int,
    wave_to: int,
    output_dir: Path,
) -> None:
    """Plot a transition matrix heatmap for one wave pair."""
    subset = transitions_df[
        (transitions_df['wave_from'] == wave_from) &
        (transitions_df['wave_to'] == wave_to)
    ]

    if len(subset) == 0:
        return

    pivot = pd.crosstab(
        subset['cluster_from'], subset['cluster_to'],
        margins=True, margins_name='Total'
    )
    pivot.index = [f"From C{i}" if i != 'Total' else 'Total' for i in pivot.index]
    pivot.columns = [f"To C{c}" if c != 'Total' else 'Total' for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot, annot=True, fmt='d', cmap='Blues',
        linewidths=0.5, ax=ax,
    )

    ax.set_title(
        f'Cluster transitions: {wave_from} → {wave_to}',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    fname = f'transition_matrix_{wave_from}_{wave_to}'
    plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.png/pdf")


# =============================================================================
# REPORT
# =============================================================================

def generate_report(
    all_assignments: pd.DataFrame,
    all_cluster_info: dict,
    all_metrics: dict,
    transitions_df: pd.DataFrame,
    waves: list,
    output_dir: Path,
) -> None:
    """Generate comprehensive text report."""
    report = []
    report.append("=" * 70)
    report.append("RISK CLUSTERING ANALYSIS — REPORT")
    report.append("=" * 70)

    for wave in waves:
        metrics = all_metrics.get(wave)
        cluster_info = all_cluster_info.get(wave)

        if metrics is None or cluster_info is None:
            continue

        report.append(f"\n{'=' * 50}")
        report.append(f"WAVE {wave}")
        report.append(f"{'=' * 50}")

        wave_data = all_assignments[all_assignments['wave'] == wave]
        report.append(f"\nEntities: {len(wave_data)}")
        for actor in sorted(wave_data['actor'].unique()):
            n = (wave_data['actor'] == actor).sum()
            report.append(f"  {translate_actor(actor)}: {n}")

        report.append(f"\nOptimal k: {metrics['best_k']}")
        best_idx = metrics['k_range'].index(metrics['best_k'])
        report.append(f"Silhouette score: {metrics['silhouettes'][best_idx]:.3f}")

        report.append(f"\nCluster profiles:")
        for _, row in cluster_info.iterrows():
            report.append(
                f"\n  Cluster {int(row['cluster'])} (n={int(row['size'])}): "
                f"distinctive = {row['distinctive']}"
            )
            report.append(
                f"    Kommun: {int(row['n_kommun'])}, "
                f"Länsstyrelse: {int(row['n_lansstyrelse'])}, "
                f"MCF: {int(row['n_MCF'])}"
            )
            centroid_cols = [c for c in row.index if c.startswith('centroid_')]
            for col in centroid_cols:
                cat = col.replace('centroid_', '')
                report.append(f"    {cat}: {row[col]:.3f}")

    # Transitions
    if len(transitions_df) > 0:
        report.append(f"\n{'=' * 50}")
        report.append("CROSS-PERIOD TRANSITIONS")
        report.append(f"{'=' * 50}")

        for i in range(len(waves) - 1):
            subset = transitions_df[
                (transitions_df['wave_from'] == waves[i]) &
                (transitions_df['wave_to'] == waves[i + 1])
            ]
            if len(subset) == 0:
                continue

            n_changed = subset['changed'].sum()
            n_total = len(subset)
            report.append(
                f"\n{waves[i]} → {waves[i+1]}: "
                f"{n_changed}/{n_total} entities changed cluster "
                f"({n_changed/n_total:.1%})"
            )

    # Save
    report_text = '\n'.join(report)
    report_path = output_dir / 'clustering_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Saved: clustering_report.txt")
    print(f"\n{report_text}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cluster entities by risk profiles per time period'
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
        default=Path('./results/municipality_clustering'),
        help='Output directory'
    )

    parser.add_argument(
        '--waves',
        type=int,
        nargs='+',
        default=[2015, 2019, 2023],
        help='Collection wave years (default: 2015 2019 2023)'
    )

    parser.add_argument(
        '--window',
        type=int,
        default=2,
        help='Years on each side of wave year to include (default: 2)'
    )

    parser.add_argument(
        '--k-range',
        type=int,
        nargs=2,
        default=[2, 10],
        help='Range of k values to try (default: 2 10)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages'
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RISK CLUSTERING ANALYSIS")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {args.input}")
    df, risk_cols = load_category_matrix(args.input)

    k_range = range(args.k_range[0], args.k_range[1] + 1)

    all_assignments = []
    all_cluster_info = {}
    all_metrics = {}

    for wave_year in args.waves:
        print(f"\n{'=' * 40}")
        print(f"WAVE {wave_year} (±{args.window} years)")
        print(f"{'=' * 40}")

        # Filter to wave
        wave_df = filter_to_wave(df, wave_year, window=args.window)
        if len(wave_df) < 5:
            print(f"  Too few entities ({len(wave_df)}), skipping wave {wave_year}")
            continue

        print(f"  {len(wave_df)} entities in wave")
        actor_counts = wave_df['actor'].value_counts().to_dict()
        translated = {translate_actor(k): v for k, v in actor_counts.items()}
        print(f"  By actor: {translated}")

        # Normalise
        print("  Normalising to proportions...")
        wave_norm = normalise_to_proportions(wave_df, risk_cols)

        if len(wave_norm) < 5:
            print(f"  Too few entities after normalisation ({len(wave_norm)}), skipping")
            continue

        X = wave_norm[risk_cols].values

        # Find optimal k
        print(f"  Finding optimal k in {list(k_range)}...")
        metrics = find_optimal_k(X, k_range=k_range)
        best_k = metrics['best_k']
        best_sil = metrics['silhouettes'][metrics['k_range'].index(best_k)]
        print(f"  Best k = {best_k} (silhouette = {best_sil:.3f})")
        all_metrics[wave_year] = metrics

        # Run clustering
        print(f"  Running clustering with k={best_k}...")
        clustering = run_clustering(X, k=best_k)

        # Characterise
        cluster_info = characterise_clusters(
            wave_norm, clustering['kmeans_labels'], risk_cols
        )
        all_cluster_info[wave_year] = cluster_info

        # Save assignments
        assignments = wave_norm[['entity', 'actor', 'year']].copy()
        assignments['cluster'] = clustering['kmeans_labels']
        assignments['wave'] = wave_year
        all_assignments.append(assignments)

        # Visualisations
        print(f"  Generating visualisations...")
        plot_elbow(metrics, wave_year, args.output)
        plot_dendrogram_fig(
            clustering['linkage_matrix'], wave_norm,
            best_k, wave_year, args.output
        )
        plot_pca_scatter(
            X, wave_norm, clustering['kmeans_labels'],
            wave_year, args.output
        )
        plot_centroid_heatmap(cluster_info, wave_year, args.output)
        plot_actor_distribution(cluster_info, wave_year, args.output)

    # Combine assignments
    if all_assignments:
        all_assignments_df = pd.concat(all_assignments, ignore_index=True)
        all_assignments_df.to_csv(
            args.output / 'cluster_assignments.csv', index=False, encoding='utf-8'
        )
        print(f"\n  Saved: cluster_assignments.csv")
    else:
        all_assignments_df = pd.DataFrame()

    # Cross-period transitions
    transitions_df = pd.DataFrame()
    if len(args.waves) >= 2 and len(all_assignments_df) > 0:
        print(f"\nComputing cross-period transitions...")
        transitions_df = compute_transitions(all_assignments_df, args.waves)

        if len(transitions_df) > 0:
            transitions_df.to_csv(
                args.output / 'cluster_transitions.csv', index=False, encoding='utf-8'
            )
            print(f"  Saved: cluster_transitions.csv")

            for i in range(len(args.waves) - 1):
                plot_transition_matrix(
                    transitions_df, args.waves[i], args.waves[i + 1], args.output
                )

    # Report
    print(f"\nGenerating report...")
    generate_report(
        all_assignments_df, all_cluster_info, all_metrics,
        transitions_df, args.waves, args.output
    )

    # Metrics summary
    metrics_rows = []
    for wave, m in all_metrics.items():
        for i, k in enumerate(m['k_range']):
            metrics_rows.append({
                'wave': wave,
                'k': k,
                'inertia': m['inertias'][i],
                'silhouette': m['silhouettes'][i],
                'calinski_harabasz': m['calinski'][i],
            })
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(
            args.output / 'cluster_metrics.csv', index=False
        )
        print(f"  Saved: cluster_metrics.csv")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {args.output}")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
