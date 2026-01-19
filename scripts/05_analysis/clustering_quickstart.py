#!/usr/bin/env python3
"""
Quick Start: Clustering Analysis Examples
==========================================

⚠️ IMPORTANT: Before running this, you MUST run fix_vectors_for_clustering.py!

    python fix_vectors_for_clustering.py \\
        --vectors sentence_vectors.npy \\
        --index sentence_vectors_index.csv \\
        --output-dir clustering_data

This de-duplicates your sentence vectors (removes category-sentence pair duplicates).
Then update the paths below to use the fixed data.

This script provides ready-to-run examples for clustering your sentence vectors.
Uncomment the example you want to run and execute the script.

Requirements:
    pip install numpy pandas scikit-learn matplotlib seaborn scipy
    pip install umap-learn  # Optional, for UMAP

Usage:
    python clustering_quickstart.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from clustering_analysis import (
    DataLoader,
    ClusteringAnalyzer,
    ClusterVisualizer,
    ResultsSaver
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths for your project structure
# After running prepare_vectors_for_clustering.py, use these:
VECTORS_PATH = Path('../data/vector/clustering/sentence_vectors_unique.npy')
INDEX_PATH = Path('../data/vector/clustering/sentence_vectors_unique_index.csv')
OUTPUT_DIR = Path('../results/clustering')

# Note: Paths assume script is run from scripts/06_analysis/
# Adjust if running from project root or elsewhere

# =============================================================================
# EXAMPLE 1: K-MEANS WITH ELBOW METHOD (RECOMMENDED START)
# =============================================================================

def example_1_kmeans_elbow():
    """
    Find optimal number of clusters using elbow method, then cluster.
    
    Best for: Initial exploration of your data
    Time: ~10-15 minutes for 240k sentences
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: K-MEANS WITH ELBOW METHOD")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    vectors, metadata = DataLoader.load_vectors_and_metadata(
        VECTORS_PATH,
        INDEX_PATH,
        sample_size=None  # Use full dataset
    )
    
    # Initialize analyzer
    analyzer = ClusteringAnalyzer(vectors, metadata)
    
    # Run elbow method
    print("\n2. Running elbow method (testing k=2 to k=20)...")
    inertias = analyzer.find_optimal_k_elbow(
        k_range=range(2, 21),
        use_minibatch=True  # Faster for large datasets
    )
    
    # Plot elbow curve
    print("\n3. Creating elbow plot...")
    visualizer = ClusterVisualizer(OUTPUT_DIR)
    elbow_results = analyzer.results['elbow']
    visualizer.plot_elbow_curve(
        elbow_results['k_range'],
        elbow_results['inertias'],
        elbow_results['silhouettes']
    )
    
    # Choose optimal k (look at the plot)
    optimal_k = 15  # Adjust based on elbow plot
    
    print(f"\n4. Clustering with k={optimal_k}...")
    labels, silhouette = analyzer.kmeans_clustering(
        n_clusters=optimal_k,
        use_minibatch=True
    )
    
    # Save results
    print("\n5. Saving results...")
    saver = ResultsSaver()
    saver.save_cluster_assignments(
        metadata,
        labels,
        'kmeans',
        OUTPUT_DIR
    )
    
    # Show distribution across original categories
    print("\n6. Creating category distribution plot...")
    visualizer.plot_cluster_distribution_by_category(
        metadata,
        labels,
        'K-Means'
    )
    
    print("\n✓ Complete! Check clustering_results/ for outputs.")
    print(f"  - elbow_curve.png: Find optimal k")
    print(f"  - kmeans_assignments.csv: Cluster labels for each sentence")
    print(f"  - kmeans_category_distribution.png: How categories map to clusters")
    
    return analyzer, labels


# =============================================================================
# EXAMPLE 2: VISUALIZE WITH T-SNE (SAMPLE FOR SPEED)
# =============================================================================

def example_2_tsne_visualization():
    """
    Create 2D visualization using t-SNE on a sample.
    
    Best for: Understanding cluster structure visually
    Time: ~5-10 minutes for 10k sample
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: T-SNE VISUALIZATION")
    print("=" * 80)
    
    # Load data (sample for speed)
    print("\n1. Loading sample (10,000 sentences)...")
    vectors, metadata = DataLoader.load_vectors_and_metadata(
        VECTORS_PATH,
        INDEX_PATH,
        sample_size=10000
    )
    
    # Initialize analyzer
    analyzer = ClusteringAnalyzer(vectors, metadata)
    
    # First, reduce with PCA (improves t-SNE results)
    print("\n2. Reducing dimensions with PCA (300D → 50D)...")
    pca_reduced = analyzer.reduce_dimensions_pca(n_components=50)
    
    # Cluster the PCA-reduced data
    print("\n3. Clustering with k-means (k=15)...")
    labels, _ = analyzer.kmeans_clustering(n_clusters=15, use_minibatch=False)
    
    # Apply t-SNE for visualization (on full sample)
    print("\n4. Running t-SNE (50D → 2D) - this takes a few minutes...")
    # Create new analyzer with PCA-reduced vectors for t-SNE
    analyzer_pca = ClusteringAnalyzer(pca_reduced, metadata)
    tsne_vectors = analyzer_pca.reduce_dimensions_tsne(
        n_components=2,
        perplexity=30,
        max_samples=10000
    )
    
    # Visualize
    print("\n5. Creating visualizations...")
    visualizer = ClusterVisualizer(OUTPUT_DIR)
    
    # Plot colored by k-means clusters
    visualizer.plot_clusters_2d(
        tsne_vectors,
        labels,
        'K-Means Clusters (t-SNE Projection)',
        'kmeans_tsne_colored_by_cluster.png',
        metadata,
        color_by_category=False
    )
    
    # Plot colored by original categories
    visualizer.plot_clusters_2d(
        tsne_vectors,
        labels,
        'Original Categories (t-SNE Projection)',
        'kmeans_tsne_colored_by_category.png',
        metadata,
        color_by_category=True
    )
    
    print("\n✓ Complete! Check clustering_results/ for:")
    print("  - kmeans_tsne_colored_by_cluster.png: See k-means cluster separation")
    print("  - kmeans_tsne_colored_by_category.png: See how original categories overlap")
    
    return analyzer, tsne_vectors, labels


# =============================================================================
# EXAMPLE 3: HIERARCHICAL CLUSTERING WITH DENDROGRAM
# =============================================================================

def example_3_hierarchical():
    """
    Hierarchical clustering to understand theme relationships.
    
    Best for: Understanding how themes nest within each other
    Time: ~5 minutes for 5k sample
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: HIERARCHICAL CLUSTERING")
    print("=" * 80)
    
    # Load sample (hierarchical is slow on large data)
    print("\n1. Loading sample (5,000 sentences)...")
    vectors, metadata = DataLoader.load_vectors_and_metadata(
        VECTORS_PATH,
        INDEX_PATH,
        sample_size=5000
    )
    
    # Initialize analyzer
    analyzer = ClusteringAnalyzer(vectors, metadata)
    
    # Hierarchical clustering
    print("\n2. Running hierarchical clustering (Ward linkage)...")
    labels = analyzer.hierarchical_clustering(
        n_clusters=12,
        method='ward'
    )
    
    # Create dendrogram
    print("\n3. Creating dendrogram...")
    visualizer = ClusterVisualizer(OUTPUT_DIR)
    visualizer.plot_dendrogram(
        analyzer.results['hierarchical']['linkage_matrix'],
        truncate_mode='lastp',
        p=30
    )
    
    # Save results
    print("\n4. Saving results...")
    saver = ResultsSaver()
    saver.save_cluster_assignments(
        metadata,
        labels,
        'hierarchical',
        OUTPUT_DIR
    )
    
    print("\n✓ Complete! Check clustering_results/ for:")
    print("  - dendrogram.png: See hierarchical relationships between themes")
    print("  - hierarchical_assignments.csv: Cluster labels")
    
    return analyzer, labels


# =============================================================================
# EXAMPLE 4: DBSCAN FOR OUTLIER DETECTION
# =============================================================================

def example_4_dbscan_outliers():
    """
    Use DBSCAN to find outlier sentences and natural clusters.
    
    Best for: Identifying unusual/interesting edge cases
    Time: ~10-15 minutes for 50k sample
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: DBSCAN OUTLIER DETECTION")
    print("=" * 80)
    
    # Load sample
    print("\n1. Loading sample (50,000 sentences)...")
    vectors, metadata = DataLoader.load_vectors_and_metadata(
        VECTORS_PATH,
        INDEX_PATH,
        sample_size=50000
    )
    
    # Initialize analyzer
    analyzer = ClusteringAnalyzer(vectors, metadata)
    
    # Try different DBSCAN parameters
    print("\n2. Testing DBSCAN with different parameters...")
    
    # Conservative (fewer outliers)
    print("\n   Test 1: eps=0.5, min_samples=20 (conservative)")
    labels_1 = analyzer.dbscan_clustering(eps=0.5, min_samples=20)
    
    # Stricter (more outliers)
    print("\n   Test 2: eps=0.3, min_samples=10 (stricter)")
    analyzer_2 = ClusteringAnalyzer(vectors, metadata)
    labels_2 = analyzer_2.dbscan_clustering(eps=0.3, min_samples=10)
    
    # Save results
    print("\n3. Saving outlier labels...")
    saver = ResultsSaver()
    saver.save_cluster_assignments(
        metadata,
        labels_1,
        'dbscan_conservative',
        OUTPUT_DIR
    )
    saver.save_cluster_assignments(
        metadata,
        labels_2,
        'dbscan_strict',
        OUTPUT_DIR
    )
    
    # Find outlier sentences
    outliers_1 = metadata[labels_1 == -1]
    outliers_2 = metadata[labels_2 == -1]
    
    print("\n✓ Complete!")
    print(f"  - Conservative: {len(outliers_1)} outliers ({100*len(outliers_1)/len(metadata):.1f}%)")
    print(f"  - Strict: {len(outliers_2)} outliers ({100*len(outliers_2)/len(metadata):.1f}%)")
    print("\nCheck clustering_results/ for cluster assignments.")
    print("Examine sentences where label == -1 for interesting outliers!")
    
    return analyzer, labels_1, outliers_1


# =============================================================================
# EXAMPLE 5: COMPARE ORIGINAL CATEGORIES VS DISCOVERED CLUSTERS
# =============================================================================

def example_5_category_cluster_analysis():
    """
    Detailed analysis of how original categories map to discovered clusters.
    
    Best for: Understanding if your 6 categories form distinct themes
    Time: ~10 minutes
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: CATEGORY VS CLUSTER ANALYSIS")
    print("=" * 80)
    
    # Load full dataset
    print("\n1. Loading full dataset...")
    vectors, metadata = DataLoader.load_vectors_and_metadata(
        VECTORS_PATH,
        INDEX_PATH
    )
    
    # Cluster
    print("\n2. Clustering with k=15...")
    analyzer = ClusteringAnalyzer(vectors, metadata)
    labels, _ = analyzer.kmeans_clustering(n_clusters=15, use_minibatch=True)
    
    # Create detailed cross-tabulation
    print("\n3. Creating category × cluster cross-tabulation...")
    df = pd.DataFrame({
        'category': metadata['category'],
        'cluster': labels,
        'year': metadata['year'],
        'municipality': metadata['municipality']
    })
    
    # Overall distribution
    print("\n" + "=" * 80)
    print("CATEGORY → CLUSTER DISTRIBUTION")
    print("=" * 80)
    
    ct = pd.crosstab(
        df['category'],
        df['cluster'],
        normalize='index'
    ) * 100
    
    print("\nPercentage of each category in each cluster:")
    print(ct.round(1))
    
    # Find dominant category per cluster
    print("\n" + "=" * 80)
    print("DOMINANT CATEGORY PER CLUSTER")
    print("=" * 80)
    
    ct_abs = pd.crosstab(df['category'], df['cluster'])
    for cluster_id in range(15):
        if cluster_id in ct_abs.columns:
            dominant_cat = ct_abs[cluster_id].idxmax()
            pct = ct[cluster_id][dominant_cat]
            total = ct_abs[cluster_id].sum()
            print(f"Cluster {cluster_id:2d}: {dominant_cat:15s} ({pct:5.1f}%) | Size: {total:6,} sentences")
    
    # Temporal evolution
    print("\n" + "=" * 80)
    print("TEMPORAL EVOLUTION OF CLUSTERS")
    print("=" * 80)
    
    temporal = pd.crosstab(df['year'], df['cluster'], normalize='index') * 100
    print("\nCluster distribution by year (% of year's sentences):")
    print(temporal.round(1))
    
    # Save detailed results
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    ct_abs.to_csv(OUTPUT_DIR / 'category_cluster_crosstab_counts.csv')
    ct.to_csv(OUTPUT_DIR / 'category_cluster_crosstab_percentages.csv')
    temporal.to_csv(OUTPUT_DIR / 'temporal_cluster_distribution.csv')
    df.to_csv(OUTPUT_DIR / 'detailed_cluster_assignments.csv', index=False)
    
    print("\n✓ Complete! Saved detailed results to clustering_results/")
    
    return df, ct


# =============================================================================
# EXAMPLE 6: COMPLETE WORKFLOW
# =============================================================================

def example_6_complete_workflow():
    """
    Complete analysis workflow combining multiple methods.
    
    Best for: Comprehensive exploration of your data
    Time: ~30 minutes
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: COMPLETE WORKFLOW")
    print("=" * 80)
    
    # Stage 1: Find optimal k
    print("\n" + "=" * 80)
    print("STAGE 1: FIND OPTIMAL K")
    print("=" * 80)
    
    vectors, metadata = DataLoader.load_vectors_and_metadata(
        VECTORS_PATH,
        INDEX_PATH
    )
    
    analyzer = ClusteringAnalyzer(vectors, metadata)
    analyzer.find_optimal_k_elbow(range(5, 26), use_minibatch=True)
    
    visualizer = ClusterVisualizer(OUTPUT_DIR)
    elbow_results = analyzer.results['elbow']
    visualizer.plot_elbow_curve(
        elbow_results['k_range'],
        elbow_results['inertias'],
        elbow_results['silhouettes']
    )
    
    # Stage 2: Cluster full dataset
    print("\n" + "=" * 80)
    print("STAGE 2: CLUSTER FULL DATASET")
    print("=" * 80)
    
    optimal_k = 15  # Adjust based on elbow
    labels, _ = analyzer.kmeans_clustering(n_clusters=optimal_k, use_minibatch=True)
    
    # Stage 3: Visualize sample
    print("\n" + "=" * 80)
    print("STAGE 3: VISUALIZE SAMPLE")
    print("=" * 80)
    
    sample_size = 8000
    sample_idx = np.random.choice(len(vectors), sample_size, replace=False)
    vectors_sample = vectors[sample_idx]
    metadata_sample = metadata.iloc[sample_idx].reset_index(drop=True)
    labels_sample = labels[sample_idx]
    
    analyzer_sample = ClusteringAnalyzer(vectors_sample, metadata_sample)
    pca_reduced = analyzer_sample.reduce_dimensions_pca(n_components=50)
    
    analyzer_pca = ClusteringAnalyzer(pca_reduced, metadata_sample)
    tsne_vectors = analyzer_pca.reduce_dimensions_tsne(n_components=2, max_samples=8000)
    
    visualizer.plot_clusters_2d(
        tsne_vectors,
        labels_sample,
        'K-Means Clusters (t-SNE)',
        'complete_tsne_clusters.png',
        metadata_sample,
        color_by_category=False
    )
    
    visualizer.plot_clusters_2d(
        tsne_vectors,
        labels_sample,
        'Original Categories (t-SNE)',
        'complete_tsne_categories.png',
        metadata_sample,
        color_by_category=True
    )
    
    # Stage 4: Hierarchical on sample
    print("\n" + "=" * 80)
    print("STAGE 4: HIERARCHICAL CLUSTERING")
    print("=" * 80)
    
    hier_sample_size = 5000
    hier_idx = np.random.choice(len(vectors), hier_sample_size, replace=False)
    vectors_hier = vectors[hier_idx]
    metadata_hier = metadata.iloc[hier_idx].reset_index(drop=True)
    
    analyzer_hier = ClusteringAnalyzer(vectors_hier, metadata_hier)
    labels_hier = analyzer_hier.hierarchical_clustering(n_clusters=optimal_k)
    
    visualizer.plot_dendrogram(
        analyzer_hier.results['hierarchical']['linkage_matrix']
    )
    
    # Stage 5: Save all results
    print("\n" + "=" * 80)
    print("STAGE 5: SAVE RESULTS")
    print("=" * 80)
    
    saver = ResultsSaver()
    saver.save_cluster_assignments(metadata, labels, 'kmeans_complete', OUTPUT_DIR)
    saver.save_summary_report(analyzer.results, OUTPUT_DIR)
    
    visualizer.plot_cluster_distribution_by_category(
        metadata,
        labels,
        'K-Means-Complete'
    )
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE WORKFLOW FINISHED!")
    print("=" * 80)
    print("\nResults in clustering_results/:")
    print("  - elbow_curve.png")
    print("  - complete_tsne_clusters.png")
    print("  - complete_tsne_categories.png")
    print("  - dendrogram.png")
    print("  - kmeans_complete_assignments.csv")
    print("  - clustering_summary.txt")
    
    return analyzer


# =============================================================================
# MAIN: RUN YOUR CHOSEN EXAMPLE
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("CLUSTERING ANALYSIS QUICK START")
    print("=" * 80)
    print("\nUncomment the example you want to run in this script.")
    print("\nAvailable examples:")
    print("  1. K-Means with Elbow Method (recommended first step)")
    print("  2. t-SNE Visualization")
    print("  3. Hierarchical Clustering")
    print("  4. DBSCAN Outlier Detection")
    print("  5. Category vs Cluster Analysis")
    print("  6. Complete Workflow (all methods)")
    
    # =========================================================================
    # UNCOMMENT THE EXAMPLE YOU WANT TO RUN:
    # =========================================================================
    
    # Example 1: Start here! Find optimal k and cluster
    # analyzer, labels = example_1_kmeans_elbow()
    
    # Example 2: Visualize clusters in 2D
    # analyzer, tsne_vectors, labels = example_2_tsne_visualization()
    
    # Example 3: Understand theme hierarchy
    # analyzer, labels = example_3_hierarchical()
    
    # Example 4: Find outliers
    # analyzer, labels, outliers = example_4_dbscan_outliers()
    
    # Example 5: Compare categories to clusters
    # df, crosstab = example_5_category_cluster_analysis()
    
    # Example 6: Complete analysis (takes ~30 minutes)
    # analyzer = example_6_complete_workflow()
    
    # =========================================================================
    
    print("\n✓ Done! Uncomment an example above to run an analysis.")
