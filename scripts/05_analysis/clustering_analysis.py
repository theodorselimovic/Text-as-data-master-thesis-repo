#!/usr/bin/env python3
"""
Clustering Analysis for Sentence Vectors
=========================================

This script provides multiple clustering approaches for analyzing sentence embeddings:
1. K-Means Clustering - Fast, requires specifying number of clusters
2. Hierarchical Clustering - Creates dendrograms, flexible with cluster numbers
3. DBSCAN - Density-based, finds outliers, no need to specify k
4. Mini-Batch K-Means - Scalable version for large datasets
5. Dimensionality Reduction (PCA, t-SNE, UMAP) for visualization

Each method has different strengths:
- K-Means: Best when you know roughly how many themes exist
- Hierarchical: Good for exploring cluster relationships
- DBSCAN: Excellent for finding natural groupings and outliers
- PCA: Fast dimensionality reduction for visualization
- t-SNE: Better local structure preservation
- UMAP: Balance of speed and quality

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2025-01-05
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Optional: UMAP (install with: pip install umap-learn)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_VECTORS_PATH = 'sentence_vectors.npy'
DEFAULT_INDEX_PATH = 'sentence_vectors_index.csv'
DEFAULT_OUTPUT_DIR = 'clustering_results'

# Clustering parameters
DEFAULT_N_CLUSTERS_KMEANS = 10
DEFAULT_N_CLUSTERS_HIERARCHICAL = 10
DEFAULT_DBSCAN_EPS = 0.5
DEFAULT_DBSCAN_MIN_SAMPLES = 10

# Dimensionality reduction
DEFAULT_N_COMPONENTS_PCA = 50
DEFAULT_N_COMPONENTS_TSNE = 2
DEFAULT_N_COMPONENTS_UMAP = 2

# Visualization
FIGURE_SIZE = (12, 8)
DPI = 150


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """Load sentence vectors and metadata."""
    
    @staticmethod
    def load_vectors_and_metadata(
        vectors_path: Path,
        index_path: Path,
        sample_size: Optional[int] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load sentence vectors and corresponding metadata.
        
        Parameters:
        -----------
        vectors_path : Path
            Path to sentence_vectors.npy
        index_path : Path
            Path to sentence_vectors_index.csv
        sample_size : Optional[int]
            If provided, randomly sample this many sentences
            
        Returns:
        --------
        Tuple[np.ndarray, pd.DataFrame]
            (vectors array, metadata DataFrame)
        """
        logging.info("=" * 80)
        logging.info("LOADING DATA")
        logging.info("=" * 80)
        
        # Load vectors
        if not vectors_path.exists():
            raise FileNotFoundError(f"Vectors file not found: {vectors_path}")
        
        vectors = np.load(vectors_path)
        logging.info(f"Loaded vectors: {vectors.shape}")
        logging.info(f"  Shape: {vectors.shape[0]:,} sentences × {vectors.shape[1]} dimensions")
        logging.info(f"  Memory: {vectors.nbytes / 1024**2:.1f} MB")
        
        # Load metadata
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        metadata = pd.read_csv(index_path)
        logging.info(f"Loaded metadata: {len(metadata):,} rows")
        
        # Verify alignment
        if len(vectors) != len(metadata):
            raise ValueError(
                f"Mismatch: {len(vectors)} vectors but {len(metadata)} metadata rows"
            )
        
        # Sample if requested
        if sample_size is not None and sample_size < len(vectors):
            logging.info(f"\nSampling {sample_size:,} sentences for analysis...")
            indices = np.random.choice(len(vectors), sample_size, replace=False)
            vectors = vectors[indices]
            metadata = metadata.iloc[indices].reset_index(drop=True)
            logging.info(f"  Sample shape: {vectors.shape}")
        
        # Print metadata summary
        logging.info("\nMetadata summary:")
        logging.info(f"  Categories: {metadata['category'].nunique()}")
        logging.info(f"  Municipalities: {metadata['municipality'].nunique()}")
        logging.info(f"  Years: {sorted(metadata['year'].unique())}")
        
        return vectors, metadata


# =============================================================================
# CLUSTERING METHODS
# =============================================================================

class ClusteringAnalyzer:
    """Perform various clustering analyses on sentence vectors."""
    
    def __init__(self, vectors: np.ndarray, metadata: pd.DataFrame):
        """
        Initialize clustering analyzer.
        
        Parameters:
        -----------
        vectors : np.ndarray
            Sentence vectors (n_samples × n_features)
        metadata : pd.DataFrame
            Metadata for each sentence
        """
        self.vectors = vectors
        self.metadata = metadata
        self.n_samples, self.n_features = vectors.shape
        
        # Storage for results
        self.results = {}
        self.reduced_vectors = {}
        
        logging.info("\n" + "=" * 80)
        logging.info("CLUSTERING ANALYZER INITIALIZED")
        logging.info("=" * 80)
        logging.info(f"Samples: {self.n_samples:,}")
        logging.info(f"Features: {self.n_features}")
    
    # -------------------------------------------------------------------------
    # K-MEANS CLUSTERING
    # -------------------------------------------------------------------------
    
    def kmeans_clustering(
        self,
        n_clusters: int = DEFAULT_N_CLUSTERS_KMEANS,
        use_minibatch: bool = False,
        random_state: int = 42
    ) -> Tuple[np.ndarray, float]:
        """
        Perform K-Means clustering.
        
        K-Means is best when:
        - You have a rough idea of how many themes/clusters exist
        - You want fast, scalable clustering
        - Clusters are roughly spherical and similar in size
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        use_minibatch : bool
            Use MiniBatchKMeans for very large datasets
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            (cluster labels, silhouette score)
        """
        logging.info("\n" + "=" * 80)
        logging.info(f"K-MEANS CLUSTERING (k={n_clusters})")
        logging.info("=" * 80)
        
        if use_minibatch:
            logging.info("Using MiniBatchKMeans for large dataset...")
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=1000,
                max_iter=100,
                n_init=3,
                verbose=0
            )
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300,
                verbose=0
            )
        
        # Fit model
        logging.info("Fitting K-Means model...")
        labels = kmeans.fit_predict(self.vectors)
        
        # Calculate metrics
        silhouette = silhouette_score(self.vectors, labels, sample_size=10000)
        calinski = calinski_harabasz_score(self.vectors, labels)
        davies = davies_bouldin_score(self.vectors, labels)
        
        logging.info("\nClustering Results:")
        logging.info(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
        logging.info(f"  Calinski-Harabasz Score: {calinski:.2f} (higher is better)")
        logging.info(f"  Davies-Bouldin Score: {davies:.4f} (lower is better)")
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        logging.info("\nCluster Sizes:")
        for cluster_id, count in zip(unique, counts):
            pct = 100 * count / len(labels)
            logging.info(f"  Cluster {cluster_id}: {count:,} sentences ({pct:.1f}%)")
        
        # Store results
        self.results['kmeans'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'cluster_centers': kmeans.cluster_centers_
        }
        
        return labels, silhouette
    
    # -------------------------------------------------------------------------
    # HIERARCHICAL CLUSTERING
    # -------------------------------------------------------------------------
    
    def hierarchical_clustering(
        self,
        n_clusters: int = DEFAULT_N_CLUSTERS_HIERARCHICAL,
        method: str = 'ward',
        metric: str = 'euclidean',
        max_samples_for_dendrogram: int = 5000
    ) -> np.ndarray:
        """
        Perform Hierarchical clustering.
        
        Hierarchical clustering is best when:
        - You want to explore relationships between clusters
        - You don't know the number of clusters upfront
        - You want to visualize cluster hierarchy via dendrogram
        
        Parameters:
        -----------
        n_clusters : int
            Number of final clusters to extract
        method : str
            Linkage method ('ward', 'average', 'complete', 'single')
        metric : str
            Distance metric
        max_samples_for_dendrogram : int
            Maximum samples to use for dendrogram visualization
            
        Returns:
        --------
        np.ndarray
            Cluster labels
        """
        logging.info("\n" + "=" * 80)
        logging.info(f"HIERARCHICAL CLUSTERING (k={n_clusters}, method={method})")
        logging.info("=" * 80)
        
        # For large datasets, sample for dendrogram
        if self.n_samples > max_samples_for_dendrogram:
            logging.info(f"Sampling {max_samples_for_dendrogram} points for dendrogram...")
            sample_idx = np.random.choice(
                self.n_samples,
                max_samples_for_dendrogram,
                replace=False
            )
            sample_vectors = self.vectors[sample_idx]
        else:
            sample_vectors = self.vectors
        
        # Compute linkage
        logging.info("Computing hierarchical linkage...")
        linkage_matrix = linkage(sample_vectors, method=method, metric=metric)
        
        # Extract clusters from full dataset
        logging.info("Extracting clusters from full dataset...")
        full_linkage = linkage(self.vectors, method=method, metric=metric)
        labels = fcluster(full_linkage, n_clusters, criterion='maxclust') - 1
        
        # Calculate metrics
        silhouette = silhouette_score(self.vectors, labels, sample_size=10000)
        calinski = calinski_harabasz_score(self.vectors, labels)
        davies = davies_bouldin_score(self.vectors, labels)
        
        logging.info("\nClustering Results:")
        logging.info(f"  Silhouette Score: {silhouette:.4f}")
        logging.info(f"  Calinski-Harabasz Score: {calinski:.2f}")
        logging.info(f"  Davies-Bouldin Score: {davies:.4f}")
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        logging.info("\nCluster Sizes:")
        for cluster_id, count in zip(unique, counts):
            pct = 100 * count / len(labels)
            logging.info(f"  Cluster {cluster_id}: {count:,} sentences ({pct:.1f}%)")
        
        # Store results
        self.results['hierarchical'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'method': method,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'linkage_matrix': linkage_matrix,
            'sample_idx': sample_idx if self.n_samples > max_samples_for_dendrogram else None
        }
        
        return labels
    
    # -------------------------------------------------------------------------
    # DBSCAN CLUSTERING
    # -------------------------------------------------------------------------
    
    def dbscan_clustering(
        self,
        eps: float = DEFAULT_DBSCAN_EPS,
        min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
        metric: str = 'euclidean'
    ) -> np.ndarray:
        """
        Perform DBSCAN clustering.
        
        DBSCAN is best when:
        - You don't know the number of clusters
        - You expect outliers/noise in your data
        - Clusters have varying densities and shapes
        - You want to find natural groupings
        
        Parameters:
        -----------
        eps : float
            Maximum distance between two samples for them to be in same neighborhood
        min_samples : int
            Minimum samples in a neighborhood for a point to be a core point
        metric : str
            Distance metric
            
        Returns:
        --------
        np.ndarray
            Cluster labels (-1 indicates noise/outliers)
        """
        logging.info("\n" + "=" * 80)
        logging.info(f"DBSCAN CLUSTERING (eps={eps}, min_samples={min_samples})")
        logging.info("=" * 80)
        
        logging.info("Fitting DBSCAN model...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
        labels = dbscan.fit_predict(self.vectors)
        
        # Count clusters and outliers
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logging.info("\nClustering Results:")
        logging.info(f"  Number of clusters: {n_clusters}")
        logging.info(f"  Number of outliers: {n_noise:,} ({100*n_noise/len(labels):.1f}%)")
        
        if n_clusters > 1:
            # Calculate metrics (excluding noise points)
            mask = labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(
                    self.vectors[mask],
                    labels[mask],
                    sample_size=min(10000, mask.sum())
                )
                logging.info(f"  Silhouette Score (excluding noise): {silhouette:.4f}")
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        logging.info("\nCluster Sizes:")
        for cluster_id, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
            pct = 100 * count / len(labels)
            cluster_name = "Noise/Outliers" if cluster_id == -1 else f"Cluster {cluster_id}"
            logging.info(f"  {cluster_name}: {count:,} sentences ({pct:.1f}%)")
        
        # Store results
        self.results['dbscan'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples
        }
        
        return labels
    
    # -------------------------------------------------------------------------
    # DIMENSIONALITY REDUCTION
    # -------------------------------------------------------------------------
    
    def reduce_dimensions_pca(
        self,
        n_components: int = DEFAULT_N_COMPONENTS_PCA
    ) -> np.ndarray:
        """
        Reduce dimensions using PCA.
        
        PCA is best when:
        - You want fast dimensionality reduction
        - Linear relationships are important
        - You need to preserve global structure
        
        Parameters:
        -----------
        n_components : int
            Number of components to keep
            
        Returns:
        --------
        np.ndarray
            Reduced vectors
        """
        logging.info("\n" + "=" * 80)
        logging.info(f"PCA DIMENSIONALITY REDUCTION (n_components={n_components})")
        logging.info("=" * 80)
        
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(self.vectors)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        logging.info(f"Variance explained: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
        logging.info(f"Reduced shape: {reduced.shape}")
        
        self.reduced_vectors['pca'] = reduced
        self.results['pca'] = {
            'reduced_vectors': reduced,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'n_components': n_components
        }
        
        return reduced
    
    def reduce_dimensions_tsne(
        self,
        n_components: int = DEFAULT_N_COMPONENTS_TSNE,
        perplexity: float = 30.0,
        max_samples: int = 10000
    ) -> np.ndarray:
        """
        Reduce dimensions using t-SNE.
        
        t-SNE is best when:
        - You want to visualize high-dimensional data in 2D/3D
        - Preserving local structure is important
        - You have enough computational resources (can be slow)
        
        Note: t-SNE is stochastic and can be slow for large datasets.
        
        Parameters:
        -----------
        n_components : int
            Number of dimensions (usually 2 or 3 for visualization)
        perplexity : float
            Related to number of nearest neighbors
        max_samples : int
            Maximum samples to use (t-SNE is slow on large datasets)
            
        Returns:
        --------
        np.ndarray
            Reduced vectors
        """
        logging.info("\n" + "=" * 80)
        logging.info(f"t-SNE DIMENSIONALITY REDUCTION (n_components={n_components})")
        logging.info("=" * 80)
        
        # Sample if dataset is too large
        if self.n_samples > max_samples:
            logging.info(f"Sampling {max_samples:,} points for t-SNE (dataset too large)...")
            sample_idx = np.random.choice(self.n_samples, max_samples, replace=False)
            vectors_to_reduce = self.vectors[sample_idx]
        else:
            sample_idx = None
            vectors_to_reduce = self.vectors
        
        logging.info("Running t-SNE (this may take a while)...")
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        reduced = tsne.fit_transform(vectors_to_reduce)
        
        logging.info(f"t-SNE complete. Output shape: {reduced.shape}")
        
        self.reduced_vectors['tsne'] = reduced
        self.results['tsne'] = {
            'reduced_vectors': reduced,
            'n_components': n_components,
            'perplexity': perplexity,
            'sample_idx': sample_idx
        }
        
        return reduced
    
    def reduce_dimensions_umap(
        self,
        n_components: int = DEFAULT_N_COMPONENTS_UMAP,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ) -> Optional[np.ndarray]:
        """
        Reduce dimensions using UMAP.
        
        UMAP is best when:
        - You want both local and global structure preservation
        - You need faster computation than t-SNE
        - You have the umap-learn package installed
        
        Parameters:
        -----------
        n_components : int
            Number of dimensions
        n_neighbors : int
            Number of neighboring points used in local approximations
        min_dist : float
            Minimum distance between points in low-dimensional space
            
        Returns:
        --------
        Optional[np.ndarray]
            Reduced vectors or None if UMAP not available
        """
        if not UMAP_AVAILABLE:
            logging.warning("UMAP not available. Install with: pip install umap-learn")
            return None
        
        logging.info("\n" + "=" * 80)
        logging.info(f"UMAP DIMENSIONALITY REDUCTION (n_components={n_components})")
        logging.info("=" * 80)
        
        logging.info("Running UMAP...")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=True
        )
        reduced = reducer.fit_transform(self.vectors)
        
        logging.info(f"UMAP complete. Output shape: {reduced.shape}")
        
        self.reduced_vectors['umap'] = reduced
        self.results['umap'] = {
            'reduced_vectors': reduced,
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist
        }
        
        return reduced
    
    # -------------------------------------------------------------------------
    # ELBOW METHOD FOR K-MEANS
    # -------------------------------------------------------------------------
    
    def find_optimal_k_elbow(
        self,
        k_range: range = range(2, 21),
        use_minibatch: bool = True
    ) -> Dict[int, float]:
        """
        Find optimal k using elbow method.
        
        Parameters:
        -----------
        k_range : range
            Range of k values to test
        use_minibatch : bool
            Use MiniBatchKMeans for speed
            
        Returns:
        --------
        Dict[int, float]
            Mapping of k to inertia values
        """
        logging.info("\n" + "=" * 80)
        logging.info("ELBOW METHOD FOR OPTIMAL K")
        logging.info("=" * 80)
        
        inertias = {}
        silhouettes = {}
        
        for k in k_range:
            logging.info(f"Testing k={k}...")
            
            if use_minibatch:
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=42,
                    batch_size=1000,
                    n_init=3
                )
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            
            labels = kmeans.fit_predict(self.vectors)
            inertias[k] = kmeans.inertia_
            
            # Calculate silhouette score
            if self.n_samples > 10000:
                silhouette = silhouette_score(
                    self.vectors,
                    labels,
                    sample_size=10000
                )
            else:
                silhouette = silhouette_score(self.vectors, labels)
            
            silhouettes[k] = silhouette
            logging.info(f"  Inertia: {inertias[k]:.2f}, Silhouette: {silhouette:.4f}")
        
        self.results['elbow'] = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouettes': silhouettes
        }
        
        return inertias


# =============================================================================
# VISUALIZATION
# =============================================================================

class ClusterVisualizer:
    """Create visualizations for clustering results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save plots
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = FIGURE_SIZE
        plt.rcParams['figure.dpi'] = DPI
    
    def plot_elbow_curve(
        self,
        k_range: List[int],
        inertias: Dict[int, float],
        silhouettes: Dict[int, float]
    ) -> None:
        """
        Plot elbow curve and silhouette scores.
        
        Parameters:
        -----------
        k_range : List[int]
            Range of k values
        inertias : Dict[int, float]
            Inertia values for each k
        silhouettes : Dict[int, float]
            Silhouette scores for each k
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(k_range, [inertias[k] for k in k_range], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(k_range, [silhouettes[k] for k in k_range], 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score by k', fontsize=14, fontweight='bold')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'elbow_curve.png'
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✓ Saved elbow curve to: {output_path}")
    
    def plot_clusters_2d(
        self,
        vectors_2d: np.ndarray,
        labels: np.ndarray,
        title: str,
        filename: str,
        metadata: Optional[pd.DataFrame] = None,
        color_by_category: bool = False
    ) -> None:
        """
        Plot clusters in 2D space.
        
        Parameters:
        -----------
        vectors_2d : np.ndarray
            2D vectors (n_samples × 2)
        labels : np.ndarray
            Cluster labels
        title : str
            Plot title
        filename : str
            Output filename
        metadata : Optional[pd.DataFrame]
            Metadata for additional coloring
        color_by_category : bool
            Color by original category instead of cluster
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Determine colors
        if color_by_category and metadata is not None:
            # Color by original category
            categories = metadata['category'].values
            unique_categories = sorted(metadata['category'].unique())
            category_to_num = {cat: i for i, cat in enumerate(unique_categories)}
            colors = [category_to_num[cat] for cat in categories]
            
            scatter = ax.scatter(
                vectors_2d[:, 0],
                vectors_2d[:, 1],
                c=colors,
                cmap='tab10',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
            
            # Create legend
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=plt.cm.tab10(category_to_num[cat]/len(unique_categories)),
                                markersize=10, label=cat)
                      for cat in unique_categories]
            ax.legend(handles=handles, title='Category', loc='best')
            
        else:
            # Color by cluster
            scatter = ax.scatter(
                vectors_2d[:, 0],
                vectors_2d[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✓ Saved 2D plot to: {output_path}")
    
    def plot_dendrogram(
        self,
        linkage_matrix: np.ndarray,
        truncate_mode: str = 'lastp',
        p: int = 30
    ) -> None:
        """
        Plot hierarchical clustering dendrogram.
        
        Parameters:
        -----------
        linkage_matrix : np.ndarray
            Linkage matrix from hierarchical clustering
        truncate_mode : str
            Truncation mode for large dendrograms
        p : int
            Number of clusters to show
        """
        plt.figure(figsize=(15, 8))
        
        dendrogram(
            linkage_matrix,
            truncate_mode=truncate_mode,
            p=p,
            color_threshold=0.7 * max(linkage_matrix[:, 2]),
            above_threshold_color='gray'
        )
        
        plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'dendrogram.png'
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✓ Saved dendrogram to: {output_path}")
    
    def plot_cluster_distribution_by_category(
        self,
        metadata: pd.DataFrame,
        labels: np.ndarray,
        method_name: str
    ) -> None:
        """
        Plot how original categories are distributed across clusters.
        
        Parameters:
        -----------
        metadata : pd.DataFrame
            Metadata with category information
        labels : np.ndarray
            Cluster labels
        method_name : str
            Name of clustering method
        """
        # Create DataFrame with clusters and categories
        df = pd.DataFrame({
            'cluster': labels,
            'category': metadata['category'].values
        })
        
        # Create crosstab
        ct = pd.crosstab(df['category'], df['cluster'], normalize='index') * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ct.plot(kind='bar', stacked=False, ax=ax, colormap='tab10')
        
        ax.set_xlabel('Original Category', fontsize=12)
        ax.set_ylabel('Percentage in Cluster (%)', fontsize=12)
        ax.set_title(
            f'{method_name}: Distribution of Categories Across Clusters',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = self.output_dir / f'{method_name.lower()}_category_distribution.png'
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✓ Saved category distribution to: {output_path}")


# =============================================================================
# RESULTS SAVER
# =============================================================================

class ResultsSaver:
    """Save clustering results to disk."""
    
    @staticmethod
    def save_cluster_assignments(
        metadata: pd.DataFrame,
        labels: np.ndarray,
        method_name: str,
        output_dir: Path
    ) -> None:
        """
        Save cluster assignments with metadata.
        
        Parameters:
        -----------
        metadata : pd.DataFrame
            Original metadata
        labels : np.ndarray
            Cluster labels
        method_name : str
            Name of clustering method
        output_dir : Path
            Output directory
        """
        df = metadata.copy()
        df[f'{method_name}_cluster'] = labels
        
        output_path = output_dir / f'{method_name}_assignments.csv'
        df.to_csv(output_path, index=False)
        
        logging.info(f"✓ Saved {method_name} assignments to: {output_path}")
    
    @staticmethod
    def save_summary_report(
        results: Dict,
        output_dir: Path
    ) -> None:
        """
        Save summary report of all clustering results.
        
        Parameters:
        -----------
        results : Dict
            Dictionary of all clustering results
        output_dir : Path
            Output directory
        """
        output_path = output_dir / 'clustering_summary.txt'
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLUSTERING ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for method, result in results.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{method.upper()}\n")
                f.write(f"{'=' * 80}\n")
                
                if method == 'kmeans':
                    f.write(f"Number of clusters: {result['n_clusters']}\n")
                    f.write(f"Silhouette Score: {result['silhouette']:.4f}\n")
                    f.write(f"Calinski-Harabasz Score: {result['calinski_harabasz']:.2f}\n")
                    f.write(f"Davies-Bouldin Score: {result['davies_bouldin']:.4f}\n")
                
                elif method == 'hierarchical':
                    f.write(f"Number of clusters: {result['n_clusters']}\n")
                    f.write(f"Linkage method: {result['method']}\n")
                    f.write(f"Silhouette Score: {result['silhouette']:.4f}\n")
                    f.write(f"Calinski-Harabasz Score: {result['calinski_harabasz']:.2f}\n")
                    f.write(f"Davies-Bouldin Score: {result['davies_bouldin']:.4f}\n")
                
                elif method == 'dbscan':
                    f.write(f"Number of clusters: {result['n_clusters']}\n")
                    f.write(f"Number of outliers: {result['n_noise']}\n")
                    f.write(f"Epsilon: {result['eps']}\n")
                    f.write(f"Min samples: {result['min_samples']}\n")
                
                elif method == 'elbow':
                    f.write("K-means elbow analysis complete\n")
                    f.write(f"K range tested: {result['k_range']}\n")
        
        logging.info(f"✓ Saved summary report to: {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_clustering_pipeline(
    vectors_path: Path,
    index_path: Path,
    output_dir: Path,
    sample_size: Optional[int] = None,
    methods: List[str] = None,
    n_clusters_kmeans: int = DEFAULT_N_CLUSTERS_KMEANS,
    n_clusters_hierarchical: int = DEFAULT_N_CLUSTERS_HIERARCHICAL,
    run_elbow: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Run complete clustering analysis pipeline.
    
    Parameters:
    -----------
    vectors_path : Path
        Path to sentence_vectors.npy
    index_path : Path
        Path to sentence_vectors_index.csv
    output_dir : Path
        Output directory for results
    sample_size : Optional[int]
        Sample size for large datasets
    methods : List[str]
        Methods to run: ['kmeans', 'hierarchical', 'dbscan', 'pca', 'tsne', 'umap']
    n_clusters_kmeans : int
        Number of clusters for k-means
    n_clusters_hierarchical : int
        Number of clusters for hierarchical
    run_elbow : bool
        Run elbow method to find optimal k
    verbose : bool
        Verbose logging
        
    Returns:
    --------
    Dict
        Results dictionary
    """
    setup_logging(verbose)
    
    if methods is None:
        methods = ['kmeans', 'pca', 'tsne']  # Default methods
    
    logging.info("=" * 80)
    logging.info("CLUSTERING ANALYSIS PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Methods: {', '.join(methods)}")
    
    # Load data
    vectors, metadata = DataLoader.load_vectors_and_metadata(
        vectors_path,
        index_path,
        sample_size
    )
    
    # Initialize analyzer
    analyzer = ClusteringAnalyzer(vectors, metadata)
    
    # Run elbow method first if requested
    if run_elbow and 'kmeans' in methods:
        analyzer.find_optimal_k_elbow(k_range=range(2, 21), use_minibatch=True)
    
    # Run clustering methods
    if 'kmeans' in methods:
        analyzer.kmeans_clustering(
            n_clusters=n_clusters_kmeans,
            use_minibatch=len(vectors) > 50000
        )
    
    if 'hierarchical' in methods:
        analyzer.hierarchical_clustering(n_clusters=n_clusters_hierarchical)
    
    if 'dbscan' in methods:
        analyzer.dbscan_clustering()
    
    # Run dimensionality reduction
    if 'pca' in methods:
        analyzer.reduce_dimensions_pca(n_components=50)
        # Also get 2D for visualization
        analyzer.reduce_dimensions_pca(n_components=2)
    
    if 'tsne' in methods:
        analyzer.reduce_dimensions_tsne(n_components=2, max_samples=10000)
    
    if 'umap' in methods:
        analyzer.reduce_dimensions_umap(n_components=2)
    
    # Visualization
    visualizer = ClusterVisualizer(output_dir)
    
    # Plot elbow curve
    if 'elbow' in analyzer.results:
        elbow_results = analyzer.results['elbow']
        visualizer.plot_elbow_curve(
            elbow_results['k_range'],
            elbow_results['inertias'],
            elbow_results['silhouettes']
        )
    
    # Plot 2D projections with clusters
    if 'kmeans' in analyzer.results and 'tsne' in analyzer.reduced_vectors:
        tsne_vectors = analyzer.reduced_vectors['tsne']
        kmeans_labels = analyzer.results['kmeans']['labels']
        
        # Get sample indices if t-SNE was sampled
        sample_idx = analyzer.results['tsne'].get('sample_idx')
        if sample_idx is not None:
            plot_labels = kmeans_labels[sample_idx]
            plot_metadata = metadata.iloc[sample_idx]
        else:
            plot_labels = kmeans_labels
            plot_metadata = metadata
        
        visualizer.plot_clusters_2d(
            tsne_vectors,
            plot_labels,
            'K-Means Clusters (t-SNE Projection)',
            'kmeans_tsne_projection.png',
            plot_metadata,
            color_by_category=False
        )
        
        # Also plot colored by original categories
        visualizer.plot_clusters_2d(
            tsne_vectors,
            plot_labels,
            'Original Categories (t-SNE Projection)',
            'categories_tsne_projection.png',
            plot_metadata,
            color_by_category=True
        )
    
    # Plot dendrogram for hierarchical clustering
    if 'hierarchical' in analyzer.results:
        visualizer.plot_dendrogram(
            analyzer.results['hierarchical']['linkage_matrix']
        )
    
    # Plot category distributions
    if 'kmeans' in analyzer.results:
        visualizer.plot_cluster_distribution_by_category(
            metadata,
            analyzer.results['kmeans']['labels'],
            'K-Means'
        )
    
    # Save results
    saver = ResultsSaver()
    
    if 'kmeans' in analyzer.results:
        saver.save_cluster_assignments(
            metadata,
            analyzer.results['kmeans']['labels'],
            'kmeans',
            output_dir
        )
    
    if 'hierarchical' in analyzer.results:
        saver.save_cluster_assignments(
            metadata,
            analyzer.results['hierarchical']['labels'],
            'hierarchical',
            output_dir
        )
    
    if 'dbscan' in analyzer.results:
        saver.save_cluster_assignments(
            metadata,
            analyzer.results['dbscan']['labels'],
            'dbscan',
            output_dir
        )
    
    saver.save_summary_report(analyzer.results, output_dir)
    
    logging.info("\n" + "=" * 80)
    logging.info("CLUSTERING PIPELINE COMPLETE!")
    logging.info("=" * 80)
    logging.info(f"Results saved to: {output_dir}")
    
    return analyzer.results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Clustering Analysis for Sentence Vectors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default methods (k-means, PCA, t-SNE)
    python clustering_analysis.py \\
        --vectors sentence_vectors.npy \\
        --index sentence_vectors_index.csv
    
    # Run all methods
    python clustering_analysis.py \\
        --vectors sentence_vectors.npy \\
        --index sentence_vectors_index.csv \\
        --methods kmeans hierarchical dbscan pca tsne umap
    
    # Sample large dataset and find optimal k
    python clustering_analysis.py \\
        --vectors sentence_vectors.npy \\
        --index sentence_vectors_index.csv \\
        --sample-size 50000 \\
        --run-elbow
    
    # Custom number of clusters
    python clustering_analysis.py \\
        --vectors sentence_vectors.npy \\
        --index sentence_vectors_index.csv \\
        --n-clusters-kmeans 15 \\
        --n-clusters-hierarchical 15
        """
    )
    
    parser.add_argument(
        '--vectors',
        type=Path,
        required=True,
        help='Path to sentence_vectors.npy'
    )
    
    parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to sentence_vectors_index.csv'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for large datasets (e.g., 50000)'
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['kmeans', 'hierarchical', 'dbscan', 'pca', 'tsne', 'umap'],
        default=['kmeans', 'pca', 'tsne'],
        help='Clustering methods to run (default: kmeans pca tsne)'
    )
    
    parser.add_argument(
        '--n-clusters-kmeans',
        type=int,
        default=DEFAULT_N_CLUSTERS_KMEANS,
        help=f'Number of clusters for k-means (default: {DEFAULT_N_CLUSTERS_KMEANS})'
    )
    
    parser.add_argument(
        '--n-clusters-hierarchical',
        type=int,
        default=DEFAULT_N_CLUSTERS_HIERARCHICAL,
        help=f'Number of clusters for hierarchical (default: {DEFAULT_N_CLUSTERS_HIERARCHICAL})'
    )
    
    parser.add_argument(
        '--run-elbow',
        action='store_true',
        help='Run elbow method to find optimal k'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        run_clustering_pipeline(
            vectors_path=args.vectors,
            index_path=args.index,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            methods=args.methods,
            n_clusters_kmeans=args.n_clusters_kmeans,
            n_clusters_hierarchical=args.n_clusters_hierarchical,
            run_elbow=args.run_elbow,
            verbose=args.verbose
        )
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        logging.exception("Unexpected error during clustering")
        return 1


if __name__ == '__main__':
    sys.exit(main())
