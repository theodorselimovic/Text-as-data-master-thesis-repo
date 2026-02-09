#!/usr/bin/env python3
"""
Post-Clustering Thematic Analysis
==================================

After clustering your sentence vectors, use this script to:
1. Extract representative sentences from each cluster
2. Identify characteristic keywords/terms per cluster
3. Name/label clusters based on their content
4. Compare cluster themes across years and municipalities

This helps you understand what each cluster actually represents.

Usage:
    python thematic_analysis.py \\
        --cluster-assignments clustering_results/kmeans_assignments.csv \\
        --vectors sentence_vectors.npy \\
        --expanded-terms expanded_terms_lemmatized_complete.csv \\
        --sentences-parquet data/processed/sentences_lemmatized.parquet

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2025-01-05
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_TOP_N_SENTENCES = 10
DEFAULT_TOP_N_TERMS = 20

# =============================================================================
# LOGGING
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
# CLUSTER THEME ANALYZER
# =============================================================================

class ClusterThemeAnalyzer:
    """Extract and analyze themes from clusters."""
    
    def __init__(
        self,
        cluster_assignments: pd.DataFrame,
        vectors: np.ndarray,
        sentences_df: pd.DataFrame = None
    ):
        """
        Initialize theme analyzer.
        
        Parameters:
        -----------
        cluster_assignments : pd.DataFrame
            DataFrame with cluster labels (must have 'kmeans_cluster' or similar column)
        vectors : np.ndarray
            Sentence vectors
        sentences_df : pd.DataFrame, optional
            DataFrame with full sentence text (from sentences_lemmatized.parquet)
        """
        self.assignments = cluster_assignments
        self.vectors = vectors
        self.sentences_df = sentences_df
        
        # Detect cluster column name
        cluster_cols = [col for col in cluster_assignments.columns 
                       if 'cluster' in col.lower()]
        if not cluster_cols:
            raise ValueError("No cluster column found in assignments DataFrame")
        
        self.cluster_col = cluster_cols[0]
        logging.info(f"Using cluster column: {self.cluster_col}")
        
        # Get unique clusters
        self.clusters = sorted(self.assignments[self.cluster_col].unique())
        logging.info(f"Found {len(self.clusters)} clusters")
    
    def get_cluster_center_indices(
        self,
        cluster_id: int,
        n_sentences: int = DEFAULT_TOP_N_SENTENCES
    ) -> np.ndarray:
        """
        Find sentences closest to cluster center.
        
        Parameters:
        -----------
        cluster_id : int
            Cluster to analyze
        n_sentences : int
            Number of sentences to return
            
        Returns:
        --------
        np.ndarray
            Indices of sentences closest to cluster center
        """
        # Get indices for this cluster
        cluster_mask = self.assignments[self.cluster_col] == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            return np.array([])
        
        # Get vectors for this cluster
        cluster_vectors = self.vectors[cluster_indices]
        
        # Compute cluster center
        center = cluster_vectors.mean(axis=0)
        
        # Find closest sentences to center
        similarities = cosine_similarity([center], cluster_vectors)[0]
        top_indices_in_cluster = similarities.argsort()[-n_sentences:][::-1]
        
        # Map back to global indices
        return cluster_indices[top_indices_in_cluster]
    
    def get_cluster_extreme_indices(
        self,
        cluster_id: int,
        n_sentences: int = DEFAULT_TOP_N_SENTENCES
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find most typical and most atypical sentences in cluster.
        
        Parameters:
        -----------
        cluster_id : int
            Cluster to analyze
        n_sentences : int
            Number of sentences to return
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (most_typical_indices, most_atypical_indices)
        """
        cluster_mask = self.assignments[self.cluster_col] == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            return np.array([]), np.array([])
        
        cluster_vectors = self.vectors[cluster_indices]
        center = cluster_vectors.mean(axis=0)
        
        # Calculate distances to center
        similarities = cosine_similarity([center], cluster_vectors)[0]
        
        # Most typical (closest to center)
        typical_indices_in_cluster = similarities.argsort()[-n_sentences:][::-1]
        typical_indices = cluster_indices[typical_indices_in_cluster]
        
        # Most atypical (farthest from center)
        atypical_indices_in_cluster = similarities.argsort()[:n_sentences]
        atypical_indices = cluster_indices[atypical_indices_in_cluster]
        
        return typical_indices, atypical_indices
    
    def extract_characteristic_terms(
        self,
        expanded_terms_df: pd.DataFrame,
        top_n: int = DEFAULT_TOP_N_TERMS
    ) -> Dict[int, pd.DataFrame]:
        """
        Find most characteristic category terms for each cluster.
        
        Uses the expanded terms to see which categories are most
        represented in each cluster.
        
        Parameters:
        -----------
        expanded_terms_df : pd.DataFrame
            DataFrame with expanded terms (from vectoranalysis.py)
        top_n : int
            Number of terms to return per cluster
            
        Returns:
        --------
        Dict[int, pd.DataFrame]
            Mapping from cluster_id to DataFrame of characteristic terms
        """
        logging.info("Extracting characteristic terms for each cluster...")
        
        # Create category → terms lookup
        category_terms = {}
        for category in expanded_terms_df['category'].unique():
            terms = set(
                expanded_terms_df[expanded_terms_df['category'] == category]['lemma']
            )
            category_terms[category] = terms
        
        cluster_characteristics = {}
        
        for cluster_id in self.clusters:
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            cluster_mask = self.assignments[self.cluster_col] == cluster_id
            cluster_assignments = self.assignments[cluster_mask]
            
            # Count category occurrences in this cluster
            category_counts = cluster_assignments['category'].value_counts()
            
            # Calculate enrichment (observed vs expected)
            cluster_size = len(cluster_assignments)
            total_size = len(self.assignments)
            
            enrichment_data = []
            for category in category_counts.index:
                observed = category_counts[category]
                observed_pct = 100 * observed / cluster_size
                
                expected = (self.assignments['category'] == category).sum()
                expected_pct = 100 * expected / total_size
                
                enrichment = observed_pct / expected_pct if expected_pct > 0 else 0
                
                enrichment_data.append({
                    'category': category,
                    'count': observed,
                    'observed_pct': observed_pct,
                    'expected_pct': expected_pct,
                    'enrichment': enrichment
                })
            
            df_enrichment = pd.DataFrame(enrichment_data)
            df_enrichment = df_enrichment.sort_values('enrichment', ascending=False)
            
            cluster_characteristics[cluster_id] = df_enrichment.head(top_n)
        
        return cluster_characteristics
    
    def generate_cluster_summaries(
        self,
        n_sentences: int = 5
    ) -> Dict[int, Dict]:
        """
        Generate comprehensive summaries for each cluster.
        
        Parameters:
        -----------
        n_sentences : int
            Number of example sentences per cluster
            
        Returns:
        --------
        Dict[int, Dict]
            Mapping from cluster_id to summary dictionary
        """
        logging.info("Generating cluster summaries...")
        
        summaries = {}
        
        for cluster_id in self.clusters:
            if cluster_id == -1:
                continue
            
            cluster_mask = self.assignments[self.cluster_col] == cluster_id
            cluster_data = self.assignments[cluster_mask]
            
            # Basic statistics
            cluster_size = len(cluster_data)
            
            # Category distribution
            category_dist = cluster_data['category'].value_counts()
            dominant_category = category_dist.index[0] if len(category_dist) > 0 else None
            
            # Temporal distribution
            if 'year' in cluster_data.columns:
                year_dist = cluster_data['year'].value_counts().sort_index()
                temporal_trend = "Increasing" if year_dist.iloc[-1] > year_dist.iloc[0] else "Decreasing"
            else:
                year_dist = None
                temporal_trend = "Unknown"
            
            # Municipal distribution
            if 'municipality' in cluster_data.columns:
                muni_dist = cluster_data['municipality'].value_counts().head(5)
            else:
                muni_dist = None
            
            # Representative sentences
            center_indices = self.get_cluster_center_indices(cluster_id, n_sentences)
            
            summaries[cluster_id] = {
                'size': cluster_size,
                'size_pct': 100 * cluster_size / len(self.assignments),
                'dominant_category': dominant_category,
                'category_distribution': category_dist,
                'year_distribution': year_dist,
                'temporal_trend': temporal_trend,
                'top_municipalities': muni_dist,
                'representative_indices': center_indices
            }
        
        return summaries


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ThematicReportGenerator:
    """Generate reports from cluster analysis."""
    
    @staticmethod
    def generate_text_report(
        summaries: Dict[int, Dict],
        characteristics: Dict[int, pd.DataFrame],
        assignments: pd.DataFrame,
        sentences_df: pd.DataFrame = None,
        output_path: Path = None
    ) -> str:
        """
        Generate detailed text report.
        
        Parameters:
        -----------
        summaries : Dict
            Cluster summaries from ClusterThemeAnalyzer
        characteristics : Dict
            Characteristic terms from ClusterThemeAnalyzer
        assignments : pd.DataFrame
            Cluster assignments
        sentences_df : pd.DataFrame, optional
            Full sentence data for showing examples
        output_path : Path, optional
            Path to save report
            
        Returns:
        --------
        str
            Report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append("CLUSTER THEMATIC ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall statistics
        n_clusters = len(summaries)
        total_sentences = len(assignments)
        
        lines.append(f"Total Sentences: {total_sentences:,}")
        lines.append(f"Number of Clusters: {n_clusters}")
        lines.append("")
        
        # Individual cluster reports
        for cluster_id in sorted(summaries.keys()):
            summary = summaries[cluster_id]
            
            lines.append("=" * 80)
            lines.append(f"CLUSTER {cluster_id}")
            lines.append("=" * 80)
            lines.append("")
            
            # Size
            lines.append(f"Size: {summary['size']:,} sentences ({summary['size_pct']:.1f}%)")
            lines.append(f"Dominant Category: {summary['dominant_category']}")
            lines.append("")
            
            # Category distribution
            lines.append("Category Distribution:")
            for cat, count in summary['category_distribution'].items():
                pct = 100 * count / summary['size']
                lines.append(f"  {cat:15s}: {count:6,} ({pct:5.1f}%)")
            lines.append("")
            
            # Characteristic categories
            if cluster_id in characteristics:
                lines.append("Category Enrichment (vs corpus average):")
                char_df = characteristics[cluster_id]
                for _, row in char_df.iterrows():
                    lines.append(
                        f"  {row['category']:15s}: "
                        f"{row['enrichment']:4.2f}x enriched "
                        f"({row['observed_pct']:5.1f}% vs {row['expected_pct']:5.1f}% expected)"
                    )
                lines.append("")
            
            # Temporal trend
            if summary['year_distribution'] is not None:
                lines.append(f"Temporal Trend: {summary['temporal_trend']}")
                lines.append("Year Distribution:")
                for year, count in summary['year_distribution'].items():
                    pct = 100 * count / summary['size']
                    lines.append(f"  {year}: {count:6,} ({pct:5.1f}%)")
                lines.append("")
            
            # Top municipalities
            if summary['top_municipalities'] is not None:
                lines.append("Top 5 Municipalities:")
                for muni, count in summary['top_municipalities'].items():
                    pct = 100 * count / summary['size']
                    lines.append(f"  {muni:30s}: {count:6,} ({pct:5.1f}%)")
                lines.append("")
            
            # Example sentences
            if sentences_df is not None and len(summary['representative_indices']) > 0:
                lines.append("Representative Sentences:")
                for i, idx in enumerate(summary['representative_indices'][:5], 1):
                    if idx < len(sentences_df):
                        # Get lemmatized text from sentences_df
                        sentence_row = sentences_df.iloc[idx]
                        text = sentence_row.get('lemmatized_text', 'N/A')
                        doc_id = sentence_row.get('doc_id', 'N/A')
                        lines.append(f"  {i}. [{doc_id}] {text[:200]}...")
                lines.append("")
            
            lines.append("")
        
        # Generate suggested cluster names
        lines.append("=" * 80)
        lines.append("SUGGESTED CLUSTER NAMES")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Based on dominant categories and enrichment:")
        lines.append("")
        
        for cluster_id in sorted(summaries.keys()):
            summary = summaries[cluster_id]
            dominant = summary['dominant_category']
            
            if cluster_id in characteristics:
                char_df = characteristics[cluster_id]
                top_categories = ', '.join(char_df.head(3)['category'].tolist())
                lines.append(
                    f"Cluster {cluster_id:2d}: \"{dominant}\" "
                    f"(also: {top_categories})"
                )
            else:
                lines.append(f"Cluster {cluster_id:2d}: \"{dominant}\"")
        
        lines.append("")
        lines.append("=" * 80)
        
        report_text = '\n'.join(lines)
        
        # Save if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logging.info(f"✓ Saved report to: {output_path}")
        
        return report_text


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_thematic_analysis(
    cluster_assignments_path: Path,
    vectors_path: Path,
    expanded_terms_path: Path,
    sentences_parquet_path: Path = None,
    output_dir: Path = None,
    verbose: bool = False
) -> Dict:
    """
    Run complete thematic analysis pipeline.
    
    Parameters:
    -----------
    cluster_assignments_path : Path
        Path to cluster assignments CSV
    vectors_path : Path
        Path to sentence vectors NPY file
    expanded_terms_path : Path
        Path to expanded terms CSV
    sentences_parquet_path : Path, optional
        Path to full sentences parquet (for showing examples)
    output_dir : Path, optional
        Output directory for reports
    verbose : bool
        Verbose logging
        
    Returns:
    --------
    Dict
        Results dictionary with summaries and characteristics
    """
    setup_logging(verbose)
    
    logging.info("=" * 80)
    logging.info("THEMATIC ANALYSIS PIPELINE")
    logging.info("=" * 80)
    
    # Load data
    logging.info("\n1. Loading data...")
    assignments = pd.read_csv(cluster_assignments_path)
    vectors = np.load(vectors_path)
    expanded_terms = pd.read_csv(expanded_terms_path)
    
    sentences_df = None
    if sentences_parquet_path and sentences_parquet_path.exists():
        sentences_df = pd.read_parquet(sentences_parquet_path)
        logging.info(f"   Loaded full sentences: {len(sentences_df):,} rows")
    
    logging.info(f"   Assignments: {len(assignments):,} rows")
    logging.info(f"   Vectors: {vectors.shape}")
    logging.info(f"   Expanded terms: {len(expanded_terms):,} terms")
    
    # Initialize analyzer
    logging.info("\n2. Initializing analyzer...")
    analyzer = ClusterThemeAnalyzer(assignments, vectors, sentences_df)
    
    # Extract characteristics
    logging.info("\n3. Extracting characteristic terms...")
    characteristics = analyzer.extract_characteristic_terms(expanded_terms)
    
    # Generate summaries
    logging.info("\n4. Generating cluster summaries...")
    summaries = analyzer.generate_cluster_summaries(n_sentences=10)
    
    # Generate report
    logging.info("\n5. Creating report...")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / 'thematic_analysis_report.txt'
    else:
        report_path = None
    
    report = ThematicReportGenerator.generate_text_report(
        summaries,
        characteristics,
        assignments,
        sentences_df,
        report_path
    )
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("CLUSTER THEMES SUMMARY")
    print("=" * 80)
    
    for cluster_id in sorted(summaries.keys()):
        summary = summaries[cluster_id]
        dominant = summary['dominant_category']
        size = summary['size']
        size_pct = summary['size_pct']
        
        if cluster_id in characteristics:
            char_df = characteristics[cluster_id]
            top_cats = ', '.join(char_df.head(2)['category'].tolist())
            print(
                f"Cluster {cluster_id:2d}: {dominant:15s} "
                f"({size:6,} sent, {size_pct:5.1f}%) | Also: {top_cats}"
            )
        else:
            print(
                f"Cluster {cluster_id:2d}: {dominant:15s} "
                f"({size:6,} sent, {size_pct:5.1f}%)"
            )
    
    logging.info("\n" + "=" * 80)
    logging.info("✓ THEMATIC ANALYSIS COMPLETE")
    logging.info("=" * 80)
    
    if report_path:
        logging.info(f"\nFull report saved to: {report_path}")
    
    return {
        'summaries': summaries,
        'characteristics': characteristics,
        'analyzer': analyzer
    }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Thematic Analysis of Clustered Sentences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python thematic_analysis.py \\
        --cluster-assignments clustering_results/kmeans_assignments.csv \\
        --vectors sentence_vectors.npy \\
        --expanded-terms expanded_terms_lemmatized_complete.csv \\
        --sentences-parquet data/processed/sentences_lemmatized.parquet \\
        --output-dir thematic_results
        """
    )
    
    parser.add_argument(
        '--cluster-assignments',
        type=Path,
        required=True,
        help='Path to cluster assignments CSV'
    )
    
    parser.add_argument(
        '--vectors',
        type=Path,
        required=True,
        help='Path to sentence_vectors.npy'
    )
    
    parser.add_argument(
        '--expanded-terms',
        type=Path,
        required=True,
        help='Path to expanded_terms_lemmatized_complete.csv'
    )
    
    parser.add_argument(
        '--sentences-parquet',
        type=Path,
        default=None,
        help='Path to sentences_lemmatized.parquet (optional, for examples)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('thematic_results'),
        help='Output directory for reports'
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
        run_thematic_analysis(
            cluster_assignments_path=args.cluster_assignments,
            vectors_path=args.vectors,
            expanded_terms_path=args.expanded_terms,
            sentences_parquet_path=args.sentences_parquet,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        logging.exception("Error during thematic analysis")
        return 1


if __name__ == '__main__':
    sys.exit(main())
