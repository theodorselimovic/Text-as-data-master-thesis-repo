#!/usr/bin/env python3
"""
Prepare Sentence Vectors for Clustering
========================================

This script prepares your sentence vectors for clustering by using the FIXED
data that already exists in your project (created by fix_sentence_ids.py).

Your project already has:
- data/vector/sentence_vectors_metadata_fixed.csv (with global_sentence_id)
- data/vector/sentence_vectors_with_metadata_fixed.parquet (complete data)

This script:
1. Loads the FIXED parquet file (already de-duplicated)
2. Extracts unique sentence vectors (one per global_sentence_id)
3. Saves clustering-ready files

Usage:
    # If running from project root:
    python scripts/06_analysis/fix_vectors_for_clustering.py

    # Or with custom paths:
    python fix_vectors_for_clustering.py \\
        --parquet data/vector/sentence_vectors_with_metadata_fixed.parquet \\
        --output-dir data/vector/clustering

Author: Swedish Risk Analysis Project
Date: 2025-01-05
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
# MAIN PIPELINE
# =============================================================================

def prepare_vectors_for_clustering(
    parquet_path: Path,
    output_dir: Path,
    verbose: bool = False
) -> None:
    """
    Prepare sentence vectors for clustering from FIXED parquet file.
    
    Parameters:
    -----------
    parquet_path : Path
        Path to sentence_vectors_with_metadata_fixed.parquet
    output_dir : Path
        Output directory
    verbose : bool
        Verbose logging
    """
    setup_logging(verbose)
    
    logging.info("=" * 80)
    logging.info("PREPARE SENTENCE VECTORS FOR CLUSTERING")
    logging.info("=" * 80)
    logging.info(f"Input: {parquet_path}")
    logging.info(f"Output: {output_dir}")
    
    # Load fixed data
    logging.info("\n" + "=" * 80)
    logging.info("LOADING FIXED DATA")
    logging.info("=" * 80)
    
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Fixed parquet file not found: {parquet_path}\n"
            f"Make sure you've run fix_sentence_ids.py first!"
        )
    
    df = pd.read_parquet(parquet_path)
    logging.info(f"Loaded {len(df):,} rows")
    logging.info(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    if 'global_sentence_id' not in df.columns:
        raise ValueError(
            "Column 'global_sentence_id' not found!\n"
            "Make sure you're using the _fixed.parquet file."
        )
    
    if 'sentence_vector' not in df.columns:
        raise ValueError("Column 'sentence_vector' not found!")
    
    # Validate data
    logging.info("\n" + "=" * 80)
    logging.info("VALIDATING DATA")
    logging.info("=" * 80)
    
    n_total = len(df)
    n_unique_sentences = df['global_sentence_id'].nunique()
    
    logging.info(f"Total rows: {n_total:,}")
    logging.info(f"Unique sentences: {n_unique_sentences:,}")
    
    if n_total == n_unique_sentences:
        logging.info("✓ No duplicates! Data is already de-duplicated.")
    else:
        logging.warning(f"⚠ Found {n_total - n_unique_sentences:,} duplicate sentence IDs")
        logging.warning("  Will take first occurrence of each unique sentence")
    
    # Check categories per sentence
    cats_per_sent = df.groupby('global_sentence_id')['category'].nunique()
    logging.info("\nCategories per sentence:")
    for n_cats, count in cats_per_sent.value_counts().sort_index().items():
        pct = 100 * count / len(cats_per_sent)
        logging.info(f"  {n_cats} categories: {count:6,} sentences ({pct:5.1f}%)")
    
    # Extract unique sentences
    logging.info("\n" + "=" * 80)
    logging.info("EXTRACTING UNIQUE SENTENCES")
    logging.info("=" * 80)
    
    # Keep first occurrence of each global_sentence_id
    df_unique = df.drop_duplicates(subset='global_sentence_id', keep='first')
    
    logging.info(f"Extracted {len(df_unique):,} unique sentences")
    
    # Extract vectors
    vectors = np.stack(df_unique['sentence_vector'].values)
    logging.info(f"Vector shape: {vectors.shape}")
    
    # Create metadata (without vectors)
    metadata_cols = ['global_sentence_id', 'doc_id', 'municipality', 'year', 
                     'category', 'sentence_id', 'sentence_text', 'word_count']
    available_cols = [col for col in metadata_cols if col in df_unique.columns]
    metadata = df_unique[available_cols].copy()
    
    # Add category information
    if 'category' in df.columns:
        # Create list of all categories for each sentence
        category_map = df.groupby('global_sentence_id')['category'].apply(
            lambda x: ','.join(sorted(set(x)))
        ).to_dict()
        
        metadata['all_categories'] = metadata['global_sentence_id'].map(category_map)
        metadata['n_categories'] = metadata['all_categories'].str.count(',') + 1
    
    # Save results
    logging.info("\n" + "=" * 80)
    logging.info("SAVING RESULTS")
    logging.info("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Unique vectors
    vectors_out = output_dir / 'sentence_vectors_unique.npy'
    np.save(vectors_out, vectors)
    logging.info(f"✓ Saved vectors: {vectors_out}")
    logging.info(f"  Shape: {vectors.shape} ({vectors.shape[0]:,} sentences × {vectors.shape[1]} dims)")
    logging.info(f"  Size: {vectors.nbytes / 1024**2:.1f} MB")
    
    # 2. Unique index
    index_out = output_dir / 'sentence_vectors_unique_index.csv'
    metadata.reset_index(drop=True).to_csv(index_out, index=False)
    logging.info(f"✓ Saved index: {index_out}")
    logging.info(f"  Rows: {len(metadata):,}")
    
    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)
    logging.info(f"\nInput data:")
    logging.info(f"  Total rows: {n_total:,}")
    logging.info(f"  Unique sentences: {n_unique_sentences:,}")
    logging.info(f"\nOutput data:")
    logging.info(f"  Unique vectors: {vectors.shape[0]:,}")
    logging.info(f"  Vector dimensions: {vectors.shape[1]}")
    logging.info(f"\nFiles for clustering:")
    logging.info(f"  --vectors {vectors_out}")
    logging.info(f"  --index {index_out}")
    
    logging.info("\n" + "=" * 80)
    logging.info("✓ COMPLETE!")
    logging.info("=" * 80)


# =============================================================================
# CLI
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Prepare sentence vectors for clustering (use fixed data)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (from project root):
    python scripts/06_analysis/prepare_vectors_for_clustering.py \\
        --parquet data/vector/sentence_vectors_with_metadata_fixed.parquet \\
        --output-dir data/vector/clustering

Example (custom location):
    python prepare_vectors_for_clustering.py \\
        --parquet /path/to/sentence_vectors_with_metadata_fixed.parquet \\
        --output-dir clustering_data

This will create:
    data/vector/clustering/sentence_vectors_unique.npy
    data/vector/clustering/sentence_vectors_unique_index.csv

Then use these for clustering:
    python clustering_analysis.py \\
        --vectors data/vector/clustering/sentence_vectors_unique.npy \\
        --index data/vector/clustering/sentence_vectors_unique_index.csv
        """
    )
    
    parser.add_argument(
        '--parquet',
        type=Path,
        default=Path('data/vector/sentence_vectors_with_metadata_fixed.parquet'),
        help='Path to fixed parquet file (default: data/vector/sentence_vectors_with_metadata_fixed.parquet)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/vector/clustering'),
        help='Output directory (default: data/vector/clustering/)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        prepare_vectors_for_clustering(
            parquet_path=args.parquet,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure you have:", file=sys.stderr)
        print("  1. Run fix_sentence_ids.py to create _fixed.parquet", file=sys.stderr)
        print("  2. Specified correct path to the fixed parquet file", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        logging.exception("Error preparing vectors")
        return 1


if __name__ == '__main__':
    sys.exit(main())
