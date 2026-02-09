#!/usr/bin/env python3
"""
Fix Sentence Data: Remove Duplicate Category Assignments

The current data has multiple rows for each (sentence, category) pair
because it creates one row per TARGET TERM found, not per CATEGORY.

This script fixes that by:
1. Keeping only unique (sentence_id, category) pairs
2. Preserving the first occurrence's metadata
"""

import pandas as pd
from pathlib import Path

def fix_sentence_data(input_file: Path, output_file: Path):
    """
    Remove duplicate (sentence_id, category) pairs.
    
    Parameters:
    -----------
    input_file : Path
        Input CSV with duplicates
    output_file : Path
        Output CSV with deduplicated data
    """
    print("="*80)
    print("FIXING DUPLICATE CATEGORY ASSIGNMENTS")
    print("="*80)
    
    # Load data
    print(f"\nLoading: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"  Original rows: {len(df):,}")
    print(f"  Unique sentences: {df['sentence_id'].nunique():,}")
    
    # Show the problem
    print("\nCategories per sentence (BEFORE):")
    cats_per_sent = df.groupby('sentence_id')['category'].count()
    print(cats_per_sent.value_counts().sort_index().head(10))
    
    # Deduplicate on (sentence_id, category)
    print("\nDeduplicating (sentence_id, category) pairs...")
    df_fixed = df.drop_duplicates(subset=['sentence_id', 'category'])
    
    print(f"  Fixed rows: {len(df_fixed):,}")
    print(f"  Unique sentences: {df_fixed['sentence_id'].nunique():,}")
    
    # Show the fix
    print("\nCategories per sentence (AFTER):")
    cats_per_sent_fixed = df_fixed.groupby('sentence_id')['category'].count()
    print(cats_per_sent_fixed.value_counts().sort_index())
    
    # Category distribution
    print("\nCategory distribution:")
    print(df_fixed['category'].value_counts())
    
    # Save
    print(f"\nSaving to: {output_file}")
    df_fixed.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("FIXED!")
    print("="*80)
    print("\nNow run co-occurrence analysis with the fixed file:")
    print(f"  python scripts/06_analysis/cooccurrence_analysis.py \\")
    print(f"      --input {output_file}")


if __name__ == '__main__':
    input_file = Path('data/vector/sentence_vectors_metadata.csv')
    output_file = Path('data/vector/sentence_vectors_metadata_fixed.csv')
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        print("Update the paths in this script to match your structure.")
        exit(1)
    
    fix_sentence_data(input_file, output_file)
