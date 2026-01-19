#!/usr/bin/env python3
"""
Data Diagnostic Script

This script helps you understand what's in your sentence data and what 
steps you need to take before running co-occurrence analysis.

Usage:
    python data_diagnostic.py --input sentence_vectors_with_metadata.parquet
"""

import pandas as pd
import argparse
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description='Diagnose sentence data')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('sentence_vectors_with_metadata.parquet'),
        help='Input parquet file'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("DATA DIAGNOSTIC REPORT")
    print("="*80)
    print(f"File: {args.input}\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(args.input)
    print(f"✓ Loaded {len(df)} rows\n")
    
    # Check columns
    print("="*80)
    print("COLUMNS")
    print("="*80)
    print(f"Available columns: {df.columns.tolist()}\n")
    
    # Check unique sentences
    if 'sentence_id' in df.columns:
        n_unique = df['sentence_id'].nunique()
        print(f"Unique sentences: {n_unique}")
        print(f"Category-sentence pairs: {len(df)}")
        print(f"Average categories per sentence: {len(df)/n_unique:.2f}\n")
    
    # Check categories
    print("="*80)
    print("CATEGORIES IN DATA")
    print("="*80)
    if 'category' in df.columns:
        categories = df['category'].value_counts()
        print(categories)
        print()
        
        # Check which categories are missing
        expected = ['resilience', 'risk', 'complexity', 'efficiency', 'equality', 'agency']
        present = set(categories.index)
        missing = set(expected) - present
        
        if missing:
            print("⚠️  MISSING CATEGORIES:")
            for cat in missing:
                print(f"   - {cat}")
            print()
    
    # Check years
    print("="*80)
    print("TEMPORAL DISTRIBUTION")
    print("="*80)
    if 'year' in df.columns:
        years = df['year'].value_counts().sort_index()
        print(years)
        print(f"\nYear range: {df['year'].min()} - {df['year'].max()}\n")
    
    # Check municipalities
    print("="*80)
    print("MUNICIPALITIES")
    print("="*80)
    if 'municipality' in df.columns:
        n_muni = df['municipality'].nunique()
        print(f"Total municipalities: {n_muni}")
        print("\nTop 10 municipalities by sentences:")
        top_muni = df.groupby('municipality')['sentence_id'].nunique().sort_values(ascending=False).head(10)
        print(top_muni)
        print()
    
    # Sample sentences
    print("="*80)
    print("SAMPLE SENTENCES")
    print("="*80)
    if 'sentence_text' in df.columns:
        for cat in df['category'].unique()[:3]:
            sample = df[df['category'] == cat].iloc[0]
            print(f"\nCategory: {cat}")
            print(f"Text: {sample['sentence_text'][:200]}...")
            if 'target_terms' in df.columns:
                print(f"Target terms: {sample['target_terms']}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if 'category' in df.columns:
        categories = set(df['category'].unique())
        
        if categories == {'risk', 'resilience'}:
            print("\n⚠️  YOU ONLY HAVE 2 CATEGORIES (risk, resilience)")
            print("\nYou need to expand seed terms for:")
            print("  - complexity")
            print("  - efficiency") 
            print("  - equality")
            print("\nSTEPS TO FIX:")
            print("  1. Run vectoranalysis.ipynb again")
            print("  2. Add seed terms for missing categories:")
            print("     complexity: ['komplex', 'svår', 'komplicerad', 'beroende', 'utmaning', 'otydlig']")
            print("     efficiency: ['effektiv', 'koordination', 'effektivitet']")
            print("     equality: ['jämförbar', 'likgildig', 'jämlik', 'likvärdig']")
            print("  3. Expand each category with k=50 neighbors")
            print("  4. Re-run sentencefiltering.ipynb to create new sentence_vectors_with_metadata.parquet")
            print("  5. Then run cooccurrence_analysis.py")
            
        elif len(categories) >= 5:
            print("\n✓ GOOD! You have sufficient categories")
            print(f"  Present: {sorted(categories)}")
            print("\nYou can run co-occurrence analysis:")
            print("  python cooccurrence_analysis.py")
        
        else:
            print(f"\n⚠️  YOU HAVE {len(categories)} CATEGORIES")
            print(f"  Present: {sorted(categories)}")
            missing = {'risk', 'resilience', 'complexity', 'efficiency', 'equality'} - categories
            if missing:
                print(f"  Missing: {sorted(missing)}")
                print("\nYou need to expand seed terms for missing categories (see steps above)")
    
    # Check if agency category exists and needs splitting
    if 'category' in df.columns and 'agency' in df['category'].values:
        print("\n✓ You have 'agency' category - this will be automatically split into:")
        print("  kommun, stat, länsstyrelse, region, näringsliv, civilsamhälle, förening")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
