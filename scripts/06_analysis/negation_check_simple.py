"""
Negation Analysis for Filtered Swedish RSA Sentences

Simple, standalone script that:
1. Loads your filtered sentence data
2. Detects negation words in lemmatized text
3. Provides summary statistics and recommendations
4. Saves results with negation flags

INSTRUCTIONS:
1. Update DATA_FILE path below to point to your actual file
2. Run: python negation_check_simple.py
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

# =============================================================================
# CONFIGURATION - UPDATE THIS PATH!
# =============================================================================

# OPTION 1: If your file is in current directory
DATA_FILE = 'sentence_vectors_metadata.csv'

# OPTION 2: If your file is elsewhere, provide full path
# DATA_FILE = '/full/path/to/sentence_vectors_metadata.csv'

# OPTION 3: If you want to use parquet (requires pyarrow)
# DATA_FILE = 'sentence_vectors_with_metadata.parquet'

# =============================================================================
# SWEDISH NEGATION WORDS (LEMMATIZED FORMS)
# =============================================================================

NEGATION_LEMMAS = [
    'inte',       # not
    'icke',       # non-, un-
    'ej',         # not (formal)
    'aldrig',     # never
    'ingen',      # no/none (includes inget/inga after lemmatization)
    'ingenting',  # nothing
    'varken',     # neither
    'knappt',     # barely/hardly
]

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("NEGATION ANALYSIS - FILTERED SENTENCES")
    print("=" * 80)
    print()
    
    # Step 1: Load data
    print(f"Loading data from: {DATA_FILE}")
    try:
        if DATA_FILE.endswith('.csv'):
            df = pd.read_csv(DATA_FILE, encoding='utf-8')
        elif DATA_FILE.endswith('.parquet'):
            df = pd.read_parquet(DATA_FILE)
        else:
            print("ERROR: File must be .csv or .parquet")
            return
        
        print(f"✓ Loaded {len(df):,} filtered sentences")
        print(f"  Columns: {df.columns.tolist()}")
        print()
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {DATA_FILE}")
        print()
        print("Please update DATA_FILE path at top of script.")
        print("Current working directory files:")
        import os
        print([f for f in os.listdir('.') if f.endswith(('.csv', '.parquet'))])
        return
    
    except Exception as e:
        print(f"ERROR loading file: {e}")
        return
    
    # Check for required column
    if 'sentence_text' not in df.columns:
        print("ERROR: 'sentence_text' column not found")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Step 2: Detect negation
    print("Detecting negation words...")
    negation_pattern = r'\b(' + '|'.join(NEGATION_LEMMAS) + r')\b'
    
    df['has_negation'] = df['sentence_text'].str.contains(
        negation_pattern,
        case=False,
        regex=True,
        na=False
    )
    
    # Extract which negation words appear
    def extract_negations(text):
        if pd.isna(text):
            return []
        return re.findall(negation_pattern, str(text).lower())
    
    df['negation_words'] = df['sentence_text'].apply(extract_negations)
    
    # Step 3: Calculate statistics
    total = len(df)
    negated = df['has_negation'].sum()
    negation_rate = (negated / total * 100) if total > 0 else 0
    
    print(f"✓ Analysis complete")
    print()
    
    # Step 4: Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Total filtered sentences: {total:,}")
    print(f"Sentences with negation:  {negated:,} ({negation_rate:.2f}%)")
    print(f"Sentences without:        {total - negated:,} ({100 - negation_rate:.2f}%)")
    print()
    
    # Negation word frequency
    if negated > 0:
        all_neg_words = [w for words in df['negation_words'] for w in words]
        neg_freq = Counter(all_neg_words)
        
        print("Most common negation words:")
        print("-" * 40)
        for word, count in neg_freq.most_common():
            pct = (count / negated) * 100
            print(f"  {word:12s}: {count:5d} ({pct:5.1f}% of negated)")
        print()
    
    # By category (if available)
    if 'category' in df.columns and negated > 0:
        print("Negation by category:")
        print("-" * 40)
        cat_stats = df.groupby('category')['has_negation'].agg(['sum', 'count', 'mean'])
        cat_stats.columns = ['Negated', 'Total', 'Rate']
        cat_stats['Rate'] = (cat_stats['Rate'] * 100).round(2)
        cat_stats = cat_stats.sort_values('Rate', ascending=False)
        print(cat_stats.to_string())
        print()
    
    # Sample negated sentences
    if negated > 0:
        print("Sample negated sentences:")
        print("-" * 40)
        sample_size = min(10, negated)
        samples = df[df['has_negation']].sample(n=sample_size, random_state=42)
        
        for i, (_, row) in enumerate(samples.iterrows(), 1):
            neg_words = ', '.join(row['negation_words'])
            text = row['sentence_text'][:80]
            if len(row['sentence_text']) > 80:
                text += "..."
            print(f"{i}. [{neg_words}] {text}")
        print()
    
    # Step 5: Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    
    if negation_rate < 5:
        level = "LOW"
        status = "✓"
        action = "Negation is not a major concern. Proceed with analysis as planned."
    elif negation_rate < 10:
        level = "MODERATE"
        status = "○"
        action = "Consider running sensitivity analysis with/without negated sentences."
    else:
        level = "HIGH"
        status = "⚠"
        action = "Strongly recommend filtering negated sentences or using as covariate."
    
    print(f"{status} Negation rate: {negation_rate:.2f}% - {level}")
    print()
    print(f"Recommendation: {action}")
    print()
    
    # Step 6: Save results
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()
    
    output_file = 'sentences_with_negation_flags.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ Saved: {output_file}")
    print(f"  (All sentences with 'has_negation' and 'negation_words' columns)")
    
    summary_file = 'negation_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"NEGATION ANALYSIS SUMMARY\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Total sentences: {total:,}\n")
        f.write(f"Negated sentences: {negated:,} ({negation_rate:.2f}%)\n")
        f.write(f"Assessment: {level}\n")
        f.write(f"\nRecommendation: {action}\n")
    print(f"✓ Saved: {summary_file}")
    print()
    
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the negation rate above")
    print("2. Check 'sentences_with_negation_flags.csv' for flagged sentences")
    print("3. Decide whether to filter/control for negation in your analysis")
    print()


if __name__ == '__main__':
    main()
