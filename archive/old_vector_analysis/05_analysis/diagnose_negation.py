"""
Diagnostic Script: Why isn't "inte" being detected?

This script checks:
1. Whether "inte" appears in your sentence_text
2. Whether the regex pattern is working correctly
3. What's actually in the sentence_text column
"""

import pandas as pd
import re

# =============================================================================
# CONFIGURATION - UPDATE THIS PATH!
# =============================================================================

DATA_FILE = 'sentence_vectors_metadata.csv'

# =============================================================================
# DIAGNOSTIC CHECKS
# =============================================================================

print("=" * 80)
print("DIAGNOSTIC: Why isn't 'inte' being detected?")
print("=" * 80)
print()

# Load data
print(f"Loading: {DATA_FILE}")
try:
    if DATA_FILE.endswith('.csv'):
        df = pd.read_csv(DATA_FILE, encoding='utf-8')
    else:
        df = pd.read_parquet(DATA_FILE)
    print(f"✓ Loaded {len(df):,} sentences")
    print()
except Exception as e:
    print(f"ERROR: {e}")
    exit()

# Check 1: Does sentence_text column exist and have data?
print("CHECK 1: Sentence text column")
print("-" * 80)
print(f"Column 'sentence_text' exists: {'sentence_text' in df.columns}")
if 'sentence_text' in df.columns:
    print(f"Non-null sentences: {df['sentence_text'].notna().sum():,}")
    print(f"Sample sentence types: {df['sentence_text'].dtype}")
    print()
    
    # Show first 5 sentences
    print("First 5 sentences:")
    for i, text in enumerate(df['sentence_text'].head(5), 1):
        print(f"{i}. {text[:100]}{'...' if len(str(text)) > 100 else ''}")
    print()
else:
    print("ERROR: No 'sentence_text' column found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

# Check 2: Search for "inte" manually
print("CHECK 2: Manual search for 'inte'")
print("-" * 80)

# Simple string search (case-insensitive)
contains_inte_simple = df['sentence_text'].str.contains('inte', case=False, na=False)
count_inte_simple = contains_inte_simple.sum()
print(f"Simple search (any 'inte'): {count_inte_simple:,} sentences")

# Word boundary search
contains_inte_word = df['sentence_text'].str.contains(r'\binte\b', case=False, regex=True, na=False)
count_inte_word = contains_inte_word.sum()
print(f"Word boundary search (\\binte\\b): {count_inte_word:,} sentences")

if count_inte_simple > 0:
    print()
    print("Sample sentences containing 'inte' (first 5):")
    samples = df[contains_inte_simple].head(5)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        text = str(row['sentence_text'])
        # Highlight 'inte' in the text
        highlighted = re.sub(r'(inte)', r'>>>>\1<<<<', text, flags=re.IGNORECASE)
        print(f"{i}. {highlighted[:120]}{'...' if len(highlighted) > 120 else ''}")
    print()
else:
    print("⚠ WARNING: No sentences contain 'inte' at all!")
    print()

# Check 3: Test the exact regex pattern from the script
print("CHECK 3: Testing the exact negation pattern")
print("-" * 80)

NEGATION_LEMMAS = ['inte', 'icke', 'ej', 'aldrig', 'ingen', 'ingenting', 'varken', 'knappt']
negation_pattern = r'\b(' + '|'.join(NEGATION_LEMMAS) + r')\b'

print(f"Pattern: {negation_pattern}")
print()

# Apply the pattern
df['has_negation'] = df['sentence_text'].str.contains(
    negation_pattern,
    case=False,
    regex=True,
    na=False
)

total_negated = df['has_negation'].sum()
print(f"Total sentences matching pattern: {total_negated:,}")
print()

# Extract what was matched
def extract_negations(text):
    if pd.isna(text):
        return []
    matches = re.findall(negation_pattern, str(text).lower())
    return matches

df['negation_words'] = df['sentence_text'].apply(extract_negations)

# Count each negation word
from collections import Counter
all_negations = [word for words in df['negation_words'] for word in words]
neg_counts = Counter(all_negations)

print("Negation words found:")
for word, count in neg_counts.most_common():
    pct = (count / total_negated * 100) if total_negated > 0 else 0
    print(f"  {word:12s}: {count:6,} ({pct:5.1f}% of negated sentences)")
print()

# Check 4: Character encoding issues?
print("CHECK 4: Character encoding check")
print("-" * 80)

# Look for any sentences that might have encoding issues
if count_inte_simple > 0:
    sample_text = df[contains_inte_simple].iloc[0]['sentence_text']
    print(f"Sample text: {sample_text}")
    print(f"Text type: {type(sample_text)}")
    print(f"Text repr: {repr(sample_text)}")
    print()
    
    # Check byte representation of 'inte'
    if 'inte' in str(sample_text).lower():
        idx = str(sample_text).lower().index('inte')
        word_bytes = sample_text[idx:idx+4]
        print(f"'inte' in text: {word_bytes}")
        print(f"Byte representation: {word_bytes.encode('utf-8')}")
        print()

# Check 5: Is the text actually lemmatized?
print("CHECK 5: Is text lemmatized?")
print("-" * 80)

# Look for common unlemmatized forms vs lemmatized forms
lemmatized_indicators = ['är', 'kan', 'ha', 'vara']
unlemmatized_indicators = ['inte', 'också', 'mycket']

sample_sentences = df['sentence_text'].head(20).tolist()
print("Sample of 20 sentences:")
for i, sent in enumerate(sample_sentences, 1):
    print(f"{i}. {str(sent)[:80]}{'...' if len(str(sent)) > 80 else ''}")
print()

# Check 6: Look at sentences that should have negation but don't
print("CHECK 6: Why might 'inte' not be detected?")
print("-" * 80)

if count_inte_simple > count_inte_word:
    print(f"⚠ Found {count_inte_simple - count_inte_word} sentences where 'inte' is part of another word")
    print()
    
    # Find examples where 'inte' is substring but not word
    is_substring = contains_inte_simple & ~contains_inte_word
    if is_substring.sum() > 0:
        print("Examples (inte as substring, not standalone word):")
        for i, (_, row) in enumerate(df[is_substring].head(5).iterrows(), 1):
            text = str(row['sentence_text'])
            print(f"{i}. {text[:100]}{'...' if len(text) > 100 else ''}")
        print()

# Final summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

if count_inte_word == 0:
    print("⚠ PROBLEM: 'inte' is not being detected as a standalone word")
    print()
    print("Possible causes:")
    print("1. The text is not actually lemmatized (shouldn't affect 'inte' though)")
    print("2. 'inte' appears as part of other words (e.g., 'integritet', 'intendent')")
    print("3. Character encoding issue (unlikely)")
    print("4. The sentence_text column doesn't contain the actual text")
    print()
    print("ACTION: Please share:")
    print("- A few example sentences from your data")
    print("- The output from CHECK 2 above")
    
elif count_inte_word < count_inte_simple:
    print(f"✓ 'inte' IS being detected: {count_inte_word:,} occurrences as standalone word")
    print(f"  (Plus {count_inte_simple - count_inte_word:,} as substring of other words - correctly excluded)")
    
else:
    print(f"✓ 'inte' IS being detected: {count_inte_word:,} occurrences")

print()
