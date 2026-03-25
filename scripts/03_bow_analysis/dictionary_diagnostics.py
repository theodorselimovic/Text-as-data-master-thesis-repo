#!/usr/bin/env python3
"""
Dictionary Diagnostics Tool

Analyzes why dictionary terms may not match against the stemmed corpus.
Classifies each term's matching status and identifies the root cause.

Root causes analyzed:
- matched: term found in corpus vocabulary
- stopword_mismatch: original term contains stopwords filtered during preprocessing
- ngram_too_long: term has more tokens than max_ngram after stopword removal
- never_appears: term genuinely doesn't appear in any document

Usage:
    python dictionary_diagnostics.py \\
        --corpus data/processed/bow_corpus_stemmed.parquet \\
        --output results/00_data_preparation/diagnostics/

    # Check specific max_ngram setting
    python dictionary_diagnostics.py \\
        --corpus data/processed/bow_corpus_stemmed.parquet \\
        --output results/00_data_preparation/diagnostics/ \\
        --max-ngram 5

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-03-18
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Set

import pandas as pd
from nltk.stem.snowball import SnowballStemmer

# Import dictionary and stopwords
sys.path.insert(0, str(Path(__file__).parent))
from risk_dictionary_categories import RISK_DICTIONARY_CATEGORIES as RISK_DICTIONARY_ORIGINAL

sys.path.insert(0, str(Path(__file__).parent.parent / '02_preprocessing'))
from preprocessing_bow import SWEDISH_STOPWORDS


# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Diagnostic Functions
# =============================================================================

def build_corpus_vocabulary(df: pd.DataFrame) -> tuple:
    """
    Build vocabulary from corpus with frequency counts.

    Parameters
    ----------
    df : pd.DataFrame
        Corpus with 'tokens' column containing lists of tokens.

    Returns
    -------
    tuple of (Set[str], Counter)
        (vocabulary_set, token_frequency_counter)
    """
    logger.info("Building corpus vocabulary...")

    token_counts = Counter()
    for tokens in df['tokens']:
        if tokens is not None and hasattr(tokens, '__iter__'):
            token_counts.update(tokens)

    vocabulary = set(token_counts.keys())
    logger.info(f"  Vocabulary size: {len(vocabulary):,} unique tokens")
    logger.info(f"  Total token occurrences: {sum(token_counts.values()):,}")

    return vocabulary, token_counts


def analyze_term(
    term: str,
    category: str,
    vocabulary: Set[str],
    token_counts: Counter,
    stopwords: Set[str],
    stemmed_stopwords: Set[str],
    stemmer: SnowballStemmer,
    max_ngram: int
) -> dict:
    """
    Analyze a single dictionary term for matching issues.

    Parameters
    ----------
    term : str
        Original dictionary term
    category : str
        Risk category
    vocabulary : Set[str]
        Corpus vocabulary
    token_counts : Counter
        Token frequency counts
    stopwords : Set[str]
        Original stopwords
    stemmed_stopwords : Set[str]
        Stemmed stopwords
    stemmer : SnowballStemmer
        Swedish stemmer
    max_ngram : int
        Maximum n-gram size in corpus

    Returns
    -------
    dict
        Analysis result with fields: term, category, stemmed_form,
        matched, reason, corpus_frequency, words_original, words_after_stopwords
    """
    words = term.lower().split()

    # Stem all words (before stopword removal)
    stemmed_all = [stemmer.stem(w) for w in words]

    # Stem words, removing stopwords (matching corpus behavior)
    stemmed_filtered = []
    stopwords_removed = []
    for w in words:
        stem = stemmer.stem(w)
        if w in stopwords or stem in stemmed_stopwords:
            stopwords_removed.append(w)
        else:
            stemmed_filtered.append(stem)

    # Create the n-gram form (what the dictionary should produce)
    if len(stemmed_filtered) == 0:
        # All words were stopwords
        stemmed_form_filtered = ""
        stemmed_form_all = '_'.join(stemmed_all)
        matched = False
        reason = "all_stopwords"
        frequency = 0
    elif len(stemmed_filtered) == 1:
        stemmed_form_filtered = stemmed_filtered[0]
        stemmed_form_all = '_'.join(stemmed_all)
        matched = stemmed_form_filtered in vocabulary
        frequency = token_counts.get(stemmed_form_filtered, 0)

        if matched:
            reason = "matched"
        elif len(stemmed_all) > 1 and stopwords_removed:
            reason = "stopword_mismatch"
        else:
            reason = "never_appears"
    else:
        # Multi-word term
        stemmed_form_filtered = '_'.join(stemmed_filtered)
        stemmed_form_all = '_'.join(stemmed_all)

        if len(stemmed_filtered) > max_ngram:
            # Too long for corpus n-grams
            matched = False
            reason = "ngram_too_long"
            frequency = 0
        else:
            matched = stemmed_form_filtered in vocabulary
            frequency = token_counts.get(stemmed_form_filtered, 0)

            if matched:
                reason = "matched"
            elif stopwords_removed and stemmed_form_all != stemmed_form_filtered:
                # Dictionary might be using stopword-inclusive form
                reason = "stopword_mismatch"
            else:
                reason = "never_appears"

    return {
        'term': term,
        'category': category,
        'stemmed_form_with_stopwords': stemmed_form_all,
        'stemmed_form_no_stopwords': stemmed_form_filtered,
        'words_original': len(words),
        'words_after_stopwords': len(stemmed_filtered),
        'stopwords_removed': ', '.join(stopwords_removed) if stopwords_removed else '',
        'matched': matched,
        'reason': reason,
        'corpus_frequency': frequency,
    }


def run_diagnostics(
    corpus_path: Path,
    max_ngram: int = 3
) -> pd.DataFrame:
    """
    Run full diagnostics on dictionary against corpus.

    Parameters
    ----------
    corpus_path : Path
        Path to stemmed corpus parquet
    max_ngram : int
        Maximum n-gram size in corpus

    Returns
    -------
    pd.DataFrame
        Diagnostic results for all dictionary terms
    """
    # Load corpus
    logger.info(f"Loading corpus from: {corpus_path}")
    df = pd.read_parquet(corpus_path)
    logger.info(f"  Loaded {len(df):,} rows")

    # Build vocabulary
    vocabulary, token_counts = build_corpus_vocabulary(df)

    # Set up stemmer and stopwords
    stemmer = SnowballStemmer('swedish')
    stopwords = SWEDISH_STOPWORDS
    stemmed_stopwords = {stemmer.stem(sw) for sw in stopwords}

    # Analyze each dictionary term
    logger.info("\nAnalyzing dictionary terms...")
    results = []

    for category, terms in RISK_DICTIONARY_ORIGINAL.items():
        for term in terms:
            result = analyze_term(
                term, category, vocabulary, token_counts,
                stopwords, stemmed_stopwords, stemmer, max_ngram
            )
            results.append(result)

    results_df = pd.DataFrame(results)

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    total = len(results_df)
    matched = (results_df['reason'] == 'matched').sum()
    stopword_mismatch = (results_df['reason'] == 'stopword_mismatch').sum()
    ngram_too_long = (results_df['reason'] == 'ngram_too_long').sum()
    never_appears = (results_df['reason'] == 'never_appears').sum()
    all_stopwords = (results_df['reason'] == 'all_stopwords').sum()

    logger.info(f"\nTotal dictionary terms: {total}")
    logger.info(f"  Matched:            {matched:3d} ({matched/total*100:5.1f}%)")
    logger.info(f"  Stopword mismatch:  {stopword_mismatch:3d} ({stopword_mismatch/total*100:5.1f}%)")
    logger.info(f"  N-gram too long:    {ngram_too_long:3d} ({ngram_too_long/total*100:5.1f}%)")
    logger.info(f"  Never appears:      {never_appears:3d} ({never_appears/total*100:5.1f}%)")
    if all_stopwords > 0:
        logger.info(f"  All stopwords:      {all_stopwords:3d} ({all_stopwords/total*100:5.1f}%)")

    # Show problematic terms by category
    logger.info("\n" + "-" * 60)
    logger.info("ISSUES BY REASON")
    logger.info("-" * 60)

    for reason in ['stopword_mismatch', 'ngram_too_long', 'never_appears']:
        subset = results_df[results_df['reason'] == reason]
        if len(subset) > 0:
            logger.info(f"\n{reason.upper()} ({len(subset)} terms):")
            for _, row in subset.head(10).iterrows():
                logger.info(f"  '{row['term']}' -> {row['stemmed_form_no_stopwords']}")
                if row['stopwords_removed']:
                    logger.info(f"    (removed: {row['stopwords_removed']})")
            if len(subset) > 10:
                logger.info(f"  ... and {len(subset) - 10} more")

    return results_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Diagnose dictionary matching issues against stemmed corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--corpus', '-c',
        type=Path,
        required=True,
        help='Path to stemmed corpus parquet (bow_corpus_stemmed.parquet)'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./results/00_data_preparation/diagnostics'),
        help='Output directory for results'
    )

    parser.add_argument(
        '--max-ngram',
        type=int,
        default=3,
        help='Maximum n-gram size in corpus (default: 3)'
    )

    args = parser.parse_args()

    # Validate input
    if not args.corpus.exists():
        logger.error(f"Corpus file not found: {args.corpus}")
        return 1

    logger.info("=" * 70)
    logger.info("DICTIONARY DIAGNOSTICS")
    logger.info("=" * 70)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Max n-gram: {args.max_ngram}")
    logger.info(f"Output: {args.output}")

    # Run diagnostics
    results_df = run_diagnostics(args.corpus, args.max_ngram)

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / 'dictionary_diagnostics.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"\nSaved detailed results to: {output_path}")

    # Save summary
    summary = results_df.groupby('reason').size().reset_index(name='count')
    summary_path = args.output / 'diagnostics_summary.csv'
    summary.to_csv(summary_path, index=False, encoding='utf-8')
    logger.info(f"Saved summary to: {summary_path}")

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
