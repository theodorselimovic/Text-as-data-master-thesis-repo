#!/usr/bin/env python3
"""
Risk Prevalence Analysis

Measures which individual risks are most common using two metrics:
1. Mention count: How many times each risk term appears across the corpus
2. Text devoted: Total characters in paragraphs where the risk is mentioned

This provides complementary views:
- Mention count captures frequency of explicit risk references
- Text devoted captures how much discussion/analysis each risk receives

Output:
- risk_prevalence.csv: Per-term metrics (mentions, paragraphs, characters, documents)
- risk_prevalence_by_actor.csv: Breakdown by actor type (kommun, lansstyrelse, MCF)
- risk_prevalence_by_wave.csv: Breakdown by time period

Usage:
    python risk_prevalence_analysis.py \\
        --corpus data/processed/bow_corpus_stemmed.parquet \\
        --output results/01_bow_analysis/prevalence/

    # With verbose output
    python risk_prevalence_analysis.py \\
        --corpus data/processed/bow_corpus_stemmed.parquet \\
        --output results/01_bow_analysis/prevalence/ \\
        --verbose

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-03-18
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem.snowball import SnowballStemmer

# Import dictionary and stopwords
sys.path.insert(0, str(Path(__file__).parent))
from risk_dictionary_individual import RISK_DICTIONARY_INDIVIDUAL

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

WAVE_LABELS = {
    0: 'pre-2015',
    1: '2015-2018',
    2: '2019-2022',
    3: '2023+',
}


def map_year_to_wave(year) -> int:
    """Map year to wave number."""
    try:
        year = int(year)
    except (TypeError, ValueError):
        return None
    if year < 2015:
        return 0
    elif year <= 2018:
        return 1
    elif year <= 2022:
        return 2
    else:
        return 3


# =============================================================================
# Dictionary Preparation
# =============================================================================

def build_stemmed_dictionary() -> dict:
    """
    Build stemmed dictionary with stopword removal.

    Returns
    -------
    dict
        {stemmed_term: {'original': original_term, 'canonical': canonical_risk}}
        where 'canonical' is the collapsed risk name from the individual dictionary.
    """
    stemmer = SnowballStemmer('swedish')
    stemmed_stopwords = {stemmer.stem(sw) for sw in SWEDISH_STOPWORDS}

    term_info = {}

    for canonical, variants in RISK_DICTIONARY_INDIVIDUAL.items():
        for term in variants:
            words = term.lower().split()

            # Stem words, filtering stopwords
            stemmed_words = []
            for w in words:
                stem = stemmer.stem(w)
                if w not in SWEDISH_STOPWORDS and stem not in stemmed_stopwords:
                    stemmed_words.append(stem)

            if not stemmed_words:
                continue

            if len(stemmed_words) == 1:
                stemmed_form = stemmed_words[0]
            else:
                stemmed_form = '_'.join(stemmed_words)

            # Map all variants to their canonical form
            if stemmed_form not in term_info:
                term_info[stemmed_form] = {
                    'original': term,
                    'canonical': canonical,
                }

    return term_info


# =============================================================================
# Prevalence Analysis
# =============================================================================

def aggregate_to_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentence-level data to paragraph-level.

    Returns DataFrame with one row per paragraph containing:
    - Combined tokens from all sentences
    - Combined text and character count
    - Metadata (doc_id, actor_type, year, wave)
    """
    logger.info("Aggregating sentences to paragraphs...")

    # Group by document and paragraph
    grouped = df.groupby(['doc_id', 'paragraph_id'])

    rows = []
    for (doc_id, para_id), group in grouped:
        # Combine tokens
        all_tokens = []
        for tokens in group['tokens']:
            if tokens is not None and hasattr(tokens, '__iter__'):
                all_tokens.extend(list(tokens))

        # Combine text
        para_text = ' '.join(group['sentence_text'].fillna(''))
        char_count = len(para_text)

        # Metadata from first row
        first = group.iloc[0]
        year = first.get('year', None)

        rows.append({
            'doc_id': doc_id,
            'paragraph_id': para_id,
            'actor_type': first.get('actor_type', 'unknown'),
            'year': year,
            'wave': map_year_to_wave(year) if pd.notna(year) else None,
            'tokens': all_tokens,
            'paragraph_text': para_text,
            'char_count': char_count,
        })

    result = pd.DataFrame(rows)
    logger.info(f"  Aggregated to {len(result):,} paragraphs")
    return result


def calculate_prevalence(
    paragraphs_df: pd.DataFrame,
    term_info: dict,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Calculate prevalence metrics for each risk (collapsed by canonical form).

    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        Paragraph-level data with 'tokens' and 'char_count' columns
    term_info : dict
        Stemmed term -> {original, canonical} mapping
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Per-risk prevalence metrics (aggregated by canonical form)
    """
    logger.info("Calculating prevalence metrics...")

    # Initialize counters for each term
    metrics = {term: {
        'mentions': 0,
        'paragraphs': 0,
        'characters': 0,
        'documents': set(),
    } for term in term_info}

    total_paragraphs = len(paragraphs_df)

    for idx, row in paragraphs_df.iterrows():
        if verbose and idx % 20000 == 0:
            logger.info(f"  Processing paragraph {idx:,}/{total_paragraphs:,}")

        tokens = row['tokens']
        if not tokens:
            continue

        # Count token occurrences in this paragraph
        token_counts = Counter(tokens)
        doc_id = row['doc_id']
        char_count = row['char_count']

        # Check each dictionary term
        for term in term_info:
            count = token_counts.get(term, 0)
            if count > 0:
                metrics[term]['mentions'] += count
                metrics[term]['paragraphs'] += 1
                metrics[term]['characters'] += char_count
                metrics[term]['documents'].add(doc_id)

    # Aggregate by canonical form
    canonical_metrics = defaultdict(lambda: {
        'mentions': 0,
        'paragraphs': 0,
        'characters': 0,
        'documents': set(),
        'variants': [],
    })

    for term, m in metrics.items():
        canonical = term_info[term]['canonical']
        canonical_metrics[canonical]['mentions'] += m['mentions']
        canonical_metrics[canonical]['paragraphs'] += m['paragraphs']
        canonical_metrics[canonical]['characters'] += m['characters']
        canonical_metrics[canonical]['documents'].update(m['documents'])
        if m['mentions'] > 0:
            canonical_metrics[canonical]['variants'].append(term)

    # Convert to DataFrame
    rows = []
    for canonical, m in canonical_metrics.items():
        rows.append({
            'risk': canonical,
            'mentions': m['mentions'],
            'paragraphs': m['paragraphs'],
            'characters': m['characters'],
            'documents': len(m['documents']),
            'stemmed_variants': ', '.join(sorted(m['variants'])),
        })

    result = pd.DataFrame(rows)

    if len(result) == 0:
        return result

    # Add derived metrics
    result['chars_per_mention'] = (
        result['characters'] / result['mentions'].replace(0, 1)
    ).round(1)
    result['mentions_per_doc'] = (
        result['mentions'] / result['documents'].replace(0, 1)
    ).round(2)

    # Sort by mentions (primary metric)
    result = result.sort_values('mentions', ascending=False).reset_index(drop=True)

    return result


def calculate_prevalence_by_group(
    paragraphs_df: pd.DataFrame,
    term_info: dict,
    group_col: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Calculate prevalence metrics grouped by a column (actor_type or wave).

    Returns long-format DataFrame with group column.
    """
    logger.info(f"Calculating prevalence by {group_col}...")

    groups = paragraphs_df[group_col].dropna().unique()
    all_results = []

    for group_val in groups:
        subset = paragraphs_df[paragraphs_df[group_col] == group_val]
        n_docs = subset['doc_id'].nunique()

        # Calculate for this group
        group_metrics = {term: {
            'mentions': 0,
            'paragraphs': 0,
            'characters': 0,
            'documents': set(),
        } for term in term_info}

        for _, row in subset.iterrows():
            tokens = row['tokens']
            if not tokens:
                continue

            token_counts = Counter(tokens)
            doc_id = row['doc_id']
            char_count = row['char_count']

            for term in term_info:
                count = token_counts.get(term, 0)
                if count > 0:
                    group_metrics[term]['mentions'] += count
                    group_metrics[term]['paragraphs'] += 1
                    group_metrics[term]['characters'] += char_count
                    group_metrics[term]['documents'].add(doc_id)

        # Aggregate by canonical form
        canonical_metrics = defaultdict(lambda: {
            'mentions': 0,
            'paragraphs': 0,
            'characters': 0,
            'documents': set(),
        })

        for term, m in group_metrics.items():
            canonical = term_info[term]['canonical']
            canonical_metrics[canonical]['mentions'] += m['mentions']
            canonical_metrics[canonical]['paragraphs'] += m['paragraphs']
            canonical_metrics[canonical]['characters'] += m['characters']
            canonical_metrics[canonical]['documents'].update(m['documents'])

        for canonical, m in canonical_metrics.items():
            all_results.append({
                group_col: group_val,
                'risk': canonical,
                'mentions': m['mentions'],
                'paragraphs': m['paragraphs'],
                'characters': m['characters'],
                'documents': len(m['documents']),
                'total_docs_in_group': n_docs,
                'mentions_per_doc': round(m['mentions'] / n_docs, 2) if n_docs > 0 else 0,
                'chars_per_doc': round(m['characters'] / n_docs, 1) if n_docs > 0 else 0,
            })

    result = pd.DataFrame(all_results)
    return result


# =============================================================================
# Output
# =============================================================================

def print_top_risks(prevalence_df: pd.DataFrame, n: int = 20) -> None:
    """Print top risks by both metrics."""
    print("\n" + "=" * 70)
    print("TOP RISKS BY MENTION COUNT")
    print("=" * 70)
    print(f"{'Rank':<5} {'Risk':<40} {'Mentions':>10} {'Docs':>6}")
    print("-" * 65)

    for i, row in prevalence_df.head(n).iterrows():
        print(f"{i+1:<5} {row['risk']:<40} {row['mentions']:>10,} "
              f"{row['documents']:>6}")

    print("\n" + "=" * 70)
    print("TOP RISKS BY TEXT DEVOTED (Characters)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Risk':<40} {'Characters':>12} {'Paragraphs':>10}")
    print("-" * 70)

    by_chars = prevalence_df.sort_values('characters', ascending=False).head(n)
    for rank, (i, row) in enumerate(by_chars.iterrows(), 1):
        print(f"{rank:<5} {row['risk']:<40} {row['characters']:>12,} "
              f"{row['paragraphs']:>10,}")


def print_actor_comparison(by_actor_df: pd.DataFrame, top_n: int = 10) -> None:
    """Print top risks comparison by actor."""
    print("\n" + "=" * 70)
    print("TOP RISKS BY ACTOR (mentions per document)")
    print("=" * 70)

    actors = by_actor_df['actor_type'].unique()

    for actor in sorted(actors):
        subset = by_actor_df[by_actor_df['actor_type'] == actor]
        n_docs = subset['total_docs_in_group'].iloc[0] if len(subset) > 0 else 0
        top = subset.nlargest(top_n, 'mentions_per_doc')

        print(f"\n{actor.upper()} ({n_docs} documents):")
        print(f"  {'Risk':<40} {'Mentions/Doc':>12} {'Total':>8}")
        print("  " + "-" * 60)

        for _, row in top.iterrows():
            if row['mentions'] > 0:
                print(f"  {row['risk']:<40} {row['mentions_per_doc']:>12.2f} "
                      f"{row['mentions']:>8,}")


# =============================================================================
# Visualization
# =============================================================================

# Actor colors and labels
ACTOR_COLORS = {'kommun': '#e41a1c', 'lansstyrelse': '#377eb8', 'MCF': '#4daf4a'}
ACTOR_LABELS = {'kommun': 'Municipality', 'lansstyrelse': 'Prefecture', 'MCF': 'MSB'}


def plot_prevalence_comparison(
    prevalence_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 30,
) -> None:
    """
    Create side-by-side bar chart comparing mentions and characters.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # Get top N by mentions
    top_df = prevalence_df.nlargest(top_n, 'mentions').copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    # Left: mentions
    ax1 = axes[0]
    ax1.barh(top_df['risk'], top_df['mentions'], color='#377eb8')
    ax1.set_xlabel('Total mentions')
    ax1.set_title('By Mention Count', fontweight='bold')
    ax1.invert_yaxis()

    # Right: characters (in thousands)
    ax2 = axes[1]
    # Sort by characters for this panel
    top_by_chars = prevalence_df.nlargest(top_n, 'characters').copy()
    ax2.barh(top_by_chars['risk'], top_by_chars['characters'] / 1000, color='#4daf4a')
    ax2.set_xlabel('Characters (thousands)')
    ax2.set_title('By Text Devoted', fontweight='bold')
    ax2.invert_yaxis()

    plt.suptitle(f'Top {top_n} Risks: Mentions vs Text Devoted', fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_prevalence_barchart.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'risk_prevalence_barchart.pdf', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: risk_prevalence_barchart.png/pdf")


def plot_prevalence_by_actor(
    by_actor_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """
    Create bar charts showing top risks per actor.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    actors = ['kommun', 'lansstyrelse', 'MCF']
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for ax, actor in zip(axes, actors):
        subset = by_actor_df[by_actor_df['actor_type'] == actor].nlargest(top_n, 'mentions_per_doc')
        ax.barh(subset['risk'], subset['mentions_per_doc'], color=ACTOR_COLORS[actor])
        ax.set_xlabel('Mentions per document')
        ax.set_title(f'{ACTOR_LABELS[actor]}', fontweight='bold')
        ax.invert_yaxis()

    plt.suptitle('Top 10 Risks by Actor (mentions/doc)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_prevalence_by_actor.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'risk_prevalence_by_actor.pdf', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: risk_prevalence_by_actor.png/pdf")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze prevalence of individual risk terms',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--corpus', '-c',
        type=Path,
        required=True,
        help='Path to stemmed corpus parquet'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./results/01_bow_analysis/prevalence'),
        help='Output directory'
    )

    parser.add_argument(
        '--top', '-n',
        type=int,
        default=20,
        help='Number of top risks to display (default: 20)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        return 1

    print("=" * 70)
    print("RISK PREVALENCE ANALYSIS")
    print("=" * 70)
    print(f"Corpus: {args.corpus}")
    print(f"Output: {args.output}")

    # Load data
    logger.info("\nLoading corpus...")
    df = pd.read_parquet(args.corpus)
    logger.info(f"  Loaded {len(df):,} sentences from {df['doc_id'].nunique()} documents")

    # Build stemmed dictionary
    logger.info("\nBuilding stemmed dictionary...")
    term_info = build_stemmed_dictionary()
    logger.info(f"  {len(term_info)} unique stemmed terms")

    # Aggregate to paragraphs
    paragraphs_df = aggregate_to_paragraphs(df)

    # Calculate overall prevalence
    logger.info("\n" + "-" * 70)
    prevalence_df = calculate_prevalence(paragraphs_df, term_info, args.verbose)

    # Calculate by actor
    by_actor_df = calculate_prevalence_by_group(
        paragraphs_df, term_info, 'actor_type', args.verbose
    )

    # Calculate by wave
    by_wave_df = calculate_prevalence_by_group(
        paragraphs_df, term_info, 'wave', args.verbose
    )
    # Add wave labels
    by_wave_df['wave_label'] = by_wave_df['wave'].map(WAVE_LABELS)

    # Print summaries
    print_top_risks(prevalence_df, args.top)
    print_actor_comparison(by_actor_df)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total_mentions = prevalence_df['mentions'].sum()
    total_chars = prevalence_df['characters'].sum()
    terms_with_mentions = (prevalence_df['mentions'] > 0).sum()
    total_terms = len(prevalence_df)

    print(f"\nCorpus: {df['doc_id'].nunique()} documents, {len(paragraphs_df):,} paragraphs")
    print(f"Dictionary: {terms_with_mentions}/{total_terms} risks found in corpus")
    print(f"Total risk mentions: {total_mentions:,}")
    print(f"Total characters in risk paragraphs: {total_chars:,}")

    # Save outputs
    args.output.mkdir(parents=True, exist_ok=True)

    # Main prevalence file
    out_path = args.output / 'risk_prevalence.csv'
    prevalence_df.to_csv(out_path, index=False, encoding='utf-8')
    logger.info(f"\nSaved: {out_path}")

    # By actor
    out_path = args.output / 'risk_prevalence_by_actor.csv'
    by_actor_df.to_csv(out_path, index=False, encoding='utf-8')
    logger.info(f"Saved: {out_path}")

    # By wave
    out_path = args.output / 'risk_prevalence_by_wave.csv'
    by_wave_df.to_csv(out_path, index=False, encoding='utf-8')
    logger.info(f"Saved: {out_path}")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_prevalence_comparison(prevalence_df, args.output, top_n=30)
    plot_prevalence_by_actor(by_actor_df, args.output, top_n=10)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
