#!/usr/bin/env python3
"""
Allvarliga Störningar Risk Analysis (Stemmed Token-Based)

Analyzes which risk categories co-occur with "allvarliga störningar" (serious
disruptions) phrases in RSA documents. This reveals how different actors frame
risks in terms of their potential to cause operational disruptions.

Uses pre-stemmed corpus for fast token matching (consistent with other BOW scripts).

Input:
    data/processed/bow_corpus_stemmed.parquet (sentence-level with tokens)

Output:
    results/01_bow_analysis/allvarliga_storningar/allvarliga_storningar_risks.csv
    results/01_bow_analysis/allvarliga_storningar/allvarliga_storningar_by_actor.csv
    results/01_bow_analysis/allvarliga_storningar/allvarliga_storningar_by_wave.csv
    results/01_bow_analysis/allvarliga_storningar/allvarliga_storningar_sample.csv

Usage:
    python allvarliga_storningar_analysis.py
    python allvarliga_storningar_analysis.py --corpus data/processed/bow_corpus_stemmed.parquet
    python allvarliga_storningar_analysis.py --verbose --sample-size 50

Requirements:
    pip install pandas pyarrow nltk
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer

# Import risk dictionaries
from risk_dictionary_individual import RISK_DICTIONARY_INDIVIDUAL
from risk_dictionary_categories import RISK_DICTIONARY_CATEGORIES as RISK_DICTIONARY

# Import stopwords from preprocessing
sys.path.insert(0, str(Path(__file__).parent.parent / '02_preprocessing'))
from preprocessing_bow import SWEDISH_STOPWORDS

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# =============================================================================
# STEMMER INITIALIZATION
# =============================================================================

STEMMER = SnowballStemmer("swedish")


def stem_phrase(phrase: str, stopwords: Set[str]) -> str:
    """
    Stem a multi-word phrase, removing stopwords.
    Returns underscore-joined stemmed tokens (matching corpus n-gram format).
    """
    tokens = phrase.lower().split()
    stemmed = [STEMMER.stem(t) for t in tokens if t not in stopwords]
    return '_'.join(stemmed) if stemmed else ''


def stem_term(term: str) -> str:
    """Stem a single term."""
    return STEMMER.stem(term.lower())


# =============================================================================
# DISRUPTION PATTERNS (STEMMED)
# =============================================================================

# Phrases indicating "serious disruptions" to actor's operations
DISRUPTION_PHRASES_RAW = [
    'allvarliga störningar',
    'allvarlig störning',
    'störningar i verksamheten',
    'störningar av verksamheten',
    'störning i verksamheten',
    'störning av verksamheten',
    'störd verksamhet',
    'samhällsstörning',
    'samhällsstörningar',
    'allvarlig samhällsstörning',
    'allvarliga samhällsstörningar',
]


def build_stemmed_disruption_terms(stopwords: Set[str]) -> Set[str]:
    """Build set of stemmed disruption n-grams."""
    stemmed_terms = set()
    for phrase in DISRUPTION_PHRASES_RAW:
        stemmed = stem_phrase(phrase, stopwords)
        if stemmed:
            stemmed_terms.add(stemmed)
    return stemmed_terms


# =============================================================================
# STEMMED RISK DICTIONARY
# =============================================================================

def build_stemmed_category_dict(stopwords: Set[str]) -> Dict[str, Set[str]]:
    """Build stemmed version of category dictionary."""
    stemmed_dict = {}
    for category, terms in RISK_DICTIONARY.items():
        stemmed_terms = set()
        for term in terms:
            stemmed = stem_phrase(term, stopwords)
            if stemmed:
                stemmed_terms.add(stemmed)
        stemmed_dict[category] = stemmed_terms
    return stemmed_dict


def build_stemmed_individual_dict(stopwords: Set[str]) -> Dict[str, Set[str]]:
    """Build stemmed version of individual risk dictionary."""
    stemmed_dict = {}
    for canonical, variants in RISK_DICTIONARY_INDIVIDUAL.items():
        stemmed_variants = set()
        for variant in variants:
            stemmed = stem_phrase(variant, stopwords)
            if stemmed:
                stemmed_variants.add(stemmed)
        if stemmed_variants:
            stemmed_dict[canonical] = stemmed_variants
    return stemmed_dict


# =============================================================================
# TOKEN MATCHING FUNCTIONS
# =============================================================================

def tokens_contain_any(tokens: List[str], target_terms: Set[str]) -> bool:
    """Check if token list contains any of the target terms."""
    token_set = set(tokens)
    return bool(token_set & target_terms)


def count_term_occurrences(tokens: List[str], target_terms: Set[str]) -> int:
    """Count how many times any target term appears in tokens."""
    count = 0
    for token in tokens:
        if token in target_terms:
            count += 1
    return count


def count_risk_terms_by_category_tokens(
    tokens: List[str],
    stemmed_dict: Dict[str, Set[str]],
) -> Dict[str, int]:
    """Count occurrences of risk terms by category using token matching."""
    results = {}
    token_set = set(tokens)

    for category, terms in stemmed_dict.items():
        # Count matches
        count = sum(1 for t in tokens if t in terms)
        results[category] = count

    return results


def count_individual_risk_terms_tokens(
    tokens: List[str],
    stemmed_individual_dict: Dict[str, Set[str]],
) -> Dict[str, int]:
    """Count individual risk terms using token matching."""
    results = {}

    for canonical, variants in stemmed_individual_dict.items():
        count = sum(1 for t in tokens if t in variants)
        if count > 0:
            results[canonical] = count

    return results


# =============================================================================
# PARAGRAPH PROCESSING
# =============================================================================

def derive_wave(year) -> int:
    """Derive wave from year."""
    try:
        year = int(year)
    except (ValueError, TypeError):
        return -1

    if year < 2015:
        return 0
    elif year <= 2018:
        return 1
    elif year <= 2022:
        return 2
    else:
        return 3


def group_sentences_to_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group sentences by paragraph, aggregating tokens.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level dataframe with doc_id, paragraph_id, tokens columns.

    Returns
    -------
    pd.DataFrame
        Paragraph-level dataframe with aggregated tokens and metadata.
    """
    logger.info("Grouping sentences into paragraphs...")

    # Columns to aggregate
    meta_cols = ['doc_id', 'paragraph_id', 'actor_type', 'year', 'municipality']
    available_meta = [col for col in meta_cols if col in df.columns]

    # Parse tokens if stored as string
    if df['tokens'].dtype == object and isinstance(df['tokens'].iloc[0], str):
        df = df.copy()
        df['tokens'] = df['tokens'].apply(eval)

    # Aggregate: concatenate token lists, keep first metadata
    def concat_tokens(token_lists):
        all_tokens = []
        for tl in token_lists:
            if isinstance(tl, list):
                all_tokens.extend(tl)
        return all_tokens

    agg_dict = {'tokens': concat_tokens}
    for col in available_meta:
        if col not in ['doc_id', 'paragraph_id']:
            agg_dict[col] = 'first'

    # Also keep sentence_text for samples if available
    if 'sentence_text' in df.columns:
        agg_dict['sentence_text'] = lambda x: ' '.join(x.astype(str))

    paragraphs = df.groupby(['doc_id', 'paragraph_id']).agg(agg_dict).reset_index()

    # Rename sentence_text to full_text
    if 'sentence_text' in paragraphs.columns:
        paragraphs.rename(columns={'sentence_text': 'full_text'}, inplace=True)

    # Derive wave from year if not present
    if 'wave' not in paragraphs.columns and 'year' in paragraphs.columns:
        paragraphs['wave'] = paragraphs['year'].apply(derive_wave)

    logger.info(f"  Created {len(paragraphs):,} paragraphs from {len(df):,} sentences")
    return paragraphs


def filter_disruption_paragraphs(
    paragraphs: pd.DataFrame,
    disruption_terms: Set[str],
) -> pd.DataFrame:
    """Filter to paragraphs containing disruption terms (token-based)."""
    logger.info("Filtering paragraphs with disruption phrases...")

    def has_disruption(tokens):
        if not isinstance(tokens, list):
            return False
        return tokens_contain_any(tokens, disruption_terms)

    mask = paragraphs['tokens'].apply(has_disruption)
    filtered = paragraphs[mask].copy()

    logger.info(f"  Found {len(filtered):,} paragraphs with disruption phrases")
    logger.info(f"  ({len(filtered)/len(paragraphs)*100:.1f}% of total paragraphs)")

    return filtered


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_individual_terms(
    paragraphs: pd.DataFrame,
    stemmed_individual_dict: Dict[str, Set[str]],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Count individual risk terms in paragraphs using token matching.
    """
    if group_col is None:
        mention_totals = defaultdict(int)
        para_totals = defaultdict(int)

        for _, row in paragraphs.iterrows():
            tokens = row['tokens']
            if not isinstance(tokens, list):
                continue

            counts = count_individual_risk_terms_tokens(tokens, stemmed_individual_dict)

            for term, count in counts.items():
                mention_totals[term] += count
                para_totals[term] += 1

        rows = []
        for term in mention_totals.keys():
            rows.append({
                'term': term,
                'mentions': mention_totals[term],
                'paragraphs': para_totals[term],
            })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df

        df = df.sort_values('mentions', ascending=False).reset_index(drop=True)
        df['pct_mentions'] = (df['mentions'] / df['mentions'].sum() * 100).round(2)
        return df

    else:
        results = []
        for group_val, group_df in paragraphs.groupby(group_col):
            mention_totals = defaultdict(int)
            para_totals = defaultdict(int)

            for _, row in group_df.iterrows():
                tokens = row['tokens']
                if not isinstance(tokens, list):
                    continue

                counts = count_individual_risk_terms_tokens(tokens, stemmed_individual_dict)

                for term, count in counts.items():
                    mention_totals[term] += count
                    para_totals[term] += 1

            for term in mention_totals.keys():
                results.append({
                    group_col: group_val,
                    'term': term,
                    'mentions': mention_totals[term],
                    'paragraphs': para_totals[term],
                })

        df = pd.DataFrame(results)
        if len(df) == 0:
            return df

        df['pct_mentions'] = df.groupby(group_col)['mentions'].transform(
            lambda x: (x / x.sum() * 100).round(2)
        )

        return df


def analyze_risk_categories(
    paragraphs: pd.DataFrame,
    stemmed_category_dict: Dict[str, Set[str]],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Count risk categories in paragraphs using token matching."""
    categories = list(stemmed_category_dict.keys())

    if group_col is None:
        totals = {cat: 0 for cat in categories}

        for _, row in paragraphs.iterrows():
            tokens = row['tokens']
            if not isinstance(tokens, list):
                continue
            counts = count_risk_terms_by_category_tokens(tokens, stemmed_category_dict)
            for cat, count in counts.items():
                totals[cat] += count

        df = pd.DataFrame([{
            'category': cat,
            'count': totals[cat],
        } for cat in categories])
        df = df.sort_values('count', ascending=False).reset_index(drop=True)
        df['pct'] = (df['count'] / df['count'].sum() * 100).round(1)
        return df

    else:
        results = []
        for group_val, group_df in paragraphs.groupby(group_col):
            totals = {cat: 0 for cat in categories}

            for _, row in group_df.iterrows():
                tokens = row['tokens']
                if not isinstance(tokens, list):
                    continue
                counts = count_risk_terms_by_category_tokens(tokens, stemmed_category_dict)
                for cat, count in counts.items():
                    totals[cat] += count

            for cat in categories:
                results.append({
                    group_col: group_val,
                    'category': cat,
                    'count': totals[cat],
                })

        df = pd.DataFrame(results)
        df['pct'] = df.groupby(group_col)['count'].transform(
            lambda x: (x / x.sum() * 100).round(1) if x.sum() > 0 else 0
        )

        return df


def extract_sample_paragraphs(
    paragraphs: pd.DataFrame,
    sample_size: int = 30,
) -> pd.DataFrame:
    """Extract a sample of paragraphs for manual inspection."""
    n = min(sample_size, len(paragraphs))
    sample = paragraphs.sample(n=n, random_state=42)

    cols = ['doc_id', 'actor_type', 'wave', 'year', 'full_text']
    available = [c for c in cols if c in sample.columns]

    return sample[available].copy()


# =============================================================================
# VISUALIZATION
# =============================================================================

CATEGORY_LABELS = {
    'naturhot': 'Natural hazards',
    'antagonistiska_hot': 'Antagonistic threats',
    'biologiska_hot': 'Biological threats',
    'teknisk_infrastruktur': 'Technical infrastructure',
    'olyckor': 'Accidents',
    'miljö_klimat': 'Environment/climate',
    'sociala_risker': 'Social risks',
    'cyber_hot': 'Cyber threats',
    'ekonomi': 'Economy',
}

WAVE_LABELS = {
    0: 'Pre-2015',
    1: '2015-2018',
    2: '2019-2022',
    3: '2023+',
}

ACTOR_LABELS = {
    'kommun': 'Municipality',
    'lansstyrelse': 'Prefecture',
    'länsstyrelse': 'Prefecture',
    'MCF': 'MSB',
}

ACTOR_COLORS = {
    'kommun': '#e41a1c',
    'lansstyrelse': '#377eb8',
    'MCF': '#4daf4a',
}


def create_visualizations(
    overall: pd.DataFrame,
    by_wave: pd.DataFrame,
    by_actor: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create and save visualizations."""
    logger.info("Creating visualizations...")

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    # 1. Overall risk category distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    overall_plot = overall.copy()
    overall_plot['label'] = overall_plot['category'].map(CATEGORY_LABELS)

    bars = ax.barh(
        overall_plot['label'],
        overall_plot['count'],
        color=sns.color_palette("Blues_r", n_colors=len(overall_plot)),
    )

    ax.set_xlabel('Number of mentions')
    ax.set_title('Risk Categories in "Allvarliga Störningar" Paragraphs', fontsize=12)
    ax.invert_yaxis()

    for bar, pct in zip(bars, overall_plot['pct']):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_overall.png')
    fig.savefig(output_dir / 'allvarliga_storningar_overall.pdf')
    plt.close(fig)

    # 2. Development over time
    fig, ax = plt.subplots(figsize=(12, 7))

    pivot = by_wave.pivot(index='wave', columns='category', values='count')
    cat_order = overall['category'].tolist()
    pivot = pivot[[c for c in cat_order if c in pivot.columns]]
    pivot.columns = [CATEGORY_LABELS.get(c, c) for c in pivot.columns]
    pivot.index = [WAVE_LABELS.get(w, w) for w in pivot.index]

    pivot.plot(kind='line', marker='o', markersize=8, linewidth=2.5, ax=ax,
               color=sns.color_palette("husl", n_colors=len(pivot.columns)))

    ax.set_xlabel('Time period')
    ax.set_ylabel('Number of mentions')
    ax.set_title('Risk Categories Over Time in "Allvarliga Störningar" Paragraphs', fontsize=12)
    ax.legend(title='Risk category', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_over_time.png')
    fig.savefig(output_dir / 'allvarliga_storningar_over_time.pdf')
    plt.close(fig)

    # 3. By actor type
    fig, ax = plt.subplots(figsize=(12, 6))

    pivot_actor = by_actor.pivot(index='category', columns='actor_type', values='count')
    pivot_actor['total'] = pivot_actor.sum(axis=1)
    pivot_actor = pivot_actor.sort_values('total', ascending=False).drop(columns='total')
    pivot_actor = pivot_actor.head(8)

    pivot_actor.index = [CATEGORY_LABELS.get(c, c) for c in pivot_actor.index]
    pivot_actor.columns = [ACTOR_LABELS.get(c, c) for c in pivot_actor.columns]

    col_order = ['Municipality', 'Prefecture', 'MSB']
    col_colors = {'Municipality': '#e41a1c', 'Prefecture': '#377eb8', 'MSB': '#4daf4a'}
    pivot_actor = pivot_actor[[c for c in col_order if c in pivot_actor.columns]]

    pivot_actor.plot(kind='barh', ax=ax, width=0.8,
                     color=[col_colors[c] for c in pivot_actor.columns])

    ax.set_xlabel('Number of mentions')
    ax.set_title('Risk Categories by Actor Type', fontsize=12)
    ax.legend(title='Actor type')
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_by_actor.png')
    fig.savefig(output_dir / 'allvarliga_storningar_by_actor.pdf')
    plt.close(fig)

    logger.info("  All visualizations saved")


def create_term_visualization(
    terms_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 25,
) -> None:
    """Create visualization for individual risk terms."""
    logger.info("Creating term-level visualization...")

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(10, 10))

    top_by_mentions = terms_df.nlargest(top_n, 'mentions').iloc[::-1]
    colors = sns.color_palette("Blues", n_colors=top_n)

    bars = ax.barh(top_by_mentions['term'], top_by_mentions['mentions'],
                   color=colors, edgecolor='none')

    for bar, val in zip(bars, top_by_mentions['mentions']):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2, f'{val:,}',
                va='center', ha='left', fontsize=8)

    ax.set_xlabel('Mentions', fontsize=11)
    ax.set_title(f'Top {top_n} Risks in "Allvarliga Störningar" Paragraphs',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, top_by_mentions['mentions'].max() * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_terms.png', dpi=150)
    fig.savefig(output_dir / 'allvarliga_storningar_terms.pdf')
    plt.close(fig)

    logger.info("  Saved term-level chart")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Analyze risks associated with "allvarliga störningar"',
    )
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/processed/bow_corpus_stemmed.parquet'),
        help='Input corpus (default: data/processed/bow_corpus_stemmed.parquet)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/01_bow_analysis/allvarliga_storningar'),
        help='Output directory',
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=30,
        help='Number of sample paragraphs to extract (default: 30)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output',
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # Build stemmed dictionaries
    logger.info("Building stemmed dictionaries...")
    stopwords = SWEDISH_STOPWORDS
    disruption_terms = build_stemmed_disruption_terms(stopwords)
    stemmed_category_dict = build_stemmed_category_dict(stopwords)
    stemmed_individual_dict = build_stemmed_individual_dict(stopwords)

    logger.info(f"  Disruption terms: {len(disruption_terms)}")
    logger.info(f"  Category dict: {len(stemmed_category_dict)} categories")
    logger.info(f"  Individual dict: {len(stemmed_individual_dict)} risks")

    # Load corpus
    logger.info(f"Loading corpus from {args.corpus}...")
    df = pd.read_parquet(args.corpus)
    logger.info(f"  Loaded {len(df):,} sentences")

    # Group to paragraphs
    paragraphs = group_sentences_to_paragraphs(df)

    # Filter to disruption paragraphs
    filtered = filter_disruption_paragraphs(paragraphs, disruption_terms)

    if len(filtered) == 0:
        logger.warning("No paragraphs found with disruption phrases!")
        return 1

    # Analyze individual risk terms
    logger.info("Analyzing individual risk terms...")
    terms_overall = analyze_individual_terms(filtered, stemmed_individual_dict)
    terms_overall.to_csv(args.output / 'allvarliga_storningar_terms.csv', index=False)

    print("\n=== Top Individual Risks in 'Allvarliga Störningar' Paragraphs ===")
    print(f"{'Risk':<35} {'Mentions':>10} {'Paragraphs':>10}")
    print("-" * 58)
    for _, row in terms_overall.head(20).iterrows():
        print(f"{row['term']:<35} {row['mentions']:>10,} {row['paragraphs']:>10,}")
    print()

    # Analyze risk categories
    logger.info("Analyzing risk categories...")
    overall = analyze_risk_categories(filtered, stemmed_category_dict)
    overall.to_csv(args.output / 'allvarliga_storningar_risks.csv', index=False)

    print("=== Top Risk Categories ===")
    print(overall.head(10).to_string(index=False))
    print()

    # Analyze by actor
    by_actor = None
    if 'actor_type' in filtered.columns:
        logger.info("Analyzing by actor_type...")
        # Normalize actor names
        filtered['actor_type'] = filtered['actor_type'].replace('länsstyrelse', 'lansstyrelse')

        by_actor = analyze_risk_categories(filtered, stemmed_category_dict, group_col='actor_type')
        by_actor.to_csv(args.output / 'allvarliga_storningar_by_actor.csv', index=False)

        terms_by_actor = analyze_individual_terms(filtered, stemmed_individual_dict, group_col='actor_type')
        terms_by_actor.to_csv(args.output / 'allvarliga_storningar_terms_by_actor.csv', index=False)

    # Analyze by wave
    by_wave = None
    if 'wave' in filtered.columns:
        logger.info("Analyzing by wave...")
        by_wave = analyze_risk_categories(filtered, stemmed_category_dict, group_col='wave')
        by_wave.to_csv(args.output / 'allvarliga_storningar_by_wave.csv', index=False)

        terms_by_wave = analyze_individual_terms(filtered, stemmed_individual_dict, group_col='wave')
        terms_by_wave.to_csv(args.output / 'allvarliga_storningar_terms_by_wave.csv', index=False)

    # Extract sample paragraphs
    logger.info(f"Extracting {args.sample_size} sample paragraphs...")
    sample = extract_sample_paragraphs(filtered, sample_size=args.sample_size)
    sample.to_csv(args.output / 'allvarliga_storningar_sample.csv', index=False)

    # Create visualizations
    if by_actor is not None and by_wave is not None:
        create_visualizations(overall, by_wave, by_actor, args.output)

    if len(terms_overall) > 0:
        create_term_visualization(terms_overall, args.output)

    # Summary
    print("=== Summary Statistics ===")
    print(f"Total paragraphs in corpus: {len(paragraphs):,}")
    print(f"Paragraphs with disruption phrases: {len(filtered):,} ({len(filtered)/len(paragraphs)*100:.1f}%)")
    if 'actor_type' in filtered.columns:
        print(f"By actor: {filtered['actor_type'].value_counts().to_dict()}")
    if 'wave' in filtered.columns:
        print(f"By wave: {filtered['wave'].value_counts().sort_index().to_dict()}")

    logger.info(f"All outputs saved to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
