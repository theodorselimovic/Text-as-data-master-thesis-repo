#!/usr/bin/env python3
"""
Allvarliga Störningar Risk Analysis (Stemmed Token-Based)

Analyzes which individual risks co-occur with serious disruption/consequence
phrases in RSA documents. This reveals how different actors frame risks in
terms of their potential to cause operational disruptions.

Uses pre-stemmed corpus for fast token matching (consistent with other BOW scripts).

Input:
    data/processed/bow_corpus_stemmed.parquet (sentence-level with tokens)

Output:
    results/01_bow_analysis/allvarliga_storningar/allvarliga_storningar_terms.csv
    results/01_bow_analysis/allvarliga_storningar/allvarliga_storningar_terms_by_actor.csv
    results/01_bow_analysis/allvarliga_storningar/allvarliga_storningar_terms_by_wave.csv
    results/01_bow_analysis/allvarliga_storningar/allvarliga_storningar_sample.csv

Usage:
    python allvarliga_storningar_analysis.py
    python allvarliga_storningar_analysis.py --corpus data/processed/bow_corpus_stemmed.parquet
    python allvarliga_storningar_analysis.py --verbose --sample-size 50

Requirements:
    pip install pandas pyarrow nltk matplotlib seaborn
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer

# Import centralized dictionary from scripts/dictionaries/
sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries import RISK_TERMS as RISK_DICTIONARY_INDIVIDUAL
from dictionaries.risk_translations import translate_term

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


# =============================================================================
# DISRUPTION PATTERNS (STEMMED)
# =============================================================================

# Phrases indicating serious disruptions/consequences to institutional operations
DISRUPTION_PHRASES_RAW = [
    # Core störning variants
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

    # Consequences
    'allvarliga konsekvenser',
    'allvarlig konsekvens',
    'svåra konsekvenser',
    'svår konsekvens',
    'omfattande konsekvenser',
    'omfattande skador',
    'stora skador',
    'allvarliga följder',

    # Functional failures
    'driftavbrott',
    'driftstörning',
    'driftstörningar',
    'funktionsbortfall',
    'avbrott i verksamheten',
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

def build_stemmed_risk_dict(stopwords: Set[str]) -> Dict[str, Set[str]]:
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

def tokens_contain_any(tokens, target_terms: Set[str]) -> bool:
    """Check if token list/array contains any of the target terms."""
    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()
    if not isinstance(tokens, (list, tuple)):
        return False
    token_set = set(tokens)
    return bool(token_set & target_terms)


def count_risk_terms(tokens, stemmed_dict: Dict[str, Set[str]]) -> Dict[str, int]:
    """Count individual risk terms using token matching."""
    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()
    if not isinstance(tokens, (list, tuple)):
        return {}

    results = {}
    for canonical, variants in stemmed_dict.items():
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
    """Group sentences by paragraph, aggregating tokens."""
    logger.info("Grouping sentences into paragraphs...")

    meta_cols = ['doc_id', 'paragraph_id', 'actor_type', 'year', 'municipality']
    available_meta = [col for col in meta_cols if col in df.columns]

    def concat_tokens(token_lists):
        all_tokens = []
        for tl in token_lists:
            if hasattr(tl, 'tolist'):
                tl = tl.tolist()
            if isinstance(tl, (list, tuple)):
                all_tokens.extend(tl)
        return all_tokens

    agg_dict = {'tokens': concat_tokens}
    for col in available_meta:
        if col not in ['doc_id', 'paragraph_id']:
            agg_dict[col] = 'first'

    if 'sentence_text' in df.columns:
        agg_dict['sentence_text'] = lambda x: ' '.join(x.astype(str))

    paragraphs = df.groupby(['doc_id', 'paragraph_id']).agg(agg_dict).reset_index()

    if 'sentence_text' in paragraphs.columns:
        paragraphs.rename(columns={'sentence_text': 'full_text'}, inplace=True)

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
        return tokens_contain_any(tokens, disruption_terms)

    mask = paragraphs['tokens'].apply(has_disruption)
    filtered = paragraphs[mask].copy()

    logger.info(f"  Found {len(filtered):,} paragraphs with disruption phrases")
    logger.info(f"  ({len(filtered)/len(paragraphs)*100:.1f}% of total paragraphs)")

    return filtered


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_risk_terms(
    paragraphs: pd.DataFrame,
    stemmed_dict: Dict[str, Set[str]],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Count individual risk terms in paragraphs using token matching."""
    if group_col is None:
        mention_totals = defaultdict(int)
        para_totals = defaultdict(int)
        char_totals = defaultdict(int)

        for _, row in paragraphs.iterrows():
            tokens = row['tokens']
            counts = count_risk_terms(tokens, stemmed_dict)
            text_len = len(row.get('full_text', '')) if 'full_text' in row else 0

            for term, count in counts.items():
                mention_totals[term] += count
                para_totals[term] += 1
                char_totals[term] += text_len

        rows = []
        for term in mention_totals.keys():
            rows.append({
                'term': term,
                'mentions': mention_totals[term],
                'paragraphs': para_totals[term],
                'characters': char_totals[term],
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
            char_totals = defaultdict(int)

            for _, row in group_df.iterrows():
                tokens = row['tokens']
                counts = count_risk_terms(tokens, stemmed_dict)
                text_len = len(row.get('full_text', '')) if 'full_text' in row else 0

                for term, count in counts.items():
                    mention_totals[term] += count
                    para_totals[term] += 1
                    char_totals[term] += text_len

            for term in mention_totals.keys():
                results.append({
                    group_col: group_val,
                    'term': term,
                    'mentions': mention_totals[term],
                    'paragraphs': para_totals[term],
                    'characters': char_totals[term],
                })

        df = pd.DataFrame(results)
        if len(df) == 0:
            return df

        df['pct_mentions'] = df.groupby(group_col)['mentions'].transform(
            lambda x: (x / x.sum() * 100).round(2)
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
    'Municipality': '#e41a1c',
    'Prefecture': '#377eb8',
    'MSB': '#4daf4a',
}


def create_term_visualization(
    terms_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 25,
) -> None:
    """Create visualization for individual risk terms."""
    logger.info("Creating term-level visualization...")

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(10, 10))

    top_terms = terms_df.nlargest(top_n, 'mentions').iloc[::-1].copy()
    top_terms['term_en'] = top_terms['term'].apply(translate_term)

    bars = ax.barh(top_terms['term_en'], top_terms['mentions'],
                   color='#377eb8', edgecolor='none')

    for bar, val in zip(bars, top_terms['mentions']):
        ax.text(val + 20, bar.get_y() + bar.get_height()/2, f'{val:,}',
                va='center', ha='left', fontsize=8)

    ax.set_xlabel('Mentions', fontsize=11)
    ax.set_title(f'Top {top_n} Risks in Disruption/Consequence Paragraphs',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, top_terms['mentions'].max() * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_terms.png', dpi=150)
    fig.savefig(output_dir / 'allvarliga_storningar_terms.pdf')
    plt.close(fig)

    logger.info("  Saved term-level chart")


def create_mentions_vs_text_visualization(
    terms_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """Create side-by-side comparison of mentions vs text devoted (characters)."""
    logger.info("Creating mentions vs text devoted visualization...")

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    # Left plot: Top N by mentions, ordered by mentions
    top_by_mentions = terms_df.nlargest(top_n, 'mentions').sort_values('mentions', ascending=True).copy()
    top_by_mentions['term_en'] = top_by_mentions['term'].apply(translate_term)
    bars1 = axes[0].barh(top_by_mentions['term_en'], top_by_mentions['mentions'],
                         color='#377eb8', edgecolor='none')
    for bar, val in zip(bars1, top_by_mentions['mentions']):
        axes[0].text(val + 20, bar.get_y() + bar.get_height()/2, f'{val:,}',
                     va='center', ha='left', fontsize=9)
    axes[0].set_xlabel('Mentions (term occurrences)', fontsize=11)
    axes[0].set_title('Mentions', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, top_by_mentions['mentions'].max() * 1.2)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Right plot: Top N by characters, ordered by characters
    top_by_chars = terms_df.nlargest(top_n, 'characters').sort_values('characters', ascending=True).copy()
    top_by_chars['term_en'] = top_by_chars['term'].apply(translate_term)
    bars2 = axes[1].barh(top_by_chars['term_en'], top_by_chars['characters'],
                         color='#e41a1c', edgecolor='none')
    for bar, val in zip(bars2, top_by_chars['characters']):
        axes[1].text(val + 5000, bar.get_y() + bar.get_height()/2, f'{val:,}',
                     va='center', ha='left', fontsize=9)
    axes[1].set_xlabel('Characters (text devoted)', fontsize=11)
    axes[1].set_title('Text Devoted', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, top_by_chars['characters'].max() * 1.15)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig.suptitle('Mentions vs Text Devoted in Disruption/Consequence Contexts',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_mentions_vs_text.png', dpi=150)
    fig.savefig(output_dir / 'allvarliga_storningar_mentions_vs_text.pdf')
    plt.close(fig)

    logger.info("  Saved mentions vs text chart")


def create_terms_by_actor_visualization(
    terms_by_actor: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """Create visualization comparing top risks by actor type."""
    logger.info("Creating terms by actor visualization...")

    # Get top N terms overall
    top_terms = terms_by_actor.groupby('term')['mentions'].sum().nlargest(top_n).index.tolist()

    # Filter to top terms
    plot_data = terms_by_actor[terms_by_actor['term'].isin(top_terms)].copy()
    plot_data['actor_label'] = plot_data['actor_type'].map(ACTOR_LABELS)

    # Pivot for grouped bar chart
    pivot = plot_data.pivot(index='term', columns='actor_label', values='mentions').fillna(0)

    # Order by total mentions
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=True).drop(columns='total')

    # Translate index to English
    pivot.index = [translate_term(t) for t in pivot.index]

    # Reorder columns
    col_order = ['Municipality', 'Prefecture', 'MSB']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(12, 8))

    pivot.plot(kind='barh', ax=ax, width=0.8,
               color=[ACTOR_COLORS[c] for c in pivot.columns])

    ax.set_xlabel('Mentions', fontsize=11)
    ax.set_ylabel('')
    ax.set_title(f'Top {top_n} Risks by Actor Type in Disruption Contexts',
                 fontsize=12, fontweight='bold')
    ax.legend(title='Actor type', loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_terms_by_actor.png', dpi=150)
    fig.savefig(output_dir / 'allvarliga_storningar_terms_by_actor.pdf')
    plt.close(fig)

    logger.info("  Saved terms by actor chart")


def create_terms_over_time_visualization(
    terms_by_wave: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """Create visualization showing top risks over time."""
    logger.info("Creating terms over time visualization...")

    # Get top N terms overall
    top_terms = terms_by_wave.groupby('term')['mentions'].sum().nlargest(top_n).index.tolist()

    # Filter and pivot
    plot_data = terms_by_wave[terms_by_wave['term'].isin(top_terms)].copy()
    pivot = plot_data.pivot(index='wave', columns='term', values='mentions').fillna(0)
    pivot.index = [WAVE_LABELS.get(w, str(w)) for w in pivot.index]
    # Translate column names to English
    pivot.columns = [translate_term(t) for t in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 7))

    pivot.plot(kind='line', marker='o', markersize=8, linewidth=2, ax=ax,
               color=sns.color_palette("husl", n_colors=len(pivot.columns)))

    ax.set_xlabel('Time period', fontsize=11)
    ax.set_ylabel('Mentions', fontsize=11)
    ax.set_title(f'Top {top_n} Risks Over Time in Disruption Contexts',
                 fontsize=12, fontweight='bold')
    ax.legend(title='Risk', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_terms_over_time.png', dpi=150)
    fig.savefig(output_dir / 'allvarliga_storningar_terms_over_time.pdf')
    plt.close(fig)

    logger.info("  Saved terms over time chart")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Analyze risks in disruption/consequence contexts',
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
    stemmed_risk_dict = build_stemmed_risk_dict(stopwords)

    logger.info(f"  Disruption terms: {len(disruption_terms)}")
    logger.info(f"  Risk terms: {len(stemmed_risk_dict)} risks")

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
    terms_overall = analyze_risk_terms(filtered, stemmed_risk_dict)
    terms_overall.to_csv(args.output / 'allvarliga_storningar_terms.csv', index=False)

    print("\n=== Top Individual Risks in Disruption/Consequence Paragraphs ===")
    print(f"{'Risk':<35} {'Mentions':>10} {'Paragraphs':>10}")
    print("-" * 58)
    for _, row in terms_overall.head(20).iterrows():
        print(f"{row['term']:<35} {row['mentions']:>10,} {row['paragraphs']:>10,}")
    print()

    # Analyze by actor
    terms_by_actor = None
    if 'actor_type' in filtered.columns:
        logger.info("Analyzing by actor_type...")
        filtered['actor_type'] = filtered['actor_type'].replace('länsstyrelse', 'lansstyrelse')

        terms_by_actor = analyze_risk_terms(filtered, stemmed_risk_dict, group_col='actor_type')
        terms_by_actor.to_csv(args.output / 'allvarliga_storningar_terms_by_actor.csv', index=False)

    # Analyze by wave
    terms_by_wave = None
    if 'wave' in filtered.columns:
        logger.info("Analyzing by wave...")
        terms_by_wave = analyze_risk_terms(filtered, stemmed_risk_dict, group_col='wave')
        terms_by_wave.to_csv(args.output / 'allvarliga_storningar_terms_by_wave.csv', index=False)

    # Extract sample paragraphs
    logger.info(f"Extracting {args.sample_size} sample paragraphs...")
    sample = extract_sample_paragraphs(filtered, sample_size=args.sample_size)
    sample.to_csv(args.output / 'allvarliga_storningar_sample.csv', index=False)

    # Create visualizations
    logger.info("Creating visualizations...")
    if len(terms_overall) > 0:
        create_term_visualization(terms_overall, args.output)
        create_mentions_vs_text_visualization(terms_overall, args.output)

    if terms_by_actor is not None and len(terms_by_actor) > 0:
        create_terms_by_actor_visualization(terms_by_actor, args.output)

    if terms_by_wave is not None and len(terms_by_wave) > 0:
        create_terms_over_time_visualization(terms_by_wave, args.output)

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
