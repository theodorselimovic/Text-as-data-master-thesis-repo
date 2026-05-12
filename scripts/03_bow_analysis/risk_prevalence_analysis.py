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
from scipy import stats
from scipy.optimize import curve_fit
from nltk.stem.snowball import SnowballStemmer

# Import centralized dictionary from scripts/dictionaries/
sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries import RISK_TERMS as RISK_DICTIONARY_INDIVIDUAL
from dictionaries.risk_translations import translate_term, translate_actor

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
            'entity': first.get('municipality', 'unknown'),
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

    # Build reverse mapping: stemmed term -> canonical
    term_to_canonical = {term: info['canonical'] for term, info in term_info.items()}

    # Track per canonical risk (avoids double-counting paragraphs with multiple variants)
    canonical_metrics = defaultdict(lambda: {
        'mentions': 0,
        'paragraphs': set(),  # Track unique (doc_id, para_id) tuples
        'documents': set(),
        'variants': set(),
    })

    # Also track paragraph char counts for later summation
    paragraph_chars = {}  # (doc_id, para_id) -> char_count

    total_paragraphs = len(paragraphs_df)

    for idx, row in paragraphs_df.iterrows():
        if verbose and idx % 20000 == 0:
            logger.info(f"  Processing paragraph {idx:,}/{total_paragraphs:,}")

        tokens = row['tokens']
        if not tokens:
            continue

        token_counts = Counter(tokens)
        doc_id = row['doc_id']
        para_id = row['paragraph_id']
        char_count = row['char_count']

        # Store paragraph char count
        paragraph_chars[(doc_id, para_id)] = char_count

        # Check each dictionary term and aggregate by canonical form
        for term, count in token_counts.items():
            if term in term_to_canonical and count > 0:
                canonical = term_to_canonical[term]
                canonical_metrics[canonical]['mentions'] += count
                canonical_metrics[canonical]['paragraphs'].add((doc_id, para_id))
                canonical_metrics[canonical]['documents'].add(doc_id)
                canonical_metrics[canonical]['variants'].add(term)

    # Convert to DataFrame with proper character counts (no double-counting)
    rows = []
    for canonical, m in canonical_metrics.items():
        # Sum characters from unique paragraphs only
        total_chars = sum(paragraph_chars.get(p, 0) for p in m['paragraphs'])

        rows.append({
            'risk': canonical,
            'mentions': m['mentions'],
            'paragraphs': len(m['paragraphs']),
            'characters': total_chars,
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

    # Build reverse mapping
    term_to_canonical = {term: info['canonical'] for term, info in term_info.items()}

    groups = paragraphs_df[group_col].dropna().unique()
    all_results = []

    for group_val in groups:
        subset = paragraphs_df[paragraphs_df[group_col] == group_val]
        n_docs = subset['doc_id'].nunique()

        # Track per canonical risk (avoids double-counting)
        canonical_metrics = defaultdict(lambda: {
            'mentions': 0,
            'paragraphs': set(),
            'documents': set(),
        })
        paragraph_chars = {}

        for _, row in subset.iterrows():
            tokens = row['tokens']
            if not tokens:
                continue

            token_counts = Counter(tokens)
            doc_id = row['doc_id']
            para_id = row['paragraph_id']
            char_count = row['char_count']

            paragraph_chars[(doc_id, para_id)] = char_count

            for term, count in token_counts.items():
                if term in term_to_canonical and count > 0:
                    canonical = term_to_canonical[term]
                    canonical_metrics[canonical]['mentions'] += count
                    canonical_metrics[canonical]['paragraphs'].add((doc_id, para_id))
                    canonical_metrics[canonical]['documents'].add(doc_id)

        for canonical, m in canonical_metrics.items():
            total_chars = sum(paragraph_chars.get(p, 0) for p in m['paragraphs'])
            all_results.append({
                group_col: group_val,
                'risk': canonical,
                'mentions': m['mentions'],
                'paragraphs': len(m['paragraphs']),
                'characters': total_chars,
                'documents': len(m['documents']),
                'total_docs_in_group': n_docs,
                'mentions_per_doc': round(m['mentions'] / n_docs, 2) if n_docs > 0 else 0,
                'chars_per_doc': round(total_chars / n_docs, 1) if n_docs > 0 else 0,
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
    top_df['risk_en'] = top_df['risk'].apply(translate_term)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    # Left: mentions
    ax1 = axes[0]
    ax1.barh(top_df['risk_en'], top_df['mentions'], color='#377eb8')
    ax1.set_xlabel('Total mentions')
    ax1.set_title('By Mention Count', fontweight='bold')
    ax1.invert_yaxis()

    # Right: characters (in thousands)
    ax2 = axes[1]
    # Sort by characters for this panel
    top_by_chars = prevalence_df.nlargest(top_n, 'characters').copy()
    top_by_chars['risk_en'] = top_by_chars['risk'].apply(translate_term)
    ax2.barh(top_by_chars['risk_en'], top_by_chars['characters'] / 1000, color='#4daf4a')
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
        subset = by_actor_df[by_actor_df['actor_type'] == actor].nlargest(top_n, 'mentions_per_doc').copy()
        subset['risk_en'] = subset['risk'].apply(translate_term)
        ax.barh(subset['risk_en'], subset['mentions_per_doc'], color=ACTOR_COLORS[actor])
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
# Distribution Analysis
# =============================================================================

def zipf_func(rank: np.ndarray, c: float, alpha: float) -> np.ndarray:
    """Zipf's law: f(r) = c / r^alpha"""
    return c / np.power(rank, alpha)


def exponential_func(rank: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exponential decay: f(r) = a * exp(-b * r)"""
    return a * np.exp(-b * rank)


def power_exponential_func(rank: np.ndarray, c: float, alpha: float, b: float) -> np.ndarray:
    """Combined power-exponential: f(r) = c / r^alpha * exp(-b*r)"""
    return c / np.power(rank, alpha) * np.exp(-b * rank)


def fit_distributions(ranks: np.ndarray, values: np.ndarray) -> dict:
    """Fit Zipf, exponential, and power-exponential distributions."""
    results = {}

    # Zipf's Law
    try:
        popt, _ = curve_fit(zipf_func, ranks, values, p0=[values[0], 1.0], maxfev=5000)
        fitted = zipf_func(ranks, *popt)
        r2 = 1 - np.sum((values - fitted)**2) / np.sum((values - np.mean(values))**2)
        results['zipf'] = {
            'params': {'c': popt[0], 'alpha': popt[1]},
            'r2': r2,
            'fitted': fitted,
            'aic': len(ranks) * np.log(np.mean((values - fitted)**2)) + 2 * 2
        }
    except Exception as e:
        logger.warning(f"Zipf fit failed: {e}")

    # Exponential
    try:
        popt, _ = curve_fit(exponential_func, ranks, values, p0=[values[0], 0.1], maxfev=5000)
        fitted = exponential_func(ranks, *popt)
        r2 = 1 - np.sum((values - fitted)**2) / np.sum((values - np.mean(values))**2)
        results['exponential'] = {
            'params': {'a': popt[0], 'b': popt[1]},
            'r2': r2,
            'fitted': fitted,
            'aic': len(ranks) * np.log(np.mean((values - fitted)**2)) + 2 * 2
        }
    except Exception as e:
        logger.warning(f"Exponential fit failed: {e}")

    # Power-Exponential
    try:
        popt, _ = curve_fit(
            power_exponential_func, ranks, values,
            p0=[values[0], 0.8, 0.01],
            bounds=([0, 0, 0], [np.inf, 5, 1]),
            maxfev=5000
        )
        fitted = power_exponential_func(ranks, *popt)
        r2 = 1 - np.sum((values - fitted)**2) / np.sum((values - np.mean(values))**2)
        results['power_exponential'] = {
            'params': {'c': popt[0], 'alpha': popt[1], 'b': popt[2]},
            'r2': r2,
            'fitted': fitted,
            'aic': len(ranks) * np.log(np.mean((values - fitted)**2)) + 2 * 3
        }
    except Exception as e:
        logger.warning(f"Power-Exponential fit failed: {e}")

    return results


def plot_full_distribution(prevalence_df: pd.DataFrame, output_dir: Path) -> None:
    """Create full distribution graph with all risks ranked by mentions."""
    df_sorted = prevalence_df.sort_values('mentions', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Bars - single color
    ax.bar(range(len(df_sorted)), df_sorted['mentions'], color='#377eb8', alpha=0.9)

    # Bold axes
    ax.set_xlabel('Risk (ranked by mentions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total mentions', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Risk Mentions Across Corpus', fontsize=16, fontweight='bold')

    # Remove x-axis tick labels, keep tick marks
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['1', '25', '50', '75', '100'], fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Remove grid lines
    ax.grid(False)
    ax.set_facecolor('white')

    # Bold spines, remove top
    ax.spines['top'].set_visible(False)
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.5)

    # Cumulative percentage lines on secondary axis
    ax2 = ax.twinx()
    cumsum_mentions = df_sorted['mentions'].cumsum() / df_sorted['mentions'].sum() * 100
    cumsum_chars = df_sorted['characters'].cumsum() / df_sorted['characters'].sum() * 100

    ax2.plot(range(len(df_sorted)), cumsum_mentions, 'k--', linewidth=2.5, label='Mentions')
    ax2.plot(range(len(df_sorted)), cumsum_chars, color='#e41a1c', linestyle='--',
             linewidth=2.5, label='Text devoted')
    ax2.set_ylabel('Cumulative percentage', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='y', labelsize=12)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    ax2.grid(False)
    ax2.legend(loc='center right', fontsize=11, framealpha=0.9)

    # Bold spines for secondary axis, remove top
    ax2.spines['top'].set_visible(False)
    for spine in ['bottom', 'left', 'right']:
        ax2.spines[spine].set_linewidth(1.5)

    # Mark 80% threshold for mentions
    idx_80 = (cumsum_mentions >= 80).idxmax()
    ax2.axvline(idx_80, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax2.annotate(f'80% at rank {idx_80+1}', xy=(idx_80, 80), xytext=(idx_80+10, 85),
                 fontsize=11, fontweight='bold', arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_distribution_full.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'risk_distribution_full.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Saved: risk_distribution_full.png/pdf")


def plot_zipf_analysis(prevalence_df: pd.DataFrame, fit_results: dict, output_dir: Path) -> None:
    """Create log-log plot with distribution fits."""
    plt.style.use('seaborn-v0_8-whitegrid')
    df_sorted = prevalence_df.sort_values('mentions', ascending=False).reset_index(drop=True)
    ranks = np.arange(1, len(df_sorted) + 1)
    values = df_sorted['mentions'].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Log-log with fits
    ax1 = axes[0]
    ax1.scatter(ranks, values, s=40, alpha=0.7, color='#377eb8', label='Observed', zorder=3)

    colors = {'zipf': '#e41a1c', 'exponential': '#4daf4a', 'power_exponential': '#984ea3'}
    labels = {'zipf': "Zipf's Law", 'exponential': 'Exponential', 'power_exponential': 'Power-Exponential'}

    for name, result in fit_results.items():
        ax1.plot(ranks, result['fitted'], color=colors.get(name, 'gray'),
                 linewidth=2, label=f"{labels.get(name, name)} (R²={result['r2']:.3f})")

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Mentions', fontsize=12)
    ax1.set_title('Log-Log Plot: Rank vs Mentions', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Residuals for best fit
    ax2 = axes[1]
    best_fit = max(fit_results.items(), key=lambda x: x[1]['r2'])
    residuals = values - best_fit[1]['fitted']

    ax2.scatter(ranks, residuals, s=30, alpha=0.6, color='#377eb8')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Rank', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title(f'Residuals: {labels.get(best_fit[0], best_fit[0])} Fit', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_distribution_zipf.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'risk_distribution_zipf.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Saved: risk_distribution_zipf.png/pdf")


def plot_rank_correlation(prevalence_df: pd.DataFrame, output_dir: Path) -> dict:
    """Create scatter plots and compute rank correlations."""
    df = prevalence_df.copy()
    df['mention_rank'] = df['mentions'].rank(ascending=False)
    df['char_rank'] = df['characters'].rank(ascending=False)

    # Compute correlations
    spearman_r, spearman_p = stats.spearmanr(df['mention_rank'], df['char_rank'])
    kendall_tau, kendall_p = stats.kendalltau(df['mention_rank'], df['char_rank'])

    correlations = {
        'spearman': {'r': spearman_r, 'p': spearman_p},
        'kendall': {'tau': kendall_tau, 'p': kendall_p},
    }

    # Plot 1: Rank vs Rank
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.scatter(df['mention_rank'], df['char_rank'], s=50, alpha=0.6, color='#377eb8')
    max_rank = max(df['mention_rank'].max(), df['char_rank'].max())
    ax1.plot([1, max_rank], [1, max_rank], 'r--', linewidth=2, label='Perfect correlation')

    # Invert axes so rank 1 is at top-left
    ax1.invert_xaxis()
    ax1.invert_yaxis()

    # Annotate outliers
    df['rank_diff'] = abs(df['mention_rank'] - df['char_rank'])
    outliers = df.nlargest(5, 'rank_diff')
    for _, row in outliers.iterrows():
        ax1.annotate(translate_term(row['risk']), xy=(row['mention_rank'], row['char_rank']),
                     xytext=(-5, -5), textcoords='offset points', fontsize=9, alpha=0.8)

    ax1.set_xlabel('Rank by Mentions', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Rank by Text Devoted', fontsize=14, fontweight='bold')
    ax1.set_title(f'Rank Correlation: Mentions vs Text Devoted\nSpearman r={spearman_r:.3f}, p={spearman_p:.2e}',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_rank_correlation.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'risk_rank_correlation.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Saved: risk_rank_correlation.png/pdf")

    # Plot 2: Characters per mention vs mentions
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.scatter(df['mentions'], df['chars_per_mention'], s=50, alpha=0.6, color='#377eb8')

    # Add trend line
    z = np.polyfit(df['mentions'], df['chars_per_mention'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['mentions'].min(), df['mentions'].max(), 100)
    ax2.plot(x_line, p(x_line), 'r--', linewidth=2)

    # Annotate outliers (highest and lowest chars_per_mention)
    top3 = df.nlargest(3, 'chars_per_mention')
    bottom3 = df.nsmallest(3, 'chars_per_mention')
    for _, row in pd.concat([top3, bottom3]).iterrows():
        ax2.annotate(translate_term(row['risk']), xy=(row['mentions'], row['chars_per_mention']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)

    ax2.set_xlabel('Mentions', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Characters per mention', fontsize=14, fontweight='bold')
    ax2.set_title('Text Density: Do frequent risks get more text per mention?',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_text_density.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'risk_text_density.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Saved: risk_text_density.png/pdf")

    return correlations


def print_distribution_analysis(prevalence_df: pd.DataFrame, fit_results: dict, correlations: dict) -> None:
    """Print distribution analysis results."""
    print("\n" + "=" * 70)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Concentration
    df_sorted = prevalence_df.sort_values('mentions', ascending=False)
    cumsum = df_sorted['mentions'].cumsum() / df_sorted['mentions'].sum()
    top10_pct = cumsum.iloc[9] * 100 if len(cumsum) >= 10 else cumsum.iloc[-1] * 100
    idx_50 = (cumsum >= 0.5).idxmax() + 1
    idx_80 = (cumsum >= 0.8).idxmax() + 1

    print(f"\nConcentration:")
    print(f"  Top 10 risks: {top10_pct:.1f}% of all mentions")
    print(f"  Top {idx_50} risks: 50% of all mentions")
    print(f"  Top {idx_80} risks: 80% of all mentions")

    # Distribution fits
    print(f"\nDistribution Fits:")
    for name, result in sorted(fit_results.items(), key=lambda x: -x[1]['r2']):
        params_str = ', '.join(f"{k}={v:.3f}" for k, v in result['params'].items())
        print(f"  {name}: R²={result['r2']:.4f}, AIC={result['aic']:.1f} ({params_str})")

    best = max(fit_results.items(), key=lambda x: x[1]['r2'])
    print(f"\n  Best fit: {best[0]} (R²={best[1]['r2']:.4f})")

    if 'zipf' in fit_results:
        alpha = fit_results['zipf']['params']['alpha']
        if alpha < 1:
            print(f"  Zipf alpha={alpha:.3f} (flatter than classic Zipf=1: more evenly distributed)")
        else:
            print(f"  Zipf alpha={alpha:.3f} (steeper than classic Zipf=1: more concentrated)")

    # Rank correlation
    print(f"\nRank Correlation (mentions vs text devoted):")
    print(f"  Spearman r = {correlations['spearman']['r']:.4f} (p={correlations['spearman']['p']:.2e})")
    print(f"  Kendall tau = {correlations['kendall']['tau']:.4f} (p={correlations['kendall']['p']:.2e})")


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

    # Distribution analysis
    logger.info("\nRunning distribution analysis...")
    df_sorted = prevalence_df.sort_values('mentions', ascending=False).reset_index(drop=True)
    ranks = np.arange(1, len(df_sorted) + 1)
    values = df_sorted['mentions'].values
    fit_results = fit_distributions(ranks, values)

    plot_full_distribution(prevalence_df, args.output)
    plot_zipf_analysis(prevalence_df, fit_results, args.output)
    correlations = plot_rank_correlation(prevalence_df, args.output)

    print_distribution_analysis(prevalence_df, fit_results, correlations)

    # Save distribution fit results
    fit_df = pd.DataFrame([
        {'distribution': name, 'r2': r['r2'], 'aic': r['aic'],
         **{f'param_{k}': v for k, v in r['params'].items()}}
        for name, r in fit_results.items()
    ])
    fit_df.to_csv(args.output / 'distribution_fits.csv', index=False)
    logger.info("Saved: distribution_fits.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
