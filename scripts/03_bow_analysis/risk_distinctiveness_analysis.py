#!/usr/bin/env python3
"""
Risk Distinctiveness Analysis

Identifies which risk terms are statistically over- or under-represented for each
actor type (kommun, lansstyrelse, MCF) using the Monroe et al. (2008) "Fightin' Words"
method with informative Dirichlet priors.

The method computes log-odds ratios with shrinkage:
- Handles different corpus sizes gracefully
- Shrinks estimates for rare words toward pooled background frequency
- Produces z-scores used as a ranking heuristic (not formal statistical inference)

Formula:
    log-odds = log((y1 + α_i) / (n1 + α_0 - y1 - α_i))
             - log((y2 + α_i) / (n2 + α_0 - y2 - α_i))

    z-score = log-odds / sqrt(1/(y1 + α_i) + 1/(y2 + α_i))

where:
    y1, y2 = word count in corpus 1 and 2
    n1, n2 = total words in corpus 1 and 2
    α_i = (y1 + y2) × prior_weight  (pooled COUNT, not frequency)
    α_0 = (n1 + n2) × prior_weight  (total prior strength)

Reference:
    Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words:
    Lexical feature selection and evaluation for identifying the content of
    political conflict. Political Analysis, 16(4), 372-403.

Output:
    - distinctiveness_<actor1>_vs_<actor2>.csv: Pairwise comparison results
    - viz_distinctiveness_<actor1>_vs_<actor2>.png: Bar charts of distinctive terms

Usage:
    python risk_distinctiveness_analysis.py \\
        --input results/01_bow_analysis/term_matrices/term_document_matrix.csv \\
        --output results/01_bow_analysis/distinctiveness/

    # By wave (temporal breakdown)
    python risk_distinctiveness_analysis.py \\
        --input results/01_bow_analysis/term_matrices/term_document_matrix.csv \\
        --output results/01_bow_analysis/distinctiveness/ \\
        --by-wave

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-03-19
"""

import argparse
import logging
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import translations
sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries.risk_translations import translate_term


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

ACTOR_DISPLAY_NAMES = {
    'kommun': 'Municipality',
    'lansstyrelse': 'Prefecture',
    'MCF': 'MSB',
}

# Standard actor colors (consistent across all visualizations)
ACTOR_COLORS = {
    'kommun': '#e41a1c',        # Red
    'lansstyrelse': '#377eb8',  # Blue
    'MCF': '#4daf4a',           # Green
}

# Display order: MSB (left), Prefecture (middle), Municipality (right)
ACTOR_ORDER = ['MCF', 'lansstyrelse', 'kommun']


def map_year_to_wave(year) -> Optional[int]:
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
# Data Loading from Matrix
# =============================================================================

METADATA_COLS = ['file', 'actor', 'entity', 'year', 'wave']


def load_matrix(input_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load term-document matrix.

    Returns
    -------
    tuple
        (df, risk_cols) where risk_cols are the canonical risk column names
    """
    df = pd.read_csv(input_path)
    risk_cols = [c for c in df.columns if c not in METADATA_COLS]
    return df, risk_cols


def aggregate_counts_from_matrix(
    df: pd.DataFrame,
    risk_cols: List[str],
) -> Tuple[Dict[str, Counter], Dict[str, int]]:
    """
    Aggregate risk counts by actor from the term-document matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Term-document matrix with 'actor' column and risk count columns
    risk_cols : list
        Names of risk count columns

    Returns
    -------
    tuple
        (actor_counts, actor_totals)
        - actor_counts: {actor: Counter({canonical_risk: count})}
        - actor_totals: {actor: total_risk_mentions}
    """
    logger.info("Aggregating counts by actor from matrix...")

    actor_counts = {}
    actor_totals = {}

    for actor in df['actor'].unique():
        actor_df = df[df['actor'] == actor]

        # Sum counts for each risk across all documents for this actor
        counts = Counter()
        for risk in risk_cols:
            total = actor_df[risk].sum()
            if total > 0:
                counts[risk] = int(total)

        actor_counts[actor] = counts
        actor_totals[actor] = sum(counts.values())

        n_risks = len([c for c in counts.values() if c > 0])
        logger.info(f"  {actor}: {actor_totals[actor]:,} mentions, {n_risks} unique risks")

    return actor_counts, actor_totals


def aggregate_counts_by_wave_from_matrix(
    df: pd.DataFrame,
    risk_cols: List[str],
) -> Dict[int, Tuple[Dict[str, Counter], Dict[str, int]]]:
    """
    Aggregate risk counts by actor, grouped by wave.

    Returns
    -------
    dict
        {wave: (actor_counts, actor_totals)}
    """
    logger.info("Aggregating counts by actor and wave from matrix...")

    results = {}

    for wave in sorted(df['wave'].dropna().unique()):
        wave_df = df[df['wave'] == wave]
        logger.info(f"\nWave {int(wave)} ({WAVE_LABELS.get(int(wave), '?')}): {len(wave_df)} documents")

        actor_counts, actor_totals = aggregate_counts_from_matrix(wave_df, risk_cols)
        results[int(wave)] = (actor_counts, actor_totals)

    return results


# =============================================================================
# Monroe et al. (2008) "Fightin' Words" Implementation
# =============================================================================

def compute_log_odds_ratio(
    counts1: Counter,
    counts2: Counter,
    total1: int,
    total2: int,
    prior_weight: float = 0.01,
) -> pd.DataFrame:
    """
    Compute log-odds ratios with informative Dirichlet prior (Monroe et al. 2008).

    The prior for each word is proportional to its pooled COUNT (not frequency),
    scaled by prior_weight. This provides meaningful shrinkage: rare words are
    pulled toward the pooled background, while common words retain their signal.

    Parameters
    ----------
    counts1, counts2 : Counter
        Word counts for corpus 1 and 2
    total1, total2 : int
        Total word counts for corpus 1 and 2
    prior_weight : float
        Controls shrinkage strength (default: 0.01). Acts as if we've observed
        an additional pseudo-corpus of size (total1 + total2) * prior_weight
        with the pooled distribution. Higher = more shrinkage toward background.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: term, count1, count2, log_odds, z_score
        Positive z_score means term is distinctive of corpus 1.

    Notes
    -----
    The key insight from Monroe et al. is that α_i (the prior for word i) should
    be proportional to the pooled COUNT, not frequency. With prior_weight=0.01:
    - A word appearing 100 times total gets α_i = 1.0
    - A word appearing 10 times total gets α_i = 0.1

    This ensures rare words shrink toward background while common words retain
    their discriminative signal.
    """
    # Get all terms from both corpora
    all_terms = set(counts1.keys()) | set(counts2.keys())

    if not all_terms:
        return pd.DataFrame(columns=['term', 'count1', 'count2', 'log_odds', 'z_score'])

    # Total prior strength = pooled corpus size × prior_weight
    alpha_0 = (total1 + total2) * prior_weight

    rows = []

    for term in all_terms:
        y1 = counts1.get(term, 0)
        y2 = counts2.get(term, 0)

        # Prior for this term = pooled COUNT × prior_weight
        # This is the key fix: use counts, not frequencies
        alpha_i = (y1 + y2) * prior_weight

        # Ensure minimum prior to avoid numerical issues
        alpha_i = max(alpha_i, 1e-10)

        # Log-odds ratio with prior (Monroe et al. Equation 16)
        # log((y1 + α_i) / (n1 + α_0 - y1 - α_i)) - log((y2 + α_i) / (n2 + α_0 - y2 - α_i))
        numerator1 = y1 + alpha_i
        denominator1 = total1 + alpha_0 - y1 - alpha_i
        numerator2 = y2 + alpha_i
        denominator2 = total2 + alpha_0 - y2 - alpha_i

        # Ensure positive denominators
        denominator1 = max(denominator1, 1e-10)
        denominator2 = max(denominator2, 1e-10)

        log_odds = np.log(numerator1 / denominator1) - np.log(numerator2 / denominator2)

        # Variance approximation (Monroe et al. Equation 22)
        variance = 1 / (y1 + alpha_i) + 1 / (y2 + alpha_i)
        z_score = log_odds / np.sqrt(variance)

        rows.append({
            'term': term,
            'count1': y1,
            'count2': y2,
            'log_odds': log_odds,
            'z_score': z_score,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values('z_score', ascending=False).reset_index(drop=True)

    return result


def compare_actors(
    actor_counts: Dict[str, Counter],
    actor_totals: Dict[str, int],
    actor1: str,
    actor2: str,
    prior_weight: float = 0.01,
) -> pd.DataFrame:
    """
    Compare two actors using Fightin' Words method.

    Parameters
    ----------
    actor_counts : dict
        {actor: Counter({term: count})}
    actor_totals : dict
        {actor: total_count}
    actor1, actor2 : str
        Actor names to compare
    prior_weight : float
        Controls shrinkage strength (default: 0.01)

    Returns
    -------
    pd.DataFrame
        Comparison results with positive z_score = distinctive of actor1
    """
    counts1 = actor_counts.get(actor1, Counter())
    counts2 = actor_counts.get(actor2, Counter())
    total1 = actor_totals.get(actor1, 0)
    total2 = actor_totals.get(actor2, 0)

    if total1 == 0 or total2 == 0:
        logger.warning(f"Empty corpus for comparison: {actor1}={total1}, {actor2}={total2}")
        return pd.DataFrame()

    result = compute_log_odds_ratio(counts1, counts2, total1, total2, prior_weight)

    # Add metadata columns
    result['actor1'] = actor1
    result['actor2'] = actor2
    result['actor1_total'] = total1
    result['actor2_total'] = total2

    return result


# =============================================================================
# Visualization
# =============================================================================

def plot_distinctive_terms(
    comparison_df: pd.DataFrame,
    actor1: str,
    actor2: str,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """
    Create two-panel bar chart showing top distinctive terms for each actor.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from compare_actors()
    actor1, actor2 : str
        Actor names
    output_path : Path
        Where to save the plot
    top_n : int
        Number of top terms to show per actor
    """
    if comparison_df.empty:
        logger.warning("Empty comparison, skipping plot")
        return

    # Get display names
    name1 = ACTOR_DISPLAY_NAMES.get(actor1, actor1)
    name2 = ACTOR_DISPLAY_NAMES.get(actor2, actor2)

    # Get top terms for each actor
    top_actor1 = comparison_df.nlargest(top_n, 'z_score')
    top_actor2 = comparison_df.nsmallest(top_n, 'z_score')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    color1 = ACTOR_COLORS.get(actor1, '#666666')
    color2 = ACTOR_COLORS.get(actor2, '#666666')

    # Left panel: Actor 1 distinctive terms
    if len(top_actor1) > 0:
        ax1.barh(
            range(len(top_actor1)),
            top_actor1['z_score'],
            color=color1,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.8,
        )
        ax1.set_yticks(range(len(top_actor1)))
        ax1.set_yticklabels([translate_term(t) for t in top_actor1['term']], fontsize=10)
        ax1.invert_yaxis()
    ax1.set_xlabel('Z-score (log-odds ratio)', fontsize=11)
    ax1.set_title(f'Distinctive of {name1}', fontsize=12, fontweight='bold')

    # Right panel: Actor 2 distinctive terms (flip sign for display)
    if len(top_actor2) > 0:
        z_scores_flipped = -top_actor2['z_score']
        ax2.barh(
            range(len(top_actor2)),
            z_scores_flipped,
            color=color2,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.8,
        )
        ax2.set_yticks(range(len(top_actor2)))
        ax2.set_yticklabels([translate_term(t) for t in top_actor2['term']], fontsize=10)
        ax2.invert_yaxis()
    ax2.set_xlabel('Z-score (log-odds ratio)', fontsize=11)
    ax2.set_title(f'Distinctive of {name2}', fontsize=12, fontweight='bold')

    # Overall title
    total1 = comparison_df['actor1_total'].iloc[0] if len(comparison_df) > 0 else 0
    total2 = comparison_df['actor2_total'].iloc[0] if len(comparison_df) > 0 else 0
    fig.suptitle(
        f'Risk Term Distinctiveness: {name1} vs {name2}\n'
        f'(n₁={total1:,}, n₂={total2:,} dictionary term mentions)',
        fontsize=13,
        fontweight='bold',
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {output_path}")


# =============================================================================
# One-vs-Rest Comparison
# =============================================================================

def compare_one_vs_rest(
    actor_counts: Dict[str, Counter],
    actor_totals: Dict[str, int],
    prior_weight: float = 0.01,
) -> pd.DataFrame:
    """
    Compare each actor against all others combined (one-vs-rest).

    Returns a single DataFrame with columns for each actor's z-score,
    enabling direct comparison of what makes each actor distinctive.

    Parameters
    ----------
    actor_counts : dict
        {actor: Counter({term: count})}
    actor_totals : dict
        {actor: total_count}
    prior_weight : float
        Controls shrinkage strength (default: 0.01)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: term, z_<actor1>, z_<actor2>, ..., distinctive_of
    """
    actors = [a for a in ACTOR_ORDER if a in actor_counts]
    all_terms = set()
    for counts in actor_counts.values():
        all_terms.update(counts.keys())

    if not all_terms:
        return pd.DataFrame()

    results = []

    for term in all_terms:
        row = {'term': term}

        for actor in actors:
            # This actor vs all others
            count_actor = actor_counts[actor].get(term, 0)
            total_actor = actor_totals[actor]

            # Combine all other actors
            count_rest = sum(
                actor_counts[a].get(term, 0)
                for a in actors if a != actor
            )
            total_rest = sum(
                actor_totals[a]
                for a in actors if a != actor
            )

            # Monroe et al. with proper scaling
            # α_i = pooled COUNT × prior_weight
            alpha_i = max((count_actor + count_rest) * prior_weight, 1e-10)
            alpha_0 = (total_actor + total_rest) * prior_weight

            numerator1 = count_actor + alpha_i
            denominator1 = max(total_actor + alpha_0 - count_actor - alpha_i, 1e-10)
            numerator2 = count_rest + alpha_i
            denominator2 = max(total_rest + alpha_0 - count_rest - alpha_i, 1e-10)

            log_odds = np.log(numerator1 / denominator1) - np.log(numerator2 / denominator2)
            variance = 1 / (count_actor + alpha_i) + 1 / (count_rest + alpha_i)
            z_score = log_odds / np.sqrt(variance)

            row[f'z_{actor}'] = z_score
            row[f'count_{actor}'] = count_actor

        results.append(row)

    df = pd.DataFrame(results)

    # Determine which actor each term is most distinctive of
    z_cols = [f'z_{a}' for a in actors]
    df['max_z'] = df[z_cols].max(axis=1)
    df['distinctive_of'] = df[z_cols].idxmax(axis=1).str.replace('z_', '')

    # Sort by maximum distinctiveness
    df = df.sort_values('max_z', ascending=False).reset_index(drop=True)

    return df


def plot_all_actors_distinctiveness(
    ovr_df: pd.DataFrame,
    actors: List[str],
    output_path: Path,
    top_n: int = 10,
) -> None:
    """
    Create multi-panel bar chart showing top distinctive terms for each actor.

    Parameters
    ----------
    ovr_df : pd.DataFrame
        Output from compare_one_vs_rest()
    actors : list
        List of actor names
    output_path : Path
        Where to save the plot
    top_n : int
        Number of top terms to show per actor
    """
    n_actors = len(actors)
    fig, axes = plt.subplots(1, n_actors, figsize=(6 * n_actors, 8))

    if n_actors == 1:
        axes = [axes]

    # Use standard actor colors

    for ax, actor in zip(axes, actors):
        z_col = f'z_{actor}'
        count_col = f'count_{actor}'

        # Get top terms for this actor (positive z-scores = distinctive of this actor)
        actor_df = ovr_df[ovr_df[z_col] > 0].nlargest(top_n, z_col)

        color = ACTOR_COLORS.get(actor, '#666666')

        ax.barh(
            range(len(actor_df)),
            actor_df[z_col],
            color=color,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.8,
        )
        ax.set_yticks(range(len(actor_df)))
        ax.set_yticklabels([translate_term(t) for t in actor_df['term']], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Z-score (vs all others)', fontsize=11)
        ax.set_title(
            f'{ACTOR_DISPLAY_NAMES.get(actor, actor)}',
            fontsize=12,
            fontweight='bold',
        )

    fig.suptitle(
        'Risk Term Distinctiveness: One-vs-Rest Comparison\n'
        '(Positive z-score = term is over-represented for this actor)',
        fontsize=13,
        fontweight='bold',
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {output_path}")


def print_one_vs_rest_summary(
    ovr_df: pd.DataFrame,
    actors: List[str],
    top_n: int = 10,
) -> None:
    """Print summary of one-vs-rest comparison."""
    print("\n" + "=" * 70)
    print("ONE-VS-REST DISTINCTIVENESS")
    print("(Each actor compared to all others combined)")
    print("=" * 70)

    for actor in actors:
        z_col = f'z_{actor}'
        count_col = f'count_{actor}'
        name = ACTOR_DISPLAY_NAMES.get(actor, actor)

        # Top distinctive terms for this actor
        top_terms = ovr_df[ovr_df[z_col] > 0].nlargest(top_n, z_col)

        print(f"\n{name} (vs all others):")
        print(f"  {'Risk':<30} {'z-score':>8} {'Count':>8}")
        print("  " + "-" * 48)

        for _, row in top_terms.iterrows():
            print(f"  {row['term']:<30} {row[z_col]:>8.2f} {row[count_col]:>8,}")


# =============================================================================
# Output
# =============================================================================

def print_comparison_summary(
    comparison_df: pd.DataFrame,
    actor1: str,
    actor2: str,
    top_n: int = 10,
) -> None:
    """Print summary of pairwise comparison."""
    if comparison_df.empty:
        print(f"\nNo data for comparison: {actor1} vs {actor2}")
        return

    name1 = ACTOR_DISPLAY_NAMES.get(actor1, actor1)
    name2 = ACTOR_DISPLAY_NAMES.get(actor2, actor2)

    n_total = len(comparison_df)

    print("\n" + "=" * 70)
    print(f"DISTINCTIVENESS: {name1} vs {name2}")
    print("=" * 70)
    print(f"Total terms compared: {n_total}")

    # Top distinctive of actor1
    top1 = comparison_df.nlargest(top_n, 'z_score')
    print(f"\nTop risks distinctive of {name1}:")
    print(f"  {'Risk':<30} {'z-score':>8} {'Count₁':>8} {'Count₂':>8}")
    print("  " + "-" * 56)
    for _, row in top1.iterrows():
        print(f"  {row['term']:<30} {row['z_score']:>8.2f} "
              f"{row['count1']:>8,} {row['count2']:>8,}")

    # Top distinctive of actor2
    top2 = comparison_df.nsmallest(top_n, 'z_score')
    print(f"\nTop risks distinctive of {name2}:")
    print(f"  {'Risk':<30} {'z-score':>8} {'Count₁':>8} {'Count₂':>8}")
    print("  " + "-" * 56)
    for _, row in top2.iterrows():
        print(f"  {row['term']:<30} {row['z_score']:>8.2f} "
              f"{row['count1']:>8,} {row['count2']:>8,}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze actor distinctiveness using Monroe et al. (2008) method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Path to term-document matrix CSV',
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./results/01_bow_analysis/distinctiveness'),
        help='Output directory',
    )

    parser.add_argument(
        '--top', '-n',
        type=int,
        default=15,
        help='Number of top distinctive terms to display/plot (default: 15)',
    )

    parser.add_argument(
        '--prior-weight', '-p',
        type=float,
        default=0.01,
        help='Shrinkage weight for Dirichlet prior (default: 0.01). Higher = more shrinkage.',
    )

    parser.add_argument(
        '--by-wave',
        action='store_true',
        help='Also compute distinctiveness by wave (temporal breakdown)',
    )

    parser.add_argument(
        '--one-vs-rest',
        action='store_true',
        help='Compare each actor against all others combined (multi-actor view)',
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input matrix not found: {args.input}")
        return 1

    print("=" * 70)
    print("RISK DISTINCTIVENESS ANALYSIS")
    print("Monroe et al. (2008) 'Fightin' Words' Method")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Prior weight: {args.prior_weight}")

    # Load term-document matrix
    logger.info("\nLoading term-document matrix...")
    df, risk_cols = load_matrix(args.input)
    n_docs = len(df)
    logger.info(f"  Loaded {n_docs} documents with {len(risk_cols)} risk terms")

    # Aggregate counts by actor
    actor_counts, actor_totals = aggregate_counts_from_matrix(df, risk_cols)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Run all pairwise comparisons
    actors = [a for a in ACTOR_ORDER if a in actor_counts]

    if len(actors) < 2:
        logger.error(f"Need at least 2 actors, found: {actors}")
        return 1

    logger.info(f"\nRunning pairwise comparisons for: {actors}")

    all_comparisons = []

    for actor1, actor2 in combinations(actors, 2):
        comparison_df = compare_actors(
            actor_counts, actor_totals,
            actor1, actor2,
            args.prior_weight,
        )

        if comparison_df.empty:
            continue

        all_comparisons.append((actor1, actor2, comparison_df))

        # Print summary
        print_comparison_summary(comparison_df, actor1, actor2, args.top)

        # Save CSV
        csv_path = args.output / f'distinctiveness_{actor1}_vs_{actor2}.csv'
        comparison_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"  Saved: {csv_path}")

        # Save visualization
        viz_path = args.output / f'viz_distinctiveness_{actor1}_vs_{actor2}.png'
        plot_distinctive_terms(comparison_df, actor1, actor2, viz_path, args.top)

    # One-vs-rest comparison
    if args.one_vs_rest:
        logger.info("\n" + "-" * 70)
        logger.info("ONE-VS-REST COMPARISON")
        logger.info("-" * 70)

        ovr_df = compare_one_vs_rest(actor_counts, actor_totals, args.prior_weight)

        # Print summary
        print_one_vs_rest_summary(ovr_df, actors, args.top)

        # Save CSV
        csv_path = args.output / 'distinctiveness_one_vs_rest.csv'
        ovr_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"  Saved: {csv_path}")

        # Save visualization
        viz_path = args.output / 'viz_distinctiveness_one_vs_rest.png'
        plot_all_actors_distinctiveness(ovr_df, actors, viz_path, args.top)

    # By-wave analysis
    if args.by_wave:
        logger.info("\n" + "=" * 70)
        logger.info("TEMPORAL BREAKDOWN (by wave)")
        logger.info("=" * 70)

        wave_data = aggregate_counts_by_wave_from_matrix(df, risk_cols)

        wave_output = args.output / 'by_wave'
        wave_output.mkdir(parents=True, exist_ok=True)

        for wave, (wave_actor_counts, wave_actor_totals) in wave_data.items():
            wave_label = WAVE_LABELS.get(wave, f'wave_{wave}')
            logger.info(f"\n--- Wave {wave}: {wave_label} ---")

            for actor1, actor2 in combinations(sorted(wave_actor_counts.keys()), 2):
                comparison_df = compare_actors(
                    wave_actor_counts, wave_actor_totals,
                    actor1, actor2,
                    args.prior_weight,
                )

                if comparison_df.empty:
                    continue

                # Save CSV
                csv_path = wave_output / f'distinctiveness_{actor1}_vs_{actor2}_wave{wave}.csv'
                comparison_df.to_csv(csv_path, index=False, encoding='utf-8')

                # Save visualization
                viz_path = wave_output / f'viz_distinctiveness_{actor1}_vs_{actor2}_wave{wave}.png'
                plot_distinctive_terms(comparison_df, actor1, actor2, viz_path, args.top)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nDocuments: {n_docs}")
    print(f"Risk terms: {len(risk_cols)}")
    print(f"\nActor totals (dictionary term mentions):")
    for actor in sorted(actor_totals.keys()):
        docs = len(df[df['actor'] == actor])
        total = actor_totals[actor]
        per_doc = total / docs if docs > 0 else 0
        print(f"  {ACTOR_DISPLAY_NAMES.get(actor, actor):<20} {total:>8,} mentions ({docs} docs, {per_doc:.1f}/doc)")

    print(f"\nComparisons saved to: {args.output}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
