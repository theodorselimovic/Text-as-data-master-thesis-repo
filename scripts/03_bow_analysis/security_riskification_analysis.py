#!/usr/bin/env python3
"""
Security Riskification Analysis

Compares analytical language (probability, consequence, risk qualifications)
between security risks and other risks. If security paragraphs use the same
qualification vocabulary as other risks, it demonstrates that security threats
have been "riskified" - absorbed into the standard risk analysis framework.

Key question (Q6): How do security threats become riskified?

Hypothesis: Security risks are analyzed with the same probability/consequence
framework as other risks, suggesting they're processed through the same
legitimacy-driven institutional mechanisms.

Output:
    results/01_bow_analysis/security_riskification/
        qualification_by_risk_type.png      # Side-by-side comparison
        qualification_distributions.png     # Detailed distributions
        statistical_comparison.txt          # Chi-square tests
        summary_stats.csv                   # Numeric summaries

Usage:
    python security_riskification_analysis.py \
        --corpus data/processed/bow_corpus_stemmed.parquet \
        --output results/01_bow_analysis/security_riskification/
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer
from scipy import stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries import get_legacy_risk_dictionary, RISK_TO_CATEGORY

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Security category (antagonistic in MSB taxonomy)
SECURITY_CATEGORIES = ['antagonistiska_hot', 'cyber_hot']

# Comparison categories
OTHER_CATEGORIES = ['naturhot', 'teknisk_infrastruktur', 'biologiska_hot',
                    'olyckor', 'miljö_klimat', 'sociala_risker', 'ekonomi']

# Colors for visualization
RISK_TYPE_COLORS = {
    'security': '#e41a1c',  # Red (same as municipality)
    'other': '#377eb8',     # Blue (same as prefecture)
}

LEVEL_ORDER = ['very_low', 'low', 'medium', 'high', 'very_high']
LEVEL_LABELS = {
    'very_low': 'Very Low',
    'low': 'Low',
    'medium': 'Medium',
    'high': 'High',
    'very_high': 'Very High',
}

# =============================================================================
# QUALIFICATION DICTIONARIES (from risk_context_analysis.py)
# =============================================================================

TARGET_WORDS_RAW = {
    'sannolikhet': ['sannolikhet', 'sannolikheten', 'sannolikhets',
                    'trolig', 'troligt', 'troliga',
                    'sannolik', 'sannolikt', 'sannolika',
                    'osannolik', 'osannolikt', 'osannolika'],
    'konsekvens': ['konsekvens', 'konsekvensen', 'konsekvenser'],
    'risk': ['risk', 'risken', 'risker', 'riskens']
}

QUALIFICATION_MAPPING_RAW = {
    'sannolikhet': {
        'very_low': ['mycket låg', 'mycket liten', 'sällsynt'],
        'low': ['låg', 'liten', 'små'],
        'medium': ['medelhög', 'mellan', 'möjlig'],
        'high': ['hög', 'stor'],
        'very_high': ['mycket hög', 'mycket stora', 'stora'],
    },
    'konsekvens': {
        'very_low': ['mycket begränsade', 'mycket liten', 'försumbara',
                     'obetydlig', 'obetydliga', 'marginell', 'marginella'],
        'low': ['begränsade', 'lindriga', 'liten', 'små', 'ringa'],
        'medium': ['kännbara', 'måttliga', 'direkta', 'märkbar', 'märkbara'],
        'high': ['allvarlig', 'allvarliga', 'betydande', 'stor', 'stora',
                 'omfattande', 'svåra', 'kraftig', 'kraftiga'],
        'very_high': ['mycket allvarlig', 'mycket allvarliga', 'mycket stora',
                      'katastrofal', 'katastrofala', 'extrem', 'extrema', 'förödande'],
    },
    'risk': {
        'very_low': ['mycket låg', 'mycket liten', 'försumbar', 'obetydlig'],
        'low': ['låg', 'liten', 'små', 'begränsad', 'måttlig'],
        'medium': ['medelhög', 'mellan', 'påtaglig', 'märkbar'],
        'high': ['hög', 'stor', 'stora', 'omfattande', 'avsevärd', 'betydande'],
        'very_high': ['mycket hög', 'mycket stora', 'extrem', 'kritisk', 'akut'],
    }
}

# =============================================================================
# STEMMING AND TAGGING
# =============================================================================

def initialize_stemmed_dictionaries() -> Tuple[Dict, Dict, Dict]:
    """Initialize stemmed versions of all dictionaries."""
    stemmer = SnowballStemmer('swedish')

    # Stem target words
    stemmed_targets = {}
    for concept, words in TARGET_WORDS_RAW.items():
        stemmed_targets[concept] = set(stemmer.stem(w) for w in words)

    # Stem qualification mapping - use underscores for n-grams (matches corpus format)
    stemmed_qualifications = {}
    stem_to_level = {}

    for concept, levels in QUALIFICATION_MAPPING_RAW.items():
        stemmed_qualifications[concept] = {}
        stem_to_level[concept] = {}

        for level, phrases in levels.items():
            stemmed_qualifications[concept][level] = set()
            for phrase in phrases:
                # Use underscore for multi-word (matches corpus n-gram format)
                stemmed = '_'.join(stemmer.stem(w) for w in phrase.split())
                stemmed_qualifications[concept][level].add(stemmed)
                stem_to_level[concept][stemmed] = level

    # Stem risk dictionary for category tagging
    # Only use single-word terms (most reliable for matching)
    risk_dict = get_legacy_risk_dictionary(include_extended=False)
    stemmed_risk_dict = {}
    for category, terms in risk_dict.items():
        stemmed_risk_dict[category] = set()
        for term in terms:
            # Only stem single words; skip multi-word phrases
            if ' ' not in term:
                stemmed_risk_dict[category].add(stemmer.stem(term))

    return stemmed_targets, stemmed_qualifications, stemmed_risk_dict


def tag_sentence_risk_category(
    tokens: List[str],
    stemmed_risk_dict: Dict[str, Set[str]]
) -> str:
    """
    Tag a sentence with its dominant risk category.

    Returns 'security', 'other', or 'none'.
    """
    token_set = set(tokens)

    category_counts = defaultdict(int)
    for category, stems in stemmed_risk_dict.items():
        matches = token_set & stems
        category_counts[category] = len(matches)

    if sum(category_counts.values()) == 0:
        return 'none'

    # Find dominant category
    dominant = max(category_counts, key=category_counts.get)

    # Map to security/other
    if dominant in SECURITY_CATEGORIES:
        return 'security'
    else:
        return 'other'


def extract_qualification(
    tokens: List[str],
    concept: str,
    stemmed_targets: Dict,
    stemmed_qualifications: Dict,
    stem_to_level: Dict,
) -> str:
    """
    Extract qualification level for a concept from a sentence.

    Returns level (very_low, low, medium, high, very_high) or None.
    """
    tokens_set = set(tokens)

    # Check if target word is present
    if not (stemmed_targets[concept] & tokens_set):
        return None

    # Check for qualification patterns (including n-grams)
    # Sort by length (longest first) to match "mycket_hög" before "hög"
    all_patterns = []
    for level in LEVEL_ORDER:
        if level in stemmed_qualifications[concept]:
            for pattern in stemmed_qualifications[concept][level]:
                all_patterns.append((pattern, level))

    all_patterns.sort(key=lambda x: len(x[0]), reverse=True)

    # Check n-gram matches first
    for pattern, level in all_patterns:
        if '_' in pattern and pattern in tokens_set:
            return level

    # Check single-word matches
    for pattern, level in all_patterns:
        if '_' not in pattern and pattern in tokens_set:
            return level

    return None


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_qualifications_by_risk_type(
    df: pd.DataFrame,
    stemmed_targets: Dict,
    stemmed_qualifications: Dict,
    stem_to_level: Dict,
    stemmed_risk_dict: Dict
) -> pd.DataFrame:
    """
    Analyze qualification distributions by risk type (security vs other).

    Returns DataFrame with counts per sentence.
    """
    results = []

    # Progress tracking
    total = len(df)
    checkpoint = total // 10

    for idx, row in df.iterrows():
        if idx % checkpoint == 0 and idx > 0:
            logger.info(f"  Progress: {idx:,}/{total:,} ({100*idx//total}%)")

        # Handle tokens - could be numpy array, list, or empty
        tokens_raw = row.get('tokens')
        if tokens_raw is None:
            continue

        # Convert numpy array to list if needed
        if hasattr(tokens_raw, 'tolist'):
            tokens = tokens_raw.tolist()
        elif isinstance(tokens_raw, list):
            tokens = tokens_raw
        else:
            continue

        if not tokens:
            continue

        # Tag risk type
        risk_type = tag_sentence_risk_category(tokens, stemmed_risk_dict)
        if risk_type == 'none':
            continue

        # Extract qualifications
        for concept in ['sannolikhet', 'konsekvens', 'risk']:
            level = extract_qualification(
                tokens, concept, stemmed_targets,
                stemmed_qualifications, stem_to_level
            )
            if level:
                results.append({
                    'doc_id': row.get('doc_id', ''),
                    'sentence_id': row.get('sentence_id', idx),
                    'risk_type': risk_type,
                    'concept': concept,
                    'level': level,
                    'actor_type': row.get('actor_type', ''),
                    'year': row.get('year', ''),
                })

    return pd.DataFrame(results)


def compute_statistics(results_df: pd.DataFrame) -> Dict:
    """Compute descriptive statistics comparing security vs other distributions."""
    stats_results = {}

    for concept in ['sannolikhet', 'konsekvens', 'risk']:
        concept_df = results_df[results_df['concept'] == concept]

        # Build contingency table
        contingency = pd.crosstab(
            concept_df['risk_type'],
            concept_df['level']
        )

        # Ensure both risk types present
        if len(contingency) < 2:
            stats_results[concept] = {'error': 'Insufficient data'}
            continue

        # Compute chi2 for Cramér's V (effect size only, not for inference)
        chi2, _, dof, _ = stats.chi2_contingency(contingency)

        # Effect size (Cramér's V) - descriptive measure of association
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        stats_results[concept] = {
            'cramers_v': cramers_v,
            'n_security': contingency.loc['security'].sum() if 'security' in contingency.index else 0,
            'n_other': contingency.loc['other'].sum() if 'other' in contingency.index else 0,
            'contingency': contingency.to_dict(),
        }

    return stats_results


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_qualification_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Create side-by-side comparison of qualifications by risk type."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    concepts = ['sannolikhet', 'konsekvens', 'risk']
    concept_labels = {'sannolikhet': 'Probability', 'konsekvens': 'Consequence', 'risk': 'Risk'}

    for ax, concept in zip(axes, concepts):
        concept_df = results_df[results_df['concept'] == concept]

        # Calculate proportions within each risk type
        props = concept_df.groupby(['risk_type', 'level']).size().unstack(fill_value=0)
        props = props.reindex(columns=[l for l in LEVEL_ORDER if l in props.columns])
        props = props.div(props.sum(axis=1), axis=0) * 100

        # Reorder index
        if 'security' in props.index and 'other' in props.index:
            props = props.reindex(['security', 'other'])

        # Plot
        x = np.arange(len(props.columns))
        width = 0.35

        if 'security' in props.index:
            ax.bar(x - width/2, props.loc['security'], width,
                   label='Security', color=RISK_TYPE_COLORS['security'], alpha=0.8)
        if 'other' in props.index:
            ax.bar(x + width/2, props.loc['other'], width,
                   label='Other', color=RISK_TYPE_COLORS['other'], alpha=0.8)

        ax.set_xlabel('Qualification Level')
        ax.set_ylabel('Percentage')
        ax.set_title(f'{concept_labels[concept]} Qualifications')
        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_LABELS.get(l, l) for l in props.columns], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 60)

    plt.tight_layout()
    plt.savefig(output_dir / 'qualification_by_risk_type.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'qualification_by_risk_type.pdf', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved qualification_by_risk_type.png")


def create_distribution_violin(results_df: pd.DataFrame, output_dir: Path):
    """Create violin plots showing full distributions."""

    # Convert level to numeric for violin plot
    level_to_num = {l: i for i, l in enumerate(LEVEL_ORDER)}
    results_df = results_df.copy()
    results_df['level_num'] = results_df['level'].map(level_to_num)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    concepts = ['sannolikhet', 'konsekvens', 'risk']
    concept_labels = {'sannolikhet': 'Probability', 'konsekvens': 'Consequence', 'risk': 'Risk'}

    for ax, concept in zip(axes, concepts):
        concept_df = results_df[results_df['concept'] == concept]

        if len(concept_df) == 0:
            continue

        sns.violinplot(
            data=concept_df,
            x='risk_type',
            y='level_num',
            palette=RISK_TYPE_COLORS,
            order=['security', 'other'],
            ax=ax,
            inner='box',
            cut=0
        )

        ax.set_xlabel('Risk Type')
        ax.set_ylabel('Qualification Level')
        ax.set_title(f'{concept_labels[concept]}')
        ax.set_yticks(range(len(LEVEL_ORDER)))
        ax.set_yticklabels([LEVEL_LABELS[l] for l in LEVEL_ORDER])
        ax.set_xticklabels(['Security', 'Other'])

    plt.tight_layout()
    plt.savefig(output_dir / 'qualification_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'qualification_distributions.pdf', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved qualification_distributions.png")


def save_statistical_results(stats_results: Dict, output_dir: Path):
    """Save descriptive comparison results."""

    with open(output_dir / 'descriptive_comparison.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SECURITY RISKIFICATION ANALYSIS: DESCRIPTIVE COMPARISON\n")
        f.write("=" * 70 + "\n\n")

        f.write("Question: Do security risks use the same qualification vocabulary\n")
        f.write("as other risks?\n\n")

        f.write("Cramér's V measures association strength between risk type and\n")
        f.write("qualification level (0 = identical distributions, 1 = completely different).\n\n")

        for concept, result in stats_results.items():
            f.write("-" * 50 + "\n")
            f.write(f"{concept.upper()}\n")
            f.write("-" * 50 + "\n")

            if 'error' in result:
                f.write(f"Error: {result['error']}\n\n")
                continue

            f.write(f"N (security): {result['n_security']}\n")
            f.write(f"N (other):    {result['n_other']}\n\n")

            f.write(f"Cramér's V:   {result['cramers_v']:.3f}\n\n")

            if result['cramers_v'] < 0.1:
                f.write("INTERPRETATION: Negligible association (V < 0.1)\n")
                f.write("=> Security and other risks use very similar qualification patterns\n")
            elif result['cramers_v'] < 0.3:
                f.write("INTERPRETATION: Small association (0.1 < V < 0.3)\n")
                f.write("=> Minor differences in qualification patterns\n")
            else:
                f.write("INTERPRETATION: Moderate/large association (V > 0.3)\n")
                f.write("=> Notable differences in qualification patterns\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 70 + "\n\n")
        f.write("Small Cramér's V values support the 'riskification' hypothesis:\n")
        f.write("security threats are processed through the same analytical\n")
        f.write("framework as other risks.\n")

    logger.info(f"Saved descriptive_comparison.txt")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Security riskification analysis")
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/processed/bow_corpus_stemmed.parquet'),
        help='Input stemmed corpus'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/01_bow_analysis/security_riskification'),
        help='Output directory'
    )
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading corpus from {args.corpus}")
    df = pd.read_parquet(args.corpus)
    logger.info(f"Loaded {len(df):,} sentences")

    # Initialize dictionaries
    logger.info("Initializing stemmed dictionaries...")
    stemmed_targets, stemmed_qualifications, stemmed_risk_dict = initialize_stemmed_dictionaries()

    # Build stem_to_level mapping
    stem_to_level = {}
    for concept, levels in stemmed_qualifications.items():
        stem_to_level[concept] = {}
        for level, stems in levels.items():
            for stem in stems:
                # Only single-word stems
                if ' ' not in stem:
                    stem_to_level[concept][stem] = level

    # Analyze
    logger.info("Analyzing qualifications by risk type...")
    results_df = analyze_qualifications_by_risk_type(
        df, stemmed_targets, stemmed_qualifications,
        stem_to_level, stemmed_risk_dict
    )

    logger.info(f"Found {len(results_df):,} qualification instances")
    logger.info(f"  Security: {(results_df['risk_type'] == 'security').sum():,}")
    logger.info(f"  Other:    {(results_df['risk_type'] == 'other').sum():,}")

    # Save raw results
    results_df.to_csv(args.output / 'qualification_results.csv', index=False)

    # Compute statistics
    logger.info("Computing statistical comparisons...")
    stats_results = compute_statistics(results_df)

    # Generate visualizations
    logger.info("Creating visualizations...")
    create_qualification_comparison(results_df, args.output)
    create_distribution_violin(results_df, args.output)

    # Save statistics
    save_statistical_results(stats_results, args.output)

    # Summary stats
    summary = results_df.groupby(['risk_type', 'concept', 'level']).size().unstack(fill_value=0)
    summary.to_csv(args.output / 'summary_stats.csv')

    logger.info(f"Done! Results saved to {args.output}")


if __name__ == '__main__':
    main()
