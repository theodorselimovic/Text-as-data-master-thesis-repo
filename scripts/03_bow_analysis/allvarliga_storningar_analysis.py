#!/usr/bin/env python3
"""
Allvarliga Störningar Risk Analysis

Analyzes which risk categories co-occur with "allvarliga störningar" (serious
disruptions) phrases in RSA documents. This reveals how different actors frame
risks in terms of their potential to cause operational disruptions.

The analysis filters paragraphs containing disruption phrases, then counts
which risk categories appear in those paragraphs using the standard risk
dictionary.

Input:
    data/processed/bert_corpus.parquet (sentence-level, ~380k sentences)

Output:
    results/bow_analysis/allvarliga_storningar_risks.csv
    results/bow_analysis/allvarliga_storningar_by_actor.csv
    results/bow_analysis/allvarliga_storningar_by_wave.csv
    results/bow_analysis/allvarliga_storningar_sample.csv (sample paragraphs)

Usage:
    python allvarliga_storningar_analysis.py
    python allvarliga_storningar_analysis.py --verbose
    python allvarliga_storningar_analysis.py --sample-size 50

Requirements:
    pip install pandas pyarrow
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# DISRUPTION PATTERNS
# =============================================================================

# Phrases indicating "serious disruptions" to actor's operations
DISRUPTION_PHRASES = [
    # Core phrases
    'allvarliga störningar',
    'allvarlig störning',
    # Variants with object
    'störningar i verksamheten',
    'störningar av verksamheten',
    'störning i verksamheten',
    'störning av verksamheten',
    # Participle form
    'störd verksamhet',
    # Samhällsstörning variants (broader societal disruption)
    'samhällsstörning',
    'samhällsstörningar',
    'allvarlig samhällsstörning',
    'allvarliga samhällsstörningar',
]


def build_disruption_pattern() -> re.Pattern:
    """Build regex pattern for disruption phrases."""
    # Sort by length (longest first) to avoid partial matches
    sorted_phrases = sorted(DISRUPTION_PHRASES, key=len, reverse=True)
    term_patterns = [r'\b' + re.escape(phrase) + r'\b' for phrase in sorted_phrases]
    combined = '|'.join(term_patterns)
    return re.compile(combined, re.IGNORECASE)


# =============================================================================
# RISK DICTIONARY (from risk_context_analysis.py)
# =============================================================================

RISK_DICTIONARY = {
    'naturhot': [
        'naturhändelser', 'naturhot', 'väderrelaterade händelser',
        'klimatförändring', 'klimatförändringarna', 'klimatförändringar',
        'översvämning', 'översvämningar', 'skyfall', 'höga flöden', 'högvatten',
        'värme', 'värmebölja', 'värmeböljor', 'torka', 'torkor',
        'ras', 'skred', 'jordskred', 'slamskred', 'erosion',
        'storm', 'stormar', 'stormfällning',
        'skogsbrand', 'skogsbränder', 'gräsbrand',
        'blixt', 'blixtnedslag', 'hagel', 'halka', 'köldknäpp',
        'stora snömängder', 'snöoväder',
        'extrem värme', 'extremvärme', 'extrem kyla', 'extremt väder',
        'låga flöden', 'lågvatten',
        'jordbävning', 'jordskalv', 'seismisk',
        'lavin', 'laviner', 'snölavin',
        'tsunami', 'tsunamier',
        'vattenbrist', 'grundvattennivå', 'grundvattenbrist',
    ],
    'biologiska_hot': [
        'epidemi', 'epidemier', 'pandemi', 'pandemier',
        'epozooti', 'epizootier', 'covid', 'coronaviruset',
        'smittsam sjukdom', 'smittsamma sjukdomar',
        'smitta', 'smittspridning', 'sjukdomsutbrott',
        'influensa', 'influensapandemi',
        'djursjukdom', 'djursjukdomar', 'zoonos', 'zoonoser',
        'antibiotikaresistens', 'resistenta bakterier',
        'hälsa', 'folkhälsa',
    ],
    'olyckor': [
        'olycka vid farlig verksamhet', 'farlig verksamhet',
        'industriolycka', 'kemikalieolycka',
        'olycka med transport av farligt gods', 'olycka med farligt gods',
        'farligt gods', 'transport av farligt gods',
        'vägolycka', 'vägolyckor', 'trafikolycka', 'trafikolyckor',
        'tågolycka', 'tågolyckor', 'järnvägsolycka', 'järnvägsolyckor',
        'bussolycka', 'bussolyckor', 'spårbundna olyckor',
        'dammbrott',
        'fartygsolycka', 'fartygsolyckor', 'båtolycka', 'båtolyckor',
        'flygolycka', 'flygolyckor', 'flyghaveri',
        'olyckor med nukleära ämnen', 'olyckor med radioaktiva ämnen',
        'kärnkraftsolycka', 'kärnteknisk olycka', 'strålning', 'radioaktivitet', 'kärnavfall',
        'brokollaps', 'tunnelolycka',
        'byggnadsras', 'byggnadskollaps',
        'försvunnen person', 'försvunna personer', 'försvunnen brukare',
        'försvinnande', 'saknad person',
    ],
    'antagonistiska_hot': [
        'statliga antagonister', 'statlig antagonist',
        'icke-statliga antagonister', 'icke-statlig antagonist',
        'terror', 'terrorism', 'terrorhot', 'terrorattentat',
        'hot och våld', 'våld', 'våldsbrott',
        'pågående dödligt våld', 'våldsbejakande extremism',
        'sabotage', 'spionage',
        'brott', 'kriminalitet', 'organiserad brottslighet',
        'vandalism', 'skadegörelse', 'inbrott',
        'desinformation', 'påverkanskampanj', 'påverkanskampanjer',
        'hybrid hot', 'hybridhot',
        'säkerhetshot', 'säkerhet', 'hot om attentat',
        'kidnapping', 'väpnad konflikt', 'påverkansoperationer',
        'upplopp', 'beväpnad konflikt',
        'krig', 'väpnat angrepp', 'invasion', 'mobilisering',
        'totalförsvar', 'totalförsvaret', 'krigstillstånd',
        'militärt hot', 'militärt angrepp',
    ],
    'cyber_hot': [
        'dataintrång', 'cyberattack', 'cyberattacker', 'cybersäkerhet',
        'nätattack', 'nätattacker', 'hackerattack', 'hackerattacker',
        'DDoS-attack', 'ddos-attack', 'ransomware', 'datavirus', 'virus',
        'IT-sabotage',
    ],
    'sociala_risker': [
        'samhällsvärden', 'värdesystem',
        'social oro', 'sociala oroligheter', 'civila oroligheter', 'upplopp',
        'folksamling', 'folksamlingar', 'stora evenemang', 'publikevenemang',
        'massevenemang', 'stor tillställning',
        'flyktingkris', 'migration', 'flyktingström', 'flyktingströmmar',
        'massflykt',
    ],
    'teknisk_infrastruktur': [
        'strömavbrott', 'elavbrott', 'kraftförsörjning', 'elförsörjning', 'effektbrist',
        'fjärrvärmebrott', 'fjärrvärme', 'värmeförsörjning',
        'vattenläcka', 'vattenläckor', 'vattenförsörjning', 'dricksvatten',
        'avloppsbrott', 'avloppssystem',
        'IT-bortfall', 'it-bortfall', 'IT-avbrott', 'it-avbrott',
        'dataförlust', 'systemfel', 'nätverksavbrott',
        'kommunikationsavbrott', 'teleavbrott', 'telebrott', 'elektroniska kommunikationer',
        'distributionsstörning', 'logistikavbrott', 'transportavbrott',
        'drivsmedelsbrist', 'bränslebrist', 'försörjningsbrist',
        'livsmedelsförsörjning', 'livsmedelsbrist', 'matförsörjning',
    ],
    'brand': [
        'brand', 'bränder', 'skogsbrand', 'skogsbränder', 'storbrand',
        'gräsbrand', 'gräsbränder', 'byggnadsbrand', 'fordonsbrand',
        'explosion', 'explosioner', 'gasexplosion', 'brandfarligt gods',
    ],
    'miljö_klimat': [
        'miljöförorening', 'kemikalieutsläpp', 'oljeutsläpp',
        'markförorening', 'luftföroreningar', 'vattenförorening',
        'miljöhot', 'miljöskada', 'utsläpp', 'föroreningar', 'klimatförändring',
        'klimatpåverkan', 'klimatrelaterade', 'klimatförändringen', 'försurning',
    ],
    'ekonomi': [
        'ekonomisk kris', 'finanskris', 'recession',
        'arbetslöshet', 'inflation', 'ekonomisk nedgång',
    ],
}


def count_risk_terms_by_category(text: str) -> Dict[str, int]:
    """Count occurrences of risk terms by category."""
    text_lower = text.lower()
    results = {}

    for category, terms in RISK_DICTIONARY.items():
        category_count = 0
        for term in terms:
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            count = len(re.findall(pattern, text_lower))
            category_count += count
        results[category] = category_count

    return results


# =============================================================================
# PARAGRAPH PROCESSING
# =============================================================================

def derive_wave(year_str: str) -> int:
    """
    Derive wave from year string.

    Wave definitions:
    - 0: pre-2015
    - 1: 2015-2018
    - 2: 2019-2022
    - 3: 2023+
    """
    try:
        year = int(year_str)
    except (ValueError, TypeError):
        return -1  # Unknown

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
    Group sentences by paragraph, preserving metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level dataframe with doc_id, paragraph_id, sentence_text.

    Returns
    -------
    pd.DataFrame
        Paragraph-level dataframe with full_text and metadata.
    """
    logger.info("Grouping sentences into paragraphs...")

    # Get unique metadata per paragraph (take first row's metadata)
    meta_cols = ['doc_id', 'paragraph_id', 'actor_type', 'wave', 'year', 'filename']
    available_meta = [col for col in meta_cols if col in df.columns]

    # Concatenate sentences within each paragraph
    paragraph_texts = df.groupby(['doc_id', 'paragraph_id']).agg({
        'sentence_text': lambda x: ' '.join(x.astype(str)),
        **{col: 'first' for col in available_meta if col not in ['doc_id', 'paragraph_id']}
    }).reset_index()

    paragraph_texts.rename(columns={'sentence_text': 'full_text'}, inplace=True)

    # Derive wave from year if not present
    if 'wave' not in paragraph_texts.columns and 'year' in paragraph_texts.columns:
        paragraph_texts['wave'] = paragraph_texts['year'].apply(derive_wave)
        logger.info("  Derived wave from year")

    logger.info(f"  Created {len(paragraph_texts):,} paragraphs from {len(df):,} sentences")
    return paragraph_texts


def filter_disruption_paragraphs(
    paragraphs: pd.DataFrame,
    pattern: re.Pattern,
) -> pd.DataFrame:
    """Filter to paragraphs containing disruption phrases."""
    logger.info("Filtering paragraphs with disruption phrases...")

    mask = paragraphs['full_text'].str.contains(pattern, regex=True, na=False)
    filtered = paragraphs[mask].copy()

    logger.info(f"  Found {len(filtered):,} paragraphs with disruption phrases")
    logger.info(f"  ({len(filtered)/len(paragraphs)*100:.1f}% of total paragraphs)")

    return filtered


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_risk_categories(
    paragraphs: pd.DataFrame,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Count risk categories in paragraphs, optionally grouped.

    Parameters
    ----------
    paragraphs : pd.DataFrame
        Paragraph-level dataframe with full_text column.
    group_col : str, optional
        Column to group by (e.g., 'actor_type', 'wave').

    Returns
    -------
    pd.DataFrame
        Risk category counts, with grouping if specified.
    """
    categories = list(RISK_DICTIONARY.keys())

    if group_col is None:
        # Overall counts
        totals = {cat: 0 for cat in categories}
        for text in paragraphs['full_text']:
            counts = count_risk_terms_by_category(text)
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
        # Grouped counts
        results = []
        for group_val, group_df in paragraphs.groupby(group_col):
            totals = {cat: 0 for cat in categories}
            for text in group_df['full_text']:
                counts = count_risk_terms_by_category(text)
                for cat, count in counts.items():
                    totals[cat] += count

            for cat in categories:
                results.append({
                    group_col: group_val,
                    'category': cat,
                    'count': totals[cat],
                })

        df = pd.DataFrame(results)

        # Add percentage within each group
        df['pct'] = df.groupby(group_col)['count'].transform(
            lambda x: (x / x.sum() * 100).round(1)
        )

        return df


def extract_sample_paragraphs(
    paragraphs: pd.DataFrame,
    sample_size: int = 30,
) -> pd.DataFrame:
    """Extract a sample of paragraphs for manual inspection."""
    n = min(sample_size, len(paragraphs))
    sample = paragraphs.sample(n=n, random_state=42)

    # Select useful columns
    cols = ['doc_id', 'actor_type', 'wave', 'year', 'full_text']
    available = [c for c in cols if c in sample.columns]

    return sample[available].copy()


# =============================================================================
# VISUALIZATION
# =============================================================================

# Nice category labels for plots
CATEGORY_LABELS = {
    'naturhot': 'Natural hazards',
    'antagonistiska_hot': 'Antagonistic threats',
    'biologiska_hot': 'Biological threats',
    'teknisk_infrastruktur': 'Technical infrastructure',
    'brand': 'Fire',
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
    'kommun': 'Municipalities',
    'lansstyrelse': 'County boards',
    'MCF': 'MSB (central)',
}


def create_visualizations(
    overall: pd.DataFrame,
    by_wave: pd.DataFrame,
    by_actor: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create and save visualizations."""
    logger.info("Creating visualizations...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 10

    # Color palette for risk categories
    colors = sns.color_palette("husl", n_colors=len(CATEGORY_LABELS))

    # -------------------------------------------------------------------------
    # 1. Overall risk category distribution (horizontal bar)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Map categories to nice labels
    overall_plot = overall.copy()
    overall_plot['label'] = overall_plot['category'].map(CATEGORY_LABELS)

    bars = ax.barh(
        overall_plot['label'],
        overall_plot['count'],
        color=sns.color_palette("Blues_r", n_colors=len(overall_plot)),
    )

    ax.set_xlabel('Number of mentions')
    ax.set_title('Risk Categories in "Allvarliga Störningar" Paragraphs', fontsize=12)
    ax.invert_yaxis()  # Highest at top

    # Add count labels
    for bar, pct in zip(bars, overall_plot['pct']):
        ax.text(
            bar.get_width() + 20,
            bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%',
            va='center',
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_overall.png')
    plt.close(fig)
    logger.info(f"  Saved overall chart")

    # -------------------------------------------------------------------------
    # 2. Development over time (stacked area or line chart)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    # Pivot for line chart
    pivot = by_wave.pivot(index='wave', columns='category', values='count')

    # Select top categories for readability
    top_cats = overall.head(6)['category'].tolist()
    pivot_top = pivot[top_cats]

    # Map to nice labels
    pivot_top.columns = [CATEGORY_LABELS.get(c, c) for c in pivot_top.columns]
    pivot_top.index = [WAVE_LABELS.get(w, w) for w in pivot_top.index]

    # Plot
    pivot_top.plot(
        kind='line',
        marker='o',
        markersize=8,
        linewidth=2.5,
        ax=ax,
        color=sns.color_palette("husl", n_colors=len(top_cats)),
    )

    ax.set_xlabel('Time period')
    ax.set_ylabel('Number of mentions')
    ax.set_title('Risk Categories Over Time in "Allvarliga Störningar" Paragraphs', fontsize=12)
    ax.legend(title='Risk category', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_over_time.png')
    plt.close(fig)
    logger.info(f"  Saved over-time chart")

    # -------------------------------------------------------------------------
    # 3. Normalized development (percentage within each wave)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    # Normalize to percentages within each wave
    pivot_pct = by_wave.pivot(index='wave', columns='category', values='pct')
    pivot_pct_top = pivot_pct[top_cats]
    pivot_pct_top.columns = [CATEGORY_LABELS.get(c, c) for c in pivot_pct_top.columns]
    pivot_pct_top.index = [WAVE_LABELS.get(w, w) for w in pivot_pct_top.index]

    pivot_pct_top.plot(
        kind='bar',
        ax=ax,
        width=0.8,
        color=sns.color_palette("husl", n_colors=len(top_cats)),
    )

    ax.set_xlabel('Time period')
    ax.set_ylabel('Percentage of mentions')
    ax.set_title('Risk Category Share Over Time (normalized)', fontsize=12)
    ax.legend(title='Risk category', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_normalized.png')
    plt.close(fig)
    logger.info(f"  Saved normalized chart")

    # -------------------------------------------------------------------------
    # 4. By actor type (grouped bar)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # Pivot for grouped bar
    pivot_actor = by_actor.pivot(index='category', columns='actor_type', values='count')

    # Reorder by total
    pivot_actor['total'] = pivot_actor.sum(axis=1)
    pivot_actor = pivot_actor.sort_values('total', ascending=False).drop(columns='total')

    # Take top categories
    pivot_actor = pivot_actor.head(8)

    # Map labels
    pivot_actor.index = [CATEGORY_LABELS.get(c, c) for c in pivot_actor.index]
    pivot_actor.columns = [ACTOR_LABELS.get(c, c) for c in pivot_actor.columns]

    pivot_actor.plot(
        kind='barh',
        ax=ax,
        width=0.8,
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],
    )

    ax.set_xlabel('Number of mentions')
    ax.set_title('Risk Categories by Actor Type', fontsize=12)
    ax.legend(title='Actor type')
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / 'allvarliga_storningar_by_actor.png')
    plt.close(fig)
    logger.info(f"  Saved actor chart")

    logger.info("  All visualizations saved")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Analyze risks associated with "allvarliga störningar"',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/processed/bert_corpus.parquet'),
        help='Input corpus (default: data/processed/bert_corpus.parquet)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/bow_analysis'),
        help='Output directory (default: results/bow_analysis)',
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

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    logger.info(f"Loading corpus from {args.input}...")
    df = pd.read_parquet(args.input)
    logger.info(f"  Loaded {len(df):,} sentences")

    # Group to paragraphs
    paragraphs = group_sentences_to_paragraphs(df)

    # Filter to disruption paragraphs
    disruption_pattern = build_disruption_pattern()
    filtered = filter_disruption_paragraphs(paragraphs, disruption_pattern)

    if len(filtered) == 0:
        logger.warning("No paragraphs found with disruption phrases!")
        return 1

    # Analyze overall risk categories
    logger.info("Analyzing risk categories (overall)...")
    overall = analyze_risk_categories(filtered)
    overall_path = args.output_dir / 'allvarliga_storningar_risks.csv'
    overall.to_csv(overall_path, index=False)
    logger.info(f"  Saved overall results to {overall_path}")

    # Print top categories
    print("\n=== Top Risk Categories in 'Allvarliga Störningar' Paragraphs ===")
    print(overall.head(10).to_string(index=False))
    print()

    # Initialize for visualization
    by_actor = None
    by_wave = None

    # Analyze by actor_type
    if 'actor_type' in filtered.columns:
        logger.info("Analyzing risk categories by actor_type...")
        by_actor = analyze_risk_categories(filtered, group_col='actor_type')
        actor_path = args.output_dir / 'allvarliga_storningar_by_actor.csv'
        by_actor.to_csv(actor_path, index=False)
        logger.info(f"  Saved actor breakdown to {actor_path}")

        # Print summary
        print("=== Breakdown by Actor Type ===")
        pivot = by_actor.pivot(index='category', columns='actor_type', values='count')
        pivot = pivot.fillna(0).astype(int)
        pivot['total'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('total', ascending=False)
        print(pivot.head(10).to_string())
        print()

    # Analyze by wave
    if 'wave' in filtered.columns:
        logger.info("Analyzing risk categories by wave...")
        by_wave = analyze_risk_categories(filtered, group_col='wave')
        wave_path = args.output_dir / 'allvarliga_storningar_by_wave.csv'
        by_wave.to_csv(wave_path, index=False)
        logger.info(f"  Saved wave breakdown to {wave_path}")

        # Print summary
        print("=== Breakdown by Wave ===")
        pivot_wave = by_wave.pivot(index='category', columns='wave', values='count')
        pivot_wave = pivot_wave.fillna(0).astype(int)
        pivot_wave['total'] = pivot_wave.sum(axis=1)
        pivot_wave = pivot_wave.sort_values('total', ascending=False)
        print(pivot_wave.head(10).to_string())
        print()

    # Extract sample paragraphs
    logger.info(f"Extracting {args.sample_size} sample paragraphs...")
    sample = extract_sample_paragraphs(filtered, sample_size=args.sample_size)
    sample_path = args.output_dir / 'allvarliga_storningar_sample.csv'
    sample.to_csv(sample_path, index=False)
    logger.info(f"  Saved sample to {sample_path}")

    # Create visualizations
    if by_actor is not None and by_wave is not None:
        create_visualizations(overall, by_wave, by_actor, args.output_dir)

    # Summary statistics
    print("=== Summary Statistics ===")
    print(f"Total paragraphs in corpus: {len(paragraphs):,}")
    print(f"Paragraphs with disruption phrases: {len(filtered):,} ({len(filtered)/len(paragraphs)*100:.1f}%)")
    if 'actor_type' in filtered.columns:
        print(f"By actor: {filtered['actor_type'].value_counts().to_dict()}")
    if 'wave' in filtered.columns:
        print(f"By wave: {filtered['wave'].value_counts().sort_index().to_dict()}")

    logger.info("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
