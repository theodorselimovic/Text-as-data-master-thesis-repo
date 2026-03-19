#!/usr/bin/env python3
"""
RSA Analysis Visualization Script

Creates visualizations for:
1. Risk categories over time (share of total risk mentions)
2. Actor comparisons (kommun vs länsstyrelse)
3. Qualification distributions with statistical tests

Usage:
    python visualize_rsa_results.py \
        --results path/to/risk_context_analysis_by_document.csv \
        --output ./figures/

Requirements:
    pip install pandas matplotlib seaborn scipy
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set1")  # Use Set1 qualitative colormap

# =============================================================================
# TRANSLATION MAPPINGS
# =============================================================================

def translate_actor(actor: str) -> str:
    """Translate actor names from Swedish to English."""
    translations = {
        'kommun': 'Municipality',
        'lansstyrelse': 'Prefecture',
        'länsstyrelse': 'Prefecture',  # Handle both spellings
        'mcf': 'MCF',
        'MCF': 'MSB',
    }
    return translations.get(actor, actor)


# Actor colors (consistent across all visualizations)
ACTOR_COLORS = {
    'kommun': '#e41a1c',        # Red
    'lansstyrelse': '#377eb8',  # Blue
    'MCF': '#4daf4a',           # Green
}


def get_actor_color(actor: str) -> str:
    """Get color for actor, with fallback."""
    return ACTOR_COLORS.get(actor, '#999999')

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_results(results_path: Path) -> pd.DataFrame:
    """Load analysis results from CSV or parquet."""
    if results_path.suffix == '.parquet':
        df = pd.read_parquet(results_path)
    else:
        df = pd.read_csv(results_path)

    # Normalize column names
    if 'actor_type' in df.columns and 'actor' not in df.columns:
        df = df.rename(columns={'actor_type': 'actor'})

    print(f"Loaded {len(df)} documents")
    print(f"Columns: {list(df.columns)}")

    # Check for required columns
    if 'year' in df.columns:
        print(f"Years: {sorted(df['year'].dropna().unique())}")
    if 'actor' in df.columns:
        # Show both original and translated actor names
        actor_counts = df['actor'].value_counts().to_dict()
        translated_counts = {translate_actor(k): v for k, v in actor_counts.items()}
        print(f"Actors: {translated_counts}")

    return df


def get_risk_columns(df: pd.DataFrame) -> list:
    """Get list of risk CATEGORY columns (thematic categories only).

    Excludes qualification columns (risk_very_low, risk_high, etc.) and
    aggregates (risk_total).
    """
    # These are the thematic risk categories from the dictionary
    thematic_categories = [
        'risk_naturhot', 'risk_biologiska_hot', 'risk_olyckor',
        'risk_antagonistiska_hot', 'risk_cyber_hot', 'risk_sociala_risker',
        'risk_teknisk_infrastruktur', 'risk_brand', 'risk_miljö_klimat',
        'risk_ekonomi'
    ]
    return [col for col in thematic_categories if col in df.columns]


def get_qualification_columns(df: pd.DataFrame, concept: str) -> list:
    """Get qualification level columns for a concept.

    Excludes UNKNOWN - we assume unclassified mentions are methodology
    discussions rather than actual qualifications.
    """
    levels = ['very_low', 'low', 'medium', 'high', 'very_high', 'change', 'uncertainty', 'acceptability']
    return [f'{concept}_{level}' for level in levels if f'{concept}_{level}' in df.columns]


# =============================================================================
# RISK TRENDS OVER TIME
# =============================================================================

# Wave definitions
WAVE_LABELS = {
    1: '2015',
    2: '2019',
    3: '2023',
}


def map_year_to_wave(year) -> int:
    """Map year to wave number (excludes pre-2015)."""
    try:
        year = int(year)
    except (TypeError, ValueError):
        return None
    if year < 2015:
        return None  # Exclude pre-2015
    elif year <= 2018:
        return 1
    elif year <= 2022:
        return 2
    else:
        return 3


def plot_risk_trends_over_time(df: pd.DataFrame, output_dir: Path):
    """
    Plot risk categories as share of total mentions over time (by wave).

    Creates a line plot with wave on x-axis, share (%) on y-axis,
    one line per risk category.
    """
    if 'year' not in df.columns:
        print("Warning: No 'year' column found, skipping time trends")
        return

    risk_cols = get_risk_columns(df)
    if not risk_cols:
        print("Warning: No risk columns found")
        return

    # Add wave column and filter to 2015+
    df = df.copy()
    df['wave'] = df['year'].apply(map_year_to_wave)
    df = df[df['wave'].notna()]

    if len(df) == 0:
        print("Warning: No data from 2015+, skipping time trends")
        return

    # Remove 'risk_' prefix for cleaner labels
    categories = [col.replace('risk_', '') for col in risk_cols]

    # Group by wave and sum risk counts
    by_wave = df.groupby('wave')[risk_cols].sum()

    # Calculate shares (each category as % of total)
    wave_total = by_wave.sum(axis=1)
    wave_shares = by_wave.div(wave_total, axis=0) * 100

    # Rename columns for plotting
    wave_shares.columns = categories

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each category
    x_labels = [WAVE_LABELS.get(w, str(int(w))) for w in wave_shares.index]
    x_pos = range(len(x_labels))

    for category in categories:
        ax.plot(x_pos, wave_shares[category].values,
                marker='o', linewidth=2, markersize=6, label=category)

    ax.set_xlabel('Wave', fontsize=12)
    ax.set_ylabel('Share of total risk mentions (%)', fontsize=12)
    ax.set_title('Risk categories over time', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_trends_over_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'risk_trends_over_time.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: risk_trends_over_time.png/pdf")


def plot_risk_trends_stacked(df: pd.DataFrame, output_dir: Path):
    """
    Plot risk categories as stacked area chart over time (by wave).
    """
    if 'year' not in df.columns:
        return

    risk_cols = get_risk_columns(df)
    if not risk_cols:
        return

    # Add wave column and filter to 2015+
    df = df.copy()
    df['wave'] = df['year'].apply(map_year_to_wave)
    df = df[df['wave'].notna()]

    if len(df) == 0:
        return

    categories = [col.replace('risk_', '') for col in risk_cols]

    # Group by wave and sum
    by_wave = df.groupby('wave')[risk_cols].sum()
    wave_total = by_wave.sum(axis=1)
    wave_shares = by_wave.div(wave_total, axis=0) * 100
    wave_shares.columns = categories

    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x_labels = [WAVE_LABELS.get(w, str(int(w))) for w in wave_shares.index]
    x_pos = range(len(x_labels))

    ax.stackplot(x_pos, wave_shares.T, labels=categories, alpha=0.8)

    ax.set_xlabel('Wave', fontsize=12)
    ax.set_ylabel('Share (%)', fontsize=12)
    ax.set_title('Risk categories over time (stacked)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_trends_stacked.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'risk_trends_stacked.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: risk_trends_stacked.png/pdf")


# =============================================================================
# ACTOR COMPARISONS
# =============================================================================

def plot_actor_risk_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Compare risk category distributions between actors.

    Creates grouped bar chart with categories on x-axis,
    bars grouped by actor.
    """
    if 'actor' not in df.columns:
        print("Warning: No 'actor' column found, skipping actor comparison")
        return

    actors = df['actor'].unique()
    if len(actors) < 2:
        print("Warning: Only one actor found, skipping comparison")
        return

    risk_cols = get_risk_columns(df)
    if not risk_cols:
        return

    categories = [col.replace('risk_', '') for col in risk_cols]

    # Calculate mean per document for each actor
    actor_means = df.groupby('actor')[risk_cols].mean()

    # Translate actor names for display
    actor_labels = [translate_actor(a) for a in actors]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(categories))
    n_actors = len(actors)
    # Calculate width to avoid overlap: total group width ~0.8, divide by number of actors
    width = 0.8 / n_actors
    multiplier = 0

    for actor, label in zip(actors, actor_labels):
        # Center the group of bars around x position
        offset = width * (multiplier - (n_actors - 1) / 2)
        values = actor_means.loc[actor].values
        color = get_actor_color(actor)
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
        multiplier += 1

    ax.set_xlabel('Risk category', fontsize=12)
    ax.set_ylabel('Average mentions per document', fontsize=12)
    ax.set_title('Risk categories by actor', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'actor_risk_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'actor_risk_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: actor_risk_comparison.png/pdf")


def plot_actor_qualification_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Compare qualification distributions between actors.

    Creates a single figure with 3 subplots (one per concept: sannolikhet, konsekvens, risk),
    showing severity levels (very_low to very_high) normalized to 100%.
    """
    if 'actor' not in df.columns:
        return

    # Fixed order: Municipality, Prefecture, MSB (left to right)
    actor_order = ['kommun', 'lansstyrelse', 'MCF']
    actors = [a for a in actor_order if a in df['actor'].unique()]
    if len(actors) < 2:
        return

    severity_levels = ['very_low', 'low', 'medium', 'high', 'very_high']
    level_labels = {'very_low': 'Very Low', 'low': 'Low', 'medium': 'Medium',
                    'high': 'High', 'very_high': 'Very High'}

    concepts = ['sannolikhet', 'konsekvens', 'risk']
    concept_labels = {'sannolikhet': 'Probability', 'konsekvens': 'Consequence', 'risk': 'Risk'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    for ax, concept in zip(axes, concepts):
        severity_cols = [f'{concept}_{level}' for level in severity_levels
                        if f'{concept}_{level}' in df.columns]

        if not severity_cols:
            continue

        levels = [col.replace(f'{concept}_', '') for col in severity_cols]

        # Calculate totals per actor and normalize
        actor_totals = df.groupby('actor')[severity_cols].sum()
        actor_pcts = actor_totals.div(actor_totals.sum(axis=1), axis=0) * 100

        x = np.arange(len(levels))
        width = 0.25
        offsets = np.linspace(-width, width, len(actors))

        for i, actor in enumerate(actors):
            if actor not in actor_pcts.index:
                continue
            values = actor_pcts.loc[actor].values
            color = get_actor_color(actor)
            label = translate_actor(actor)
            ax.bar(x + offsets[i], values, width * 0.9, label=label, color=color, alpha=0.8)

        ax.set_title(concept_labels[concept], fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage')
        ax.set_xlabel('Qualification Level')
        ax.set_xticks(x)
        ax.set_xticklabels([level_labels.get(l, l) for l in levels], rotation=45, ha='right')
        ax.legend(fontsize=8)

    plt.suptitle('Qualification Distribution by Actor Type (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'qualification_by_actor.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'qualification_by_actor.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: qualification_by_actor.png/pdf")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize RSA analysis results'
    )

    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to analysis results (CSV or parquet)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./figures'),
        help='Output directory for figures'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n{'='*60}")
    print("Loading data...")
    print(f"{'='*60}\n")
    df = load_results(args.results)

    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}\n")

    # Time trends
    plot_risk_trends_over_time(df, args.output)
    plot_risk_trends_stacked(df, args.output)

    # Actor comparisons
    plot_actor_risk_comparison(df, args.output)
    plot_actor_qualification_comparison(df, args.output)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {args.output}")
    print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
