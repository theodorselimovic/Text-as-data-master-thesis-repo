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
from scipy import stats
from collections import defaultdict

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
        'länsstyrelse': 'Prefecture',
        'mcf': 'MCF',
        'MCF': 'MCF',
    }
    return translations.get(actor, actor)

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_results(results_path: Path) -> pd.DataFrame:
    """Load analysis results from CSV or parquet."""
    if results_path.suffix == '.parquet':
        df = pd.read_parquet(results_path)
    else:
        df = pd.read_csv(results_path)

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

def plot_risk_trends_over_time(df: pd.DataFrame, output_dir: Path):
    """
    Plot risk categories as share of total mentions over time.

    Creates a line plot with year on x-axis, share (%) on y-axis,
    one line per risk category.
    """
    if 'year' not in df.columns:
        print("Warning: No 'year' column found, skipping time trends")
        return

    risk_cols = get_risk_columns(df)
    if not risk_cols:
        print("Warning: No risk columns found")
        return

    # Remove 'risk_' prefix for cleaner labels
    categories = [col.replace('risk_', '') for col in risk_cols]

    # Group by year and sum risk counts
    yearly = df.groupby('year')[risk_cols].sum()

    # Calculate shares (each category as % of total)
    yearly_total = yearly.sum(axis=1)
    yearly_shares = yearly.div(yearly_total, axis=0) * 100

    # Rename columns for plotting
    yearly_shares.columns = categories

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each category
    for category in categories:
        ax.plot(yearly_shares.index, yearly_shares[category],
                marker='o', linewidth=2, markersize=6, label=category)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Share of total risk mentions (%)', fontsize=12)
    ax.set_title('Risk categories over time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Rotate x-axis labels if many years
    if len(yearly_shares.index) > 10:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_trends_over_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'risk_trends_over_time.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: risk_trends_over_time.png/pdf")


def plot_risk_trends_stacked(df: pd.DataFrame, output_dir: Path):
    """
    Plot risk categories as stacked area chart over time.
    """
    if 'year' not in df.columns:
        return

    risk_cols = get_risk_columns(df)
    if not risk_cols:
        return

    categories = [col.replace('risk_', '') for col in risk_cols]

    # Group by year and sum
    yearly = df.groupby('year')[risk_cols].sum()
    yearly_total = yearly.sum(axis=1)
    yearly_shares = yearly.div(yearly_total, axis=0) * 100
    yearly_shares.columns = categories

    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.stackplot(yearly_shares.index, yearly_shares.T, labels=categories, alpha=0.8)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Share (%)', fontsize=12)
    ax.set_title('Risk categories over time (stacked)', fontsize=14, fontweight='bold')
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
        bars = ax.bar(x + offset, values, width, label=label, alpha=0.8)
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

    Creates TWO plots per concept (sannolikhet, konsekvens, risk):
    1. Severity levels (very_low to very_high) - normalized to 100%
    2. Other categories (change, uncertainty, acceptability)
    """
    if 'actor' not in df.columns:
        return

    actors = df['actor'].unique()
    if len(actors) < 2:
        return

    # Translate actor names for display
    actor_labels = [translate_actor(a) for a in actors]

    # Define the two category groups
    severity_levels = ['very_low', 'low', 'medium', 'high', 'very_high']
    other_categories = ['change', 'uncertainty', 'acceptability']

    # Concept name translations
    concept_translations = {
        'sannolikhet': 'Probability',
        'konsekvens': 'Consequence',
        'risk': 'Risk'
    }

    for concept in ['sannolikhet', 'konsekvens', 'risk']:
        concept_en = concept_translations[concept]

        # --- Chart 1: Severity levels (normalized to 100%) ---
        severity_cols = [f'{concept}_{level}' for level in severity_levels
                        if f'{concept}_{level}' in df.columns]

        if severity_cols:
            levels = [col.replace(f'{concept}_', '') for col in severity_cols]

            # Calculate totals per actor for severity levels only
            actor_totals = df.groupby('actor')[severity_cols].sum()

            # Normalize to 100% within severity levels
            actor_pcts = actor_totals.div(actor_totals.sum(axis=1), axis=0) * 100

            # Create grouped bar chart
            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(levels))
            n_actors = len(actors)
            # Calculate width to avoid overlap
            width = 0.8 / n_actors
            multiplier = 0

            for actor, label in zip(actors, actor_labels):
                # Center the group of bars around x position
                offset = width * (multiplier - (n_actors - 1) / 2)
                values = actor_pcts.loc[actor].values
                bars = ax.bar(x + offset, values, width, label=label, alpha=0.8)
                multiplier += 1

            ax.set_xlabel('Qualification level', fontsize=12)
            ax.set_ylabel('Share (%)', fontsize=12)
            ax.set_title(f'{concept_en} - severity levels by actor', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(levels, rotation=45, ha='right')
            ax.legend()

            plt.tight_layout()
            plt.savefig(output_dir / f'actor_{concept}_severity.png', dpi=150, bbox_inches='tight')
            plt.savefig(output_dir / f'actor_{concept}_severity.pdf', bbox_inches='tight')
            plt.close()

            print(f"Saved: actor_{concept}_severity.png/pdf")

        # --- Chart 2: Other categories (change, uncertainty, acceptability) ---
        other_cols = [f'{concept}_{cat}' for cat in other_categories
                     if f'{concept}_{cat}' in df.columns]

        if other_cols:
            categories = [col.replace(f'{concept}_', '') for col in other_cols]

            # Calculate totals per actor
            actor_totals = df.groupby('actor')[other_cols].sum()

            # For other categories, show as rate per document (not normalized to 100%)
            doc_counts = df.groupby('actor').size()
            actor_rates = actor_totals.div(doc_counts, axis=0)

            # Create grouped bar chart
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(categories))
            n_actors = len(actors)
            # Calculate width to avoid overlap
            width = 0.8 / n_actors
            multiplier = 0

            for actor, label in zip(actors, actor_labels):
                # Center the group of bars around x position
                offset = width * (multiplier - (n_actors - 1) / 2)
                values = actor_rates.loc[actor].values
                bars = ax.bar(x + offset, values, width, label=label, alpha=0.8)
                multiplier += 1

            ax.set_xlabel('Category', fontsize=12)
            ax.set_ylabel('Average per document', fontsize=12)
            ax.set_title(f'{concept_en} - other categories by actor', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.legend()

            plt.tight_layout()
            plt.savefig(output_dir / f'actor_{concept}_other.png', dpi=150, bbox_inches='tight')
            plt.savefig(output_dir / f'actor_{concept}_other.pdf', bbox_inches='tight')
            plt.close()

            print(f"Saved: actor_{concept}_other.png/pdf")


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def chi_square_pairwise_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Perform pairwise Chi-square tests comparing actors.

    For each pair of actors, tests whether the distribution of qualifications
    differs significantly. Uses Bonferroni correction for multiple comparisons.

    A Chi-square test compares observed frequencies to expected frequencies
    under the null hypothesis of independence. For a 2xK contingency table
    (2 actors x K categories), it tests whether the distribution across
    categories differs between the two actors.
    """
    if 'actor' not in df.columns:
        return

    actors = sorted(df['actor'].unique())
    if len(actors) < 2:
        return

    # Generate all pairs
    from itertools import combinations
    actor_pairs = list(combinations(actors, 2))
    n_comparisons = len(actor_pairs)

    # Bonferroni correction: adjust alpha for multiple comparisons
    alpha = 0.05
    bonferroni_alpha = alpha / n_comparisons

    results = []

    def run_pairwise_test(df_subset, cols, test_name, actor1, actor2):
        """Run Chi-square test for a specific pair of actors."""
        contingency = df_subset[df_subset['actor'].isin([actor1, actor2])].groupby('actor')[cols].sum()

        # Remove columns with all zeros
        contingency = contingency.loc[:, (contingency > 0).any()]

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return None

        # Translate actor names for display
        label1 = translate_actor(actor1)
        label2 = translate_actor(actor2)

        try:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            return {
                'Test': test_name,
                'Comparison': f'{label1} vs {label2}',
                'Chi-square': chi2,
                'p-value': p,
                'p-value (Bonferroni)': min(p * n_comparisons, 1.0),
                'Degrees of freedom': dof,
                'Significant (p<0.05)': 'Yes' if p < alpha else 'No',
                'Significant (Bonferroni)': 'Yes' if p < bonferroni_alpha else 'No'
            }
        except Exception as e:
            print(f"  Warning: Could not compute Chi-square for {test_name} {label1} vs {label2}: {e}")
            return None

    # Test risk categories for each pair
    risk_cols = get_risk_columns(df)
    if risk_cols:
        for actor1, actor2 in actor_pairs:
            result = run_pairwise_test(df, risk_cols, 'Risk categories', actor1, actor2)
            if result:
                results.append(result)

    # Test each qualification concept for each pair
    for concept in ['sannolikhet', 'konsekvens', 'risk']:
        qual_cols = get_qualification_columns(df, concept)
        if qual_cols:
            for actor1, actor2 in actor_pairs:
                result = run_pairwise_test(df, qual_cols, f'{concept.capitalize()} qualifications', actor1, actor2)
                if result:
                    results.append(result)

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / 'chi_square_pairwise_tests.csv', index=False)

        # Create detailed text report
        report = [
            "Pairwise Chi-Square Tests: Actor Comparison",
            "=" * 60,
            "",
            f"Number of actor pairs: {n_comparisons}",
            f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}",
            "",
            "Interpretation:",
            "- Chi-square tests whether the distribution of categories differs between two groups",
            "- A significant result means the groups have different patterns of category usage",
            "- Bonferroni correction adjusts for multiple comparisons to reduce false positives",
            "",
            "=" * 60,
            ""
        ]

        # Group by test type
        for test_name in results_df['Test'].unique():
            report.append(f"\n{test_name}:")
            report.append("-" * 40)

            test_results = results_df[results_df['Test'] == test_name]
            for _, r in test_results.iterrows():
                report.append(f"\n  {r['Comparison']}:")
                report.append(f"    Chi-square: {r['Chi-square']:.2f}")
                report.append(f"    p-value: {r['p-value']:.6f}")
                report.append(f"    p-value (Bonferroni-adjusted): {r['p-value (Bonferroni)']:.6f}")
                report.append(f"    Degrees of freedom: {r['Degrees of freedom']}")
                report.append(f"    Significant (raw p<0.05): {r['Significant (p<0.05)']}")
                report.append(f"    Significant (Bonferroni): {r['Significant (Bonferroni)']}")

        with open(output_dir / 'chi_square_pairwise_tests.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"Saved: chi_square_pairwise_tests.csv/txt")
        print(f"\nPairwise Chi-Square Test Results (Bonferroni alpha = {bonferroni_alpha:.4f}):")
        print(results_df[['Test', 'Comparison', 'Chi-square', 'p-value', 'Significant (Bonferroni)']].to_string(index=False))


# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================

def create_summary_dashboard(df: pd.DataFrame, output_dir: Path):
    """
    Create a summary dashboard with multiple plots.
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. Risk category totals (bar chart)
    ax1 = fig.add_subplot(2, 2, 1)
    risk_cols = get_risk_columns(df)
    if risk_cols:
        totals = df[risk_cols].sum().sort_values(ascending=True)
        categories = [col.replace('risk_', '') for col in totals.index]
        ax1.barh(categories, totals.values, color=sns.color_palette("Set1", len(categories)))
        ax1.set_xlabel('Total mentions')
        ax1.set_title('Risk categories (total)', fontweight='bold')

    # 2. Sannolikhet/Probability distribution (pie chart)
    ax2 = fig.add_subplot(2, 2, 2)
    sann_cols = get_qualification_columns(df, 'sannolikhet')
    if sann_cols:
        totals = df[sann_cols].sum()
        totals = totals[totals > 0]
        labels = [col.replace('sannolikhet_', '') for col in totals.index]
        ax2.pie(totals.values, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette("Set1", len(labels)))
        ax2.set_title('Probability - distribution', fontweight='bold')

    # 3. Konsekvens/Consequence distribution (pie chart)
    ax3 = fig.add_subplot(2, 2, 3)
    kons_cols = get_qualification_columns(df, 'konsekvens')
    if kons_cols:
        totals = df[kons_cols].sum()
        totals = totals[totals > 0]
        labels = [col.replace('konsekvens_', '') for col in totals.index]
        ax3.pie(totals.values, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette("Set1", len(labels)))
        ax3.set_title('Consequence - distribution', fontweight='bold')

    # 4. Documents per year/actor
    ax4 = fig.add_subplot(2, 2, 4)
    if 'year' in df.columns and 'actor' in df.columns:
        # Translate actor names in pivot table columns
        pivot = df.groupby(['year', 'actor']).size().unstack(fill_value=0)
        pivot.columns = [translate_actor(col) for col in pivot.columns]
        pivot.plot(kind='bar', ax=ax4, alpha=0.8, width=0.8, color=sns.color_palette("Set1", len(pivot.columns)))
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of documents')
        ax4.set_title('Documents by year and actor', fontweight='bold')
        ax4.legend(title='Actor')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    elif 'year' in df.columns:
        yearly_counts = df.groupby('year').size()
        ax4.bar(yearly_counts.index, yearly_counts.values, alpha=0.8,
                color=sns.color_palette("Set1", 1)[0])
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of documents')
        ax4.set_title('Documents by year', fontweight='bold')

    plt.suptitle('RSA Analysis - Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'summary_dashboard.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: summary_dashboard.png/pdf")


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

    # Statistical tests (pairwise Chi-square with Bonferroni correction)
    chi_square_pairwise_comparison(df, args.output)

    # Summary dashboard
    create_summary_dashboard(df, args.output)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {args.output}")
    print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
