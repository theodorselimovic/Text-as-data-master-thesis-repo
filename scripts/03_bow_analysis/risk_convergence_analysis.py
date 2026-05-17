#!/usr/bin/env python3
"""
Risk Convergence Analysis (Eta-squared Decomposition)

Identifies which individual risks drive convergence between actors
(Municipality, Prefecture, MSB) across waves by decomposing variance.

Method:
    For each risk, compute eta² = SS_between / SS_total per wave.
    This is the proportion of variance explained by actor type.

    Risks where eta² DECREASES over waves are convergence drivers —
    actors were different but became similar on that risk.

    This directly decomposes PERMANOVA R² into per-risk contributions.

Input:  term_document_matrix.csv (individual risk counts per document)
Output: convergence drivers ranking, eta² trends, visualizations

Usage:
    python risk_convergence_analysis.py \\
        --input results/01_bow_analysis/term_matrices/term_document_matrix.csv \\
        --output results/01_bow_analysis/convergence/

    python risk_convergence_analysis.py \\
        --input results/01_bow_analysis/term_matrices/term_document_matrix.csv \\
        --output results/01_bow_analysis/convergence/ \\
        --top-n 30 --verbose

Requirements:
    pip install pandas numpy matplotlib seaborn scipy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set1")

METADATA_COLS = ['file', 'actor', 'entity', 'year', 'wave', 'total_risk_mentions']

ACTOR_TRANSLATIONS = {
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

WAVE_LABELS = {
    1: 'Wave 1\n(2015-2018)',
    2: 'Wave 2\n(2019-2022)',
    3: 'Wave 3\n(2023+)',
}

WAVE_RANGES = {
    1: '2015-2018',
    2: '2019-2022',
    3: '2023+',
}


# Import translations
sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries.risk_translations import (
    translate_term,
    translate_actor as _translate_actor_base,
)


def translate_actor(actor: str) -> str:
    """Translate actor names from Swedish to English."""
    return ACTOR_TRANSLATIONS.get(actor, actor)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_normalize(input_path: Path, waves: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """
    Load term-document matrix, filter to waves, normalize to proportions.

    Returns
    -------
    tuple of (pd.DataFrame, list[str])
        (normalized dataframe, risk_column_names)
    """
    df = pd.read_csv(input_path)

    # Identify risk columns (everything except metadata)
    risk_cols = [c for c in df.columns if c not in METADATA_COLS]

    print(f"  Loaded {len(df)} documents, {len(risk_cols)} risk terms")

    # Filter to waves
    df = df[df['wave'].isin(waves)].copy()
    print(f"  After wave filter: {len(df)} documents")

    # Remove zero-mention rows
    row_totals = df[risk_cols].sum(axis=1)
    zero_mask = row_totals == 0
    if zero_mask.any():
        print(f"  Removing {zero_mask.sum()} zero-mention documents")
        df = df[~zero_mask].copy()
        row_totals = df[risk_cols].sum(axis=1)

    # Normalize to proportions
    df[risk_cols] = df[risk_cols].div(row_totals, axis=0)

    print(f"  Final: {len(df)} documents")
    actor_counts = df['actor'].value_counts().to_dict()
    print(f"  By actor: {', '.join(f'{translate_actor(k)}: {v}' for k, v in actor_counts.items())}")

    return df.reset_index(drop=True), risk_cols


# =============================================================================
# ETA-SQUARED COMPUTATION
# =============================================================================

def compute_eta_squared(
    df: pd.DataFrame,
    risk_cols: list[str],
    waves: list[int],
) -> pd.DataFrame:
    """
    Compute eta² (variance explained by actor) for each risk per wave.

    Eta² = SS_between / SS_total

    This is equivalent to a one-way ANOVA effect size for actor type.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized document data
    risk_cols : list[str]
        Column names for risk terms
    waves : list[int]
        Waves to analyze

    Returns
    -------
    pd.DataFrame
        Columns: risk, wave, eta_squared, mean_proportion, ss_total
    """
    results = []

    for wave in waves:
        df_wave = df[df['wave'] == wave]
        n_docs = len(df_wave)

        if n_docs < 10:
            continue

        for risk in risk_cols:
            values = df_wave[risk].values
            actors = df_wave['actor'].values

            # Grand mean
            grand_mean = np.mean(values)

            # Total sum of squares
            ss_total = np.sum((values - grand_mean) ** 2)

            if ss_total < 1e-12:  # Skip risks with no variance
                continue

            # Between-group sum of squares
            ss_between = 0
            for actor in df_wave['actor'].unique():
                mask = actors == actor
                n_actor = mask.sum()
                actor_mean = values[mask].mean()
                ss_between += n_actor * (actor_mean - grand_mean) ** 2

            # Eta-squared
            eta_sq = ss_between / ss_total

            results.append({
                'risk': risk,
                'wave': wave,
                'eta_squared': eta_sq,
                'mean_proportion': grand_mean,
                'ss_total': ss_total,
                'n_docs': n_docs,
            })

    return pd.DataFrame(results)


def compute_convergence_drivers(eta_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each risk, compute trend in eta² over waves.

    Convergence drivers = risks with DECREASING eta² (negative slope).
    This means actor type explains less variance → actors becoming similar.

    Returns
    -------
    pd.DataFrame
        Columns: risk, eta_slope, mean_proportion,
                 eta_w1, eta_w2, eta_w3, eta_drop
        Sorted by eta_slope ascending (most negative = strongest convergence)
    """
    results = []

    for risk, group in eta_df.groupby('risk'):
        if len(group) < 2:
            continue

        waves = group['wave'].values
        eta_values = group['eta_squared'].values

        # Linear regression (slope only, no inference)
        slope, intercept, r_value, _, _ = stats.linregress(waves, eta_values)

        # Get per-wave values
        wave_eta = dict(zip(group['wave'], group['eta_squared']))
        mean_prop = group['mean_proportion'].mean()
        mean_var = group['ss_total'].mean()

        # Convergence magnitude: how much did eta drop
        eta_w1 = wave_eta.get(1, np.nan)
        eta_w3 = wave_eta.get(3, np.nan)
        if not np.isnan(eta_w1) and not np.isnan(eta_w3):
            eta_drop = eta_w1 - eta_w3  # positive = convergence
        else:
            eta_drop = -slope * 2  # approximate from slope

        results.append({
            'risk': risk,
            'eta_slope': slope,
            'mean_proportion': mean_prop,
            'mean_variance': mean_var,
            'eta_w1': wave_eta.get(1, np.nan),
            'eta_w2': wave_eta.get(2, np.nan),
            'eta_w3': wave_eta.get(3, np.nan),
            'eta_drop': eta_drop,
        })

    df = pd.DataFrame(results)

    # Sort by eta_slope ascending (most negative = strongest convergence)
    df = df.sort_values('eta_slope', ascending=True).reset_index(drop=True)

    return df


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_convergence_drivers_bar(
    drivers_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """
    Horizontal bar chart: eta² change from Wave 1 to Wave 3.
    Shows top N risks by eta drop (largest convergence).
    """
    # Filter to risks with data in both waves, sort by eta_drop
    plot_df = drivers_df.dropna(subset=['eta_w1', 'eta_w3']).copy()
    plot_df = plot_df.sort_values('eta_drop', ascending=False).head(top_n)

    if len(plot_df) == 0:
        print("  Skipping bar plot: no risks with W1 and W3 data")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    y_pos = np.arange(len(plot_df))

    # Plot W1 and W3 eta values
    bar_height = 0.35
    ax.barh(y_pos - bar_height/2, plot_df['eta_w1'], bar_height,
            label='Wave 1 (2015-18)', color='#fc8d62', alpha=0.8)
    ax.barh(y_pos + bar_height/2, plot_df['eta_w3'], bar_height,
            label='Wave 3 (2023+)', color='#66c2a5', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([translate_term(r) for r in plot_df['risk']], fontsize=9)
    ax.set_xlabel('Eta² (variance explained by actor type)', fontsize=11)
    ax.set_title('Top Convergence Drivers: Risks where actors became most similar',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)

    # Add arrow annotations for all shown risks
    for i, (_, row) in enumerate(plot_df.iterrows()):
        drop = row['eta_w1'] - row['eta_w3']
        ax.annotate(f'↓{drop:.0%}',
                    xy=(max(row['eta_w1'], row['eta_w3']) + 0.02, i),
                    fontsize=9, color='darkgreen', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_drivers_bar.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'convergence_drivers_bar.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: convergence_drivers_bar.png/pdf")


# =============================================================================
# REPORT
# =============================================================================

def generate_report(
    drivers_df: pd.DataFrame,
    eta_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 30,
) -> None:
    """Generate text report summarizing convergence drivers."""
    report = []
    report.append("=" * 70)
    report.append("RISK CONVERGENCE ANALYSIS (Eta-squared Decomposition)")
    report.append("=" * 70)

    report.append("\nMethod:")
    report.append("  For each risk, computed eta² = SS_between / SS_total per wave")
    report.append("  Eta² = proportion of variance explained by actor type")
    report.append("  Decreasing eta² = actors becoming less distinct = convergence")
    report.append("  This directly decomposes PERMANOVA R² into per-risk contributions")

    report.append(f"\n{'=' * 60}")
    report.append(f"TOP {min(top_n, len(drivers_df))} CONVERGENCE DRIVERS")
    report.append("(risks where actors became MORE similar)")
    report.append(f"{'=' * 60}")
    report.append("")
    report.append(f"{'Rank':<5} {'Risk':<28} {'Slope':>8} {'Eta W1':>8} {'Eta W3':>8} {'Drop':>8}")
    report.append("-" * 67)

    for idx, row in drivers_df.head(top_n).iterrows():
        eta_w1 = f"{row['eta_w1']:.2f}" if not np.isnan(row['eta_w1']) else "n/a"
        eta_w3 = f"{row['eta_w3']:.2f}" if not np.isnan(row['eta_w3']) else "n/a"
        drop = f"{row['eta_drop']:.2f}" if not np.isnan(row['eta_drop']) else "n/a"

        report.append(f"{idx+1:<5} {row['risk']:<28} {row['eta_slope']:>+8.4f} {eta_w1:>8} {eta_w3:>8} {drop:>8}")

    # Summary stats
    n_converging = (drivers_df['eta_slope'] < 0).sum()
    n_diverging = (drivers_df['eta_slope'] > 0).sum()

    report.append(f"\n{'=' * 60}")
    report.append("SUMMARY")
    report.append(f"{'=' * 60}")
    report.append(f"\nTotal risks analyzed: {len(drivers_df)}")
    report.append(f"Risks with decreasing eta² (convergence): {n_converging}")
    report.append(f"Risks with increasing eta² (divergence): {n_diverging}")

    # Interpretation
    report.append(f"\n{'=' * 60}")
    report.append("INTERPRETATION")
    report.append(f"{'=' * 60}")
    report.append("""
Eta² ranges from 0 (actors identical) to 1 (actors completely distinct).

Top convergence drivers are risks where:
  - Actors WERE different (high eta² in early waves)
  - Actors BECAME similar (low eta² in later waves)

These risks explain WHY overall PERMANOVA R² is decreasing —
they are the specific areas where actor-type distinctiveness collapsed.
""")

    report_text = '\n'.join(report)
    report_path = output_dir / 'convergence_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Saved: convergence_report.txt")
    print(f"\n{report_text}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Identify convergence-driving risks via eta² decomposition'
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to term_document_matrix.csv'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./results/01_bow_analysis/convergence'),
        help='Output directory'
    )

    parser.add_argument(
        '--waves',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='Wave numbers to analyze (default: 1 2 3)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=25,
        help='Number of top risks to show in visualizations (default: 25)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages'
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RISK CONVERGENCE ANALYSIS (Eta-squared Decomposition)")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {args.input}")
    df, risk_cols = load_and_normalize(args.input, args.waves)

    # Compute eta² per risk per wave
    print(f"\n{'=' * 40}")
    print("COMPUTING ETA-SQUARED")
    print(f"{'=' * 40}")
    print("  (variance explained by actor type, per risk)")

    eta_df = compute_eta_squared(df, risk_cols, args.waves)
    print(f"  Computed eta² for {len(eta_df)} (risk, wave) combinations")

    # Compute convergence drivers
    print(f"\n{'=' * 40}")
    print("IDENTIFYING CONVERGENCE DRIVERS")
    print(f"{'=' * 40}")

    drivers_df = compute_convergence_drivers(eta_df)
    print(f"  Analyzed trends for {len(drivers_df)} risks")

    n_converging = (drivers_df['eta_slope'] < 0).sum()
    n_diverging = (drivers_df['eta_slope'] > 0).sum()
    print(f"  Converging: {n_converging}, Diverging: {n_diverging}")

    # Save CSVs
    eta_df.to_csv(args.output / 'eta_squared_by_wave.csv', index=False)
    drivers_df.to_csv(args.output / 'convergence_drivers.csv', index=False)
    print(f"\n  Saved: eta_squared_by_wave.csv, convergence_drivers.csv")

    # Visualizations
    print(f"\n{'=' * 40}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 40}")

    plot_convergence_drivers_bar(drivers_df, args.output, top_n=10)

    # Report
    print(f"\n{'=' * 40}")
    print("REPORT")
    print(f"{'=' * 40}")

    generate_report(drivers_df, eta_df, args.output, args.top_n)

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {args.output}")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
