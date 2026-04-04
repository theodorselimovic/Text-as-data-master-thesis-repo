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
        Columns: risk, eta_slope, p_value, mean_proportion,
                 eta_w1, eta_w2, eta_w3, convergence_magnitude
        Sorted by eta_slope ascending (most negative = strongest convergence)
    """
    results = []

    for risk, group in eta_df.groupby('risk'):
        if len(group) < 2:
            continue

        waves = group['wave'].values
        eta_values = group['eta_squared'].values

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(waves, eta_values)

        # Get per-wave values
        wave_eta = dict(zip(group['wave'], group['eta_squared']))
        mean_prop = group['mean_proportion'].mean()
        mean_var = group['ss_total'].mean()

        # Convergence magnitude: how much did eta drop, weighted by importance
        eta_w1 = wave_eta.get(1, np.nan)
        eta_w3 = wave_eta.get(3, np.nan)
        if not np.isnan(eta_w1) and not np.isnan(eta_w3):
            eta_drop = eta_w1 - eta_w3  # positive = convergence
        else:
            eta_drop = -slope * 2  # approximate from slope

        results.append({
            'risk': risk,
            'eta_slope': slope,
            'p_value': p_value,
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


WAVE_TO_YEAR = {
    1: 2015,  # Wave 1: 2015-2018 → map to start year
    2: 2019,  # Wave 2: 2019-2022 → map to start year
    3: 2023,  # Wave 3: 2023+ → map to start year
}


def compute_actor_profiles(
    df: pd.DataFrame,
    risk_cols: list[str],
    top_risks: list[str],
    waves: list[int],
) -> pd.DataFrame:
    """
    Compute mean proportion per actor per time period for top risks.

    Uses different temporal granularity by actor:
    - Municipality: wave-based, mapped to start year (2015, 2019, 2023)
    - Prefecture/MSB: year-on-year (individual years)

    All actors use years on x-axis for consistent visualization.

    Useful for understanding HOW actors converged.
    """
    records = []

    for actor in df['actor'].unique():
        actor_name = translate_actor(actor)
        df_actor_full = df[df['actor'] == actor]

        if actor_name == 'Municipality':
            # Wave-based for municipalities, mapped to start year
            for wave in waves:
                df_slice = df_actor_full[df_actor_full['wave'] == wave]
                if len(df_slice) == 0:
                    continue
                year = WAVE_TO_YEAR.get(wave, 2015 + (wave - 1) * 4)
                for risk in top_risks:
                    if risk in df_slice.columns:
                        records.append({
                            'time_period': f'W{wave} ({year})',
                            'time_numeric': year,
                            'time_type': 'wave',
                            'actor': actor_name,
                            'risk': risk,
                            'mean_proportion': df_slice[risk].mean(),
                            'n_docs': len(df_slice),
                        })
        else:
            # Year-on-year for Prefecture and MSB
            for year in sorted(df_actor_full['year'].unique()):
                df_slice = df_actor_full[df_actor_full['year'] == year]
                if len(df_slice) == 0:
                    continue
                for risk in top_risks:
                    if risk in df_slice.columns:
                        records.append({
                            'time_period': str(int(year)),
                            'time_numeric': year,
                            'time_type': 'year',
                            'actor': actor_name,
                            'risk': risk,
                            'mean_proportion': df_slice[risk].mean(),
                            'n_docs': len(df_slice),
                        })

    return pd.DataFrame(records)


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_convergence_drivers_bar(
    drivers_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 25,
) -> None:
    """
    Horizontal bar chart: eta² change from Wave 1 to Wave 3.
    """
    # Filter to risks with data in both waves
    plot_df = drivers_df.dropna(subset=['eta_w1', 'eta_w3']).head(top_n).copy()

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
    ax.set_yticklabels(plot_df['risk'], fontsize=9)
    ax.set_xlabel('Eta² (variance explained by actor type)', fontsize=11)
    ax.set_title('Top Convergence Drivers: Actor distinctiveness decreased',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)

    # Add arrow annotations for top 5
    for i, (_, row) in enumerate(plot_df.head(5).iterrows()):
        drop = row['eta_w1'] - row['eta_w3']
        ax.annotate(f'↓{drop:.0%}',
                    xy=(max(row['eta_w1'], row['eta_w3']) + 0.02, i),
                    fontsize=9, color='darkgreen', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_drivers_bar.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'convergence_drivers_bar.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: convergence_drivers_bar.png/pdf")


def plot_convergence_trends(
    drivers_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 12,
) -> None:
    """
    Line plot: eta² trend over waves for top convergence drivers.
    """
    top_drivers = drivers_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    waves = [1, 2, 3]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(top_drivers)))

    for idx, (_, row) in enumerate(top_drivers.iterrows()):
        eta_values = [row['eta_w1'], row['eta_w2'], row['eta_w3']]
        valid_waves = [w for w, e in zip(waves, eta_values) if not np.isnan(e)]
        valid_eta = [e for e in eta_values if not np.isnan(e)]

        ax.plot(valid_waves, valid_eta, 'o-',
                label=row['risk'],
                color=colors[idx], linewidth=2, markersize=8)

    ax.set_xlabel('Wave', fontsize=11)
    ax.set_ylabel('Eta² (variance explained by actor)', fontsize=11)
    ax.set_title('Convergence Drivers: Decreasing actor distinctiveness over time',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(waves)
    ax.set_xticklabels([WAVE_LABELS[w] for w in waves])
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

    # Add annotation
    ax.annotate('↓ Lower eta² = actors more similar',
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_trends.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'convergence_trends.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: convergence_trends.png/pdf")


def plot_convergence_heatmap(
    drivers_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 25,
) -> None:
    """
    Heatmap showing eta² per wave for top convergence drivers.
    """
    top_drivers = drivers_df.head(top_n).copy()

    # Create matrix
    matrix = top_drivers[['risk', 'eta_w1', 'eta_w2', 'eta_w3']].copy()
    matrix = matrix.set_index('risk')
    matrix.columns = ['Wave 1', 'Wave 2', 'Wave 3']

    fig, ax = plt.subplots(figsize=(8, max(8, top_n * 0.35)))

    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                vmin=0, vmax=1, ax=ax,
                cbar_kws={'label': 'Eta² (actor distinctiveness)'})

    ax.set_title('Convergence Drivers: Eta² by wave (green = similar)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'convergence_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: convergence_heatmap.png/pdf")


def plot_actor_convergence(
    actor_profiles: pd.DataFrame,
    risk: str,
    output_dir: Path,
) -> None:
    """
    Line plot showing how actors converged on a specific risk.

    Municipality uses wave averages (mapped to 2015, 2019, 2023).
    Prefecture/MSB use individual years.
    """
    risk_data = actor_profiles[actor_profiles['risk'] == risk]

    if len(risk_data) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for actor in sorted(risk_data['actor'].unique()):
        actor_data = risk_data[risk_data['actor'] == actor].sort_values('time_numeric')
        color = ACTOR_COLORS.get(actor, 'gray')

        # Use different markers for wave vs year data
        marker = 'o' if actor == 'Municipality' else 's'
        ax.plot(actor_data['time_numeric'], actor_data['mean_proportion'], f'{marker}-',
                label=actor, color=color, linewidth=2, markersize=8)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Mean proportion of risk mentions', fontsize=11)
    ax.set_title(f'Actor Convergence: {risk}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    safe_name = risk.replace('/', '_').replace(' ', '_')
    plt.savefig(output_dir / f'actor_convergence_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_convergence_grid(
    actor_profiles: pd.DataFrame,
    top_risks: list[str],
    output_dir: Path,
) -> None:
    """
    Grid of small multiples showing convergence for top risks.

    All actors plotted on year axis:
    - Municipality: wave averages at 2015, 2019, 2023 (circles)
    - Prefecture/MSB: individual years (squares)
    """
    n_risks = min(len(top_risks), 9)
    n_cols = 3
    n_rows = (n_risks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_risks > 1 else [axes]

    for idx, risk in enumerate(top_risks[:n_risks]):
        ax = axes[idx]
        risk_data = actor_profiles[actor_profiles['risk'] == risk]

        for actor in sorted(risk_data['actor'].unique()):
            actor_data = risk_data[risk_data['actor'] == actor].sort_values('time_numeric')
            color = ACTOR_COLORS.get(actor, 'gray')

            # Different markers: circle for wave averages, square for yearly data
            marker = 'o' if actor == 'Municipality' else 's'
            ax.plot(actor_data['time_numeric'], actor_data['mean_proportion'], f'{marker}-',
                    label=actor, color=color, linewidth=2, markersize=5)

        ax.set_title(risk, fontsize=11, fontweight='bold')
        ax.set_ylim(bottom=0)

        if idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    # Add shared axis labels
    fig.supxlabel('Year', fontsize=12)
    fig.supylabel('Mean proportion of risk mentions', fontsize=12)

    # Hide unused subplots
    for idx in range(n_risks, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('How Actors Converged on Top Risks\n(Municipality: wave averages, Prefecture/MSB: yearly)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig(output_dir / 'convergence_grid.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'convergence_grid.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: convergence_grid.png/pdf")


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
    report.append(f"{'Rank':<5} {'Risk':<28} {'Slope':>8} {'p':>8} {'Eta W1':>8} {'Eta W3':>8} {'Drop':>8}")
    report.append("-" * 75)

    for idx, row in drivers_df.head(top_n).iterrows():
        eta_w1 = f"{row['eta_w1']:.2f}" if not np.isnan(row['eta_w1']) else "n/a"
        eta_w3 = f"{row['eta_w3']:.2f}" if not np.isnan(row['eta_w3']) else "n/a"
        drop = f"{row['eta_drop']:.2f}" if not np.isnan(row['eta_drop']) else "n/a"

        report.append(f"{idx+1:<5} {row['risk']:<28} {row['eta_slope']:>+8.4f} {row['p_value']:>8.3f} {eta_w1:>8} {eta_w3:>8} {drop:>8}")

    # Summary stats
    n_converging = (drivers_df['eta_slope'] < 0).sum()
    n_diverging = (drivers_df['eta_slope'] > 0).sum()
    n_sig_converging = ((drivers_df['eta_slope'] < 0) & (drivers_df['p_value'] < 0.05)).sum()

    report.append(f"\n{'=' * 60}")
    report.append("SUMMARY")
    report.append(f"{'=' * 60}")
    report.append(f"\nTotal risks analyzed: {len(drivers_df)}")
    report.append(f"Risks with decreasing eta² (convergence): {n_converging}")
    report.append(f"Risks with increasing eta² (divergence): {n_diverging}")
    report.append(f"Significant convergence (p<0.05): {n_sig_converging}")

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

    # Compute actor profiles for top risks
    top_risks = drivers_df.head(args.top_n)['risk'].tolist()
    actor_profiles = compute_actor_profiles(df, risk_cols, top_risks, args.waves)
    actor_profiles.to_csv(args.output / 'actor_profiles_top_risks.csv', index=False)

    # Visualizations
    print(f"\n{'=' * 40}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 40}")

    plot_convergence_drivers_bar(drivers_df, args.output, args.top_n)
    plot_convergence_trends(drivers_df, args.output, min(args.top_n, 12))
    plot_convergence_heatmap(drivers_df, args.output, args.top_n)
    plot_top_convergence_grid(actor_profiles, top_risks[:9], args.output)

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
