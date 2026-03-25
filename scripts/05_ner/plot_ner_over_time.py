#!/usr/bin/env python3
"""
Plot NER entity counts over time by actor type.

Creates three graphs showing unique LOC, ORG, and EVN counts:
- Municipalities: average per wave
- Länsstyrelsen: average per year
- MCF: average per year

Usage:
    python plot_ner_over_time.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PATH = Path("results/02_bert_analysis/ner/entities_by_document.csv")
OUTPUT_DIR = Path("results/02_bert_analysis/ner/visualizations")

# Wave mapping
def year_to_wave(year: int) -> int:
    if year < 2015:
        return 0
    elif 2015 <= year <= 2018:
        return 1
    elif 2019 <= year <= 2022:
        return 2
    else:
        return 3

WAVE_LABELS = {
    0: 'Pre-2015',
    1: '2015-2018',
    2: '2019-2022',
    3: '2023+'
}

# Actor type mapping for display
ACTOR_LABELS = {
    'kommun': 'Municipality',
    'lansstyrelse': 'Prefecture',
    'MCF': 'MCF'
}

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load data
    print(f"Loading data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH) if INPUT_PATH.suffix == '.parquet' else pd.read_csv(INPUT_PATH)

    # Add wave column
    df['wave'] = df['year'].apply(year_to_wave)

    # Normalize actor_type
    df['actor_type'] = df['actor_type'].replace({
        'kommun': 'kommun',
        'länsstyrelse': 'lansstyrelse',
        'lansstyrelse': 'lansstyrelse',
        'MCF': 'MCF'
    })

    print(f"Loaded {len(df)} documents")
    print(f"Actor types: {df['actor_type'].value_counts().to_dict()}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Entity types to plot
    entity_types = [
        ('LOC_unique', 'Unique Locations'),
        ('ORG_unique', 'Unique Organizations'),
        ('EVN_unique', 'Unique Events')
    ]

    # Create a figure for each entity type
    for col, title in entity_types:
        fig, ax = plt.subplots(figsize=(10, 6))

        # --- Municipalities: average per wave ---
        kommun_df = df[df['actor_type'] == 'kommun'].copy()
        if len(kommun_df) > 0:
            kommun_avg = kommun_df.groupby('wave')[col].mean()
            # Convert wave to x positions (use wave midpoint year for alignment)
            wave_positions = {0: 2012, 1: 2016.5, 2: 2020.5, 3: 2024}
            x_kommun = [wave_positions[w] for w in kommun_avg.index]
            ax.plot(x_kommun, kommun_avg.values, 'o-', label='Municipality (avg/wave)',
                   markersize=10, linewidth=2, color='#2ecc71')
            # Add wave labels
            for w, x, y in zip(kommun_avg.index, x_kommun, kommun_avg.values):
                ax.annotate(WAVE_LABELS[w], (x, y), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=8, color='#2ecc71')

        # --- Länsstyrelsen: average per year ---
        lan_df = df[df['actor_type'] == 'lansstyrelse'].copy()
        if len(lan_df) > 0:
            lan_avg = lan_df.groupby('year')[col].mean()
            ax.plot(lan_avg.index, lan_avg.values, 's-', label='Prefecture (avg/year)',
                   markersize=8, linewidth=2, color='#3498db')

        # --- MCF: one per year ---
        mcf_df = df[df['actor_type'] == 'MCF'].copy()
        if len(mcf_df) > 0:
            mcf_avg = mcf_df.groupby('year')[col].mean()
            ax.plot(mcf_avg.index, mcf_avg.values, '^-', label='MCF (per year)',
                   markersize=8, linewidth=2, color='#e74c3c')

        # Formatting
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f'Average {title} per Document', fontsize=12)
        ax.set_title(f'{title} Over Time by Actor Type', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set x-axis range
        ax.set_xlim(2010, 2026)

        # Save
        output_path = OUTPUT_DIR / f"{col.lower().replace('_unique', '')}_over_time.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

    # Also create a combined figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (col, title) in enumerate(entity_types):
        ax = axes[idx]

        # Municipalities
        kommun_df = df[df['actor_type'] == 'kommun'].copy()
        if len(kommun_df) > 0:
            kommun_avg = kommun_df.groupby('wave')[col].mean()
            wave_positions = {0: 2012, 1: 2016.5, 2: 2020.5, 3: 2024}
            x_kommun = [wave_positions[w] for w in kommun_avg.index]
            ax.plot(x_kommun, kommun_avg.values, 'o-', label='Municipality',
                   markersize=8, linewidth=2, color='#2ecc71')

        # Länsstyrelsen
        lan_df = df[df['actor_type'] == 'lansstyrelse'].copy()
        if len(lan_df) > 0:
            lan_avg = lan_df.groupby('year')[col].mean()
            ax.plot(lan_avg.index, lan_avg.values, 's-', label='Prefecture',
                   markersize=6, linewidth=2, color='#3498db')

        # MCF
        mcf_df = df[df['actor_type'] == 'MCF'].copy()
        if len(mcf_df) > 0:
            mcf_avg = mcf_df.groupby('year')[col].mean()
            ax.plot(mcf_avg.index, mcf_avg.values, '^-', label='MCF',
                   markersize=6, linewidth=2, color='#e74c3c')

        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel(f'Avg per Doc', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2010, 2026)

    fig.suptitle('NER Entity Counts Over Time by Actor Type', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    combined_path = OUTPUT_DIR / "ner_entities_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {combined_path}")
    plt.close(fig)

    print("\nDone!")


if __name__ == '__main__':
    main()
