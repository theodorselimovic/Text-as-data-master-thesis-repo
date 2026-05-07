#!/usr/bin/env python3
"""
Security Adoption Timing Analysis

Computes simple metrics showing that security risks are concentrated in Wave 3
(2023+), supporting the thesis argument that security threats are recent
additions being absorbed into standard RSA frameworks.

Outputs:
    1. % of security risk mentions by wave
    2. % of entities first adopting security in Wave 3
    3. Comparison across risk categories (nature/technical/antagonistic)

Usage:
    python scripts/03_bow_analysis/security_adoption_timing.py

Author: Swedish Risk Analysis Text-as-Data Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dictionaries.risk_categories import RISK_TO_CATEGORY

# =============================================================================
# CONFIGURATION
# =============================================================================

TERM_MATRIX_PATH = PROJECT_ROOT / "results/01_bow_analysis/term_matrices/term_document_matrix.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/01_bow_analysis/security_adoption_timing"

WAVE_LABELS = {0: 'Pre-2015', 1: '2015-2018', 2: '2019-2022', 3: '2023+'}

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_term_matrix() -> pd.DataFrame:
    """Load term-document matrix with metadata."""
    df = pd.read_csv(TERM_MATRIX_PATH)
    print(f"Loaded {len(df)} documents")
    print(f"Actors: {df['actor'].value_counts().to_dict()}")
    print(f"Waves: {df['wave'].value_counts().sort_index().to_dict()}")
    return df


def get_risk_columns(df: pd.DataFrame) -> list:
    """Get columns that are risk term counts."""
    metadata = ['file', 'actor', 'entity', 'year', 'wave', 'total_risk_mentions']
    return [c for c in df.columns if c not in metadata]


def categorize_risks(risk_cols: list) -> dict:
    """Map risk columns to their category (nature/technical/antagonistic/other)."""
    col_to_category = {}
    for col in risk_cols:
        risk_name = col.replace('risk_', '')
        category = RISK_TO_CATEGORY.get(risk_name, 'other')
        col_to_category[col] = category
    return col_to_category


def compute_mentions_by_wave(df: pd.DataFrame, risk_cols: list, col_to_category: dict) -> pd.DataFrame:
    """Compute total mentions per category per wave."""
    results = []

    for category in ['nature', 'technical', 'antagonistic', 'other']:
        cat_cols = [c for c, cat in col_to_category.items() if cat == category]

        for wave in sorted(df['wave'].unique()):
            wave_df = df[df['wave'] == wave]
            total_mentions = wave_df[cat_cols].sum().sum()
            results.append({
                'category': category,
                'wave': wave,
                'wave_label': WAVE_LABELS.get(wave, str(wave)),
                'total_mentions': total_mentions,
                'n_docs': len(wave_df)
            })

    return pd.DataFrame(results)


def compute_mention_percentages(mentions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute % of each category's mentions occurring in each wave."""
    pct_df = mentions_df.copy()
    cat_totals = pct_df.groupby('category')['total_mentions'].transform('sum')
    pct_df['pct_of_category'] = (pct_df['total_mentions'] / cat_totals * 100).round(1)
    return pct_df


def compute_first_adoption_wave(df: pd.DataFrame, risk_cols: list, col_to_category: dict) -> pd.DataFrame:
    """For each entity and category, find the first wave where they mention any risk in that category."""
    results = []

    for category in ['nature', 'technical', 'antagonistic']:
        cat_cols = [c for c, cat in col_to_category.items() if cat == category]

        for entity in df['entity'].unique():
            entity_df = df[df['entity'] == entity].sort_values('wave')

            first_wave = None
            for _, row in entity_df.iterrows():
                if row[cat_cols].sum() > 0:
                    first_wave = row['wave']
                    break

            if first_wave is not None:
                results.append({
                    'entity': entity,
                    'actor': entity_df['actor'].iloc[0],
                    'category': category,
                    'first_adoption_wave': first_wave
                })

    return pd.DataFrame(results)


def summarize_first_adoption(adoption_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize: % of entities first adopting in each wave, by category."""
    results = []

    for category in ['nature', 'technical', 'antagonistic']:
        cat_df = adoption_df[adoption_df['category'] == category]
        total_entities = len(cat_df)

        for wave in sorted(adoption_df['first_adoption_wave'].unique()):
            n_adopted = len(cat_df[cat_df['first_adoption_wave'] == wave])
            pct = (n_adopted / total_entities * 100) if total_entities > 0 else 0

            results.append({
                'category': category,
                'wave': wave,
                'wave_label': WAVE_LABELS.get(wave, str(wave)),
                'n_entities_first_adopt': n_adopted,
                'pct_entities_first_adopt': round(pct, 1)
            })

    return pd.DataFrame(results)


def print_summary(mentions_pct: pd.DataFrame, adoption_summary: pd.DataFrame):
    """Print thesis-ready summary statistics."""
    print("\n" + "="*70)
    print("SECURITY ADOPTION TIMING ANALYSIS - SUMMARY")
    print("="*70)

    # Key metric 1: % of security mentions in Wave 3
    def get_pct(cat, wave):
        row = mentions_pct[(mentions_pct['category'] == cat) & (mentions_pct['wave'] == wave)]
        return row['pct_of_category'].values[0] if len(row) > 0 else 0

    security_wave3 = get_pct('antagonistic', 3)
    nature_wave3 = get_pct('nature', 3)
    technical_wave3 = get_pct('technical', 3)

    print(f"\n📊 MENTION CONCENTRATION IN WAVE 3 (2023+):")
    print(f"   Security (antagonistic): {security_wave3}% of all mentions")
    print(f"   Nature:                  {nature_wave3}% of all mentions")
    print(f"   Technical:               {technical_wave3}% of all mentions")

    # Key metric 2: % of entities first adopting in Wave 3
    def get_adopt_pct(cat, wave):
        row = adoption_summary[(adoption_summary['category'] == cat) & (adoption_summary['wave'] == wave)]
        return row['pct_entities_first_adopt'].values[0] if len(row) > 0 else 0

    security_adopt_w3 = get_adopt_pct('antagonistic', 3)
    nature_adopt_w3 = get_adopt_pct('nature', 3)
    technical_adopt_w3 = get_adopt_pct('technical', 3)

    print(f"\n📊 FIRST ADOPTION IN WAVE 3 (2023+):")
    print(f"   Security (antagonistic): {security_adopt_w3}% of entities first adopt in Wave 3")
    print(f"   Nature:                  {nature_adopt_w3}% of entities first adopt in Wave 3")
    print(f"   Technical:               {technical_adopt_w3}% of entities first adopt in Wave 3")

    # Full breakdown tables
    print("\n" + "-"*70)
    print("DETAILED BREAKDOWN: Mentions by Wave (%)")
    print("-"*70)
    pivot = mentions_pct.pivot(index='category', columns='wave_label', values='pct_of_category')
    col_order = ['Pre-2015', '2015-2018', '2019-2022', '2023+']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    print(pivot.to_string())

    print("\n" + "-"*70)
    print("DETAILED BREAKDOWN: First Adoption Wave (%)")
    print("-"*70)
    pivot2 = adoption_summary.pivot(index='category', columns='wave_label', values='pct_entities_first_adopt')
    pivot2 = pivot2[[c for c in col_order if c in pivot2.columns]]
    print(pivot2.to_string())


def main():
    """Run the analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading term-document matrix...")
    df = load_term_matrix()

    risk_cols = get_risk_columns(df)
    col_to_category = categorize_risks(risk_cols)

    print(f"\nRisk terms by category:")
    for cat in ['nature', 'technical', 'antagonistic', 'other']:
        n = len([c for c, ct in col_to_category.items() if ct == cat])
        print(f"  {cat}: {n} terms")

    print("\nComputing mentions by wave...")
    mentions_df = compute_mentions_by_wave(df, risk_cols, col_to_category)
    mentions_pct = compute_mention_percentages(mentions_df)

    print("Computing first adoption wave per entity...")
    adoption_df = compute_first_adoption_wave(df, risk_cols, col_to_category)
    adoption_summary = summarize_first_adoption(adoption_df)

    print_summary(mentions_pct, adoption_summary)

    mentions_pct.to_csv(OUTPUT_DIR / "mentions_by_wave.csv", index=False)
    adoption_summary.to_csv(OUTPUT_DIR / "first_adoption_by_wave.csv", index=False)
    adoption_df.to_csv(OUTPUT_DIR / "entity_first_adoption.csv", index=False)

    print(f"\n✓ Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
