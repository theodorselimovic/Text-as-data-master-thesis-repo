#!/usr/bin/env python3
"""
Merge PDF Texts from All Actors

Combines extracted PDF texts from different institutional actors:
- Kommun (municipal risk analyses)
- Länsstyrelse (county administrative board analyses)
- MCF (MSB national risk assessments)

Extracts year from filenames and adds actor labels for comparative analysis.

Usage:
    python merge_all_actors.py \
        --kommun ./data/kommun_extraction/pdf_texts.parquet \
        --lansstyrelse ./data/lansstyrelse_extraction/pdf_texts.csv \
        --mcf ./data/mcf_extraction/pdf_texts.csv \
        --output ./data/merged/pdf_texts_all_actors.parquet

Output:
    Parquet file with columns: file, text, actor, year
"""

import argparse
import re
import pandas as pd
from pathlib import Path


def extract_year(filename: str) -> int:
    """
    Extract year from filename.

    Handles patterns like:
    - "RSA Skellefteå 2015.pdf" -> 2015
    - "RSA Kalmar Län 2022.pdf" -> 2022
    - "NRSB MCF 2021.pdf" -> 2021
    - "RSA Hylte 2015 Maskad.pdf" -> 2015

    Returns None if no year found.
    """
    # Look for 4-digit year (reasonable range: 2000-2030)
    match = re.search(r'\b(20[0-2][0-9])\b', filename)
    if match:
        return int(match.group(1))
    return None


def load_data(path: Path, actor: str) -> pd.DataFrame:
    """Load data from parquet or CSV and add actor column."""
    if not path.exists():
        print(f"Warning: {path} does not exist, skipping")
        return pd.DataFrame()

    # Load based on file extension
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Ensure required columns exist
    if 'file' not in df.columns or 'text' not in df.columns:
        raise ValueError(f"Data from {path} missing required columns 'file' and/or 'text'")

    # Add/overwrite actor column
    df['actor'] = actor

    # Extract year from filenames
    df['year'] = df['file'].apply(extract_year)

    # Report statistics
    years_found = df['year'].notna().sum()
    years_missing = df['year'].isna().sum()
    print(f"  Loaded {len(df)} documents from {path}")
    print(f"  Actor: {actor}")
    print(f"  Years extracted: {years_found}, missing: {years_missing}")
    if years_found > 0:
        year_range = f"{int(df['year'].min())}-{int(df['year'].max())}"
        print(f"  Year range: {year_range}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Merge PDF texts from all institutional actors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python merge_all_actors.py \\
        --kommun ./data/kommun_extraction/pdf_texts.parquet \\
        --lansstyrelse ./data/lansstyrelse_extraction/pdf_texts.csv \\
        --mcf ./data/mcf_extraction/pdf_texts.csv \\
        --output ./data/merged/pdf_texts_all_actors.parquet
        """
    )

    parser.add_argument(
        '--kommun',
        type=Path,
        required=True,
        help='Path to kommun (municipal) PDF texts'
    )

    parser.add_argument(
        '--lansstyrelse',
        type=Path,
        required=True,
        help='Path to länsstyrelse (county) PDF texts'
    )

    parser.add_argument(
        '--mcf',
        type=Path,
        required=True,
        help='Path to MCF (national) PDF texts'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for merged parquet file'
    )

    args = parser.parse_args()

    print("="*60)
    print("MERGE ALL ACTORS")
    print("="*60)

    # Load each source
    print("\nLoading kommun data...")
    df_kommun = load_data(args.kommun, 'kommun')

    print("\nLoading länsstyrelse data...")
    df_lan = load_data(args.lansstyrelse, 'länsstyrelse')

    print("\nLoading MCF data...")
    df_mcf = load_data(args.mcf, 'MCF')

    # Merge all
    print("\n" + "="*60)
    print("MERGING...")
    print("="*60)

    dfs = [df for df in [df_kommun, df_lan, df_mcf] if len(df) > 0]
    if not dfs:
        print("Error: No data to merge!")
        return 1

    df_merged = pd.concat(dfs, ignore_index=True)

    # Ensure consistent column order
    columns = ['file', 'text', 'actor', 'year']
    df_merged = df_merged[columns]

    print(f"\nTotal documents: {len(df_merged)}")
    print("\nBy actor:")
    print(df_merged['actor'].value_counts().to_string())
    print("\nYear coverage:")
    print(df_merged.groupby('actor')['year'].agg(['min', 'max', 'count']).to_string())

    # Save
    print("\n" + "="*60)
    print("SAVING...")
    print("="*60)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(args.output, index=False)
    print(f"\n✓ Saved merged data to: {args.output}")
    print(f"  Columns: {list(df_merged.columns)}")
    print(f"  Total documents: {len(df_merged)}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
