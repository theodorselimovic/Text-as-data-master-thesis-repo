#!/usr/bin/env python3
"""
Term-Document Matrix Builder

Builds term-level and category-level document matrices from the merged
RSA corpus. Extracts entity names (municipality/county/MCF) from filenames
and counts each individual risk term per document.

Outputs:
    - term_document_matrix.csv: one column per risk term (~100 terms)
    - category_document_matrix.csv: one column per risk category (8 categories)
    - term_metadata.csv: term -> category lookup table

Usage:
    python term_document_matrix.py \\
        --texts path/to/pdf_texts_all_actors.parquet \\
        --output ./results/term_document_matrix/

    python term_document_matrix.py \\
        --texts path/to/pdf_texts_all_actors.parquet \\
        --output ./results/term_document_matrix/ \\
        --verbose

Requirements:
    pip install pandas pyarrow
"""

import re
import argparse
import logging
from pathlib import Path

import pandas as pd

# Import the risk dictionaries from the analysis script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from risk_context_analysis import get_risk_dictionary, RISK_DICTIONARY_ORIGINAL

# =============================================================================
# CONFIGURATION
# =============================================================================

# RSA filename parsing pattern (from preprocessing.py)
RSA_FILENAME_PATTERN = re.compile(
    r"^RSA\s+(?P<entity>.+?)\s+(?P<year>(?:19|20)\d{2})"
    r"(?:\s+(?P<maskad>[Mm]askad|[Mm]askerad))?\s*\.pdf$",
    re.IGNORECASE,
)

# Fallback pattern for non-RSA filenames (e.g., NRSB MCF 2021.pdf)
NON_RSA_PATTERN = re.compile(
    r"^(?P<prefix>\w+)\s+(?P<entity>.+?)\s+(?P<year>(?:19|20)\d{2})"
    r"(?:\s+(?P<maskad>[Mm]askad))?\s*\.pdf$",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

def extract_entity(filename: str) -> str:
    """
    Extract entity name from RSA filename.

    Handles:
        'RSA Skellefteå 2015.pdf'       -> 'Skellefteå'
        'RSA Kalmar Län 2022.pdf'        -> 'Kalmar Län'
        'NRSB MCF 2021.pdf'              -> 'MCF'
        'RSA Ale 2015 Maskad.pdf'        -> 'Ale'

    Parameters
    ----------
    filename : str
        The PDF filename.

    Returns
    -------
    str
        The entity name, or 'unknown' if parsing fails.
    """
    # Try RSA pattern first
    match = RSA_FILENAME_PATTERN.match(filename)
    if match:
        return match.group('entity').strip()

    # Try non-RSA pattern (e.g., NRSB MCF...)
    match = NON_RSA_PATTERN.match(filename)
    if match:
        return match.group('entity').strip()

    logger.warning(f"Could not parse entity from filename: {filename}")
    return 'unknown'


def map_year_to_wave(year: int) -> int:
    """
    Map publication year to wave number.

    Wave mapping:
        Wave 0: pre-2015
        Wave 1: 2015-2018
        Wave 2: 2019-2022
        Wave 3: >= 2023

    Parameters
    ----------
    year : int
        Publication year

    Returns
    -------
    int
        Wave number (0, 1, 2, 3)
    """
    if year < 2015:
        return 0
    elif 2015 <= year <= 2018:
        return 1
    elif 2019 <= year <= 2022:
        return 2
    else:  # year >= 2023
        return 3


# =============================================================================
# TERM-LEVEL COUNTING
# =============================================================================

def count_terms_per_document(text: str, risk_dictionary: dict) -> dict:
    """
    Count each individual risk term in a document.

    Unlike count_risk_terms() in risk_context_analysis.py which returns
    category-level sums, this returns a flat dict with one entry per term.

    Parameters
    ----------
    text : str
        The document text.
    risk_dictionary : dict
        The RISK_DICTIONARY mapping category -> list of terms.

    Returns
    -------
    dict
        {term: count} for every term in the dictionary.
    """
    text_lower = text.lower()
    term_counts = {}

    for category, terms in risk_dictionary.items():
        for term in terms:
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            count = len(re.findall(pattern, text_lower))
            # Use the original term as column name (not lowered)
            term_counts[term] = count

    return term_counts


def aggregate_to_categories(
    term_counts: dict, risk_dictionary: dict
) -> dict:
    """
    Aggregate term-level counts to category-level.

    Parameters
    ----------
    term_counts : dict
        {term: count} from count_terms_per_document().
    risk_dictionary : dict
        The RISK_DICTIONARY.

    Returns
    -------
    dict
        {category: sum_of_term_counts}.
    """
    category_counts = {}
    for category, terms in risk_dictionary.items():
        category_counts[category] = sum(
            term_counts.get(term, 0) for term in terms
        )
    return category_counts


# =============================================================================
# TERM METADATA
# =============================================================================

def build_term_metadata(risk_dictionary: dict) -> pd.DataFrame:
    """
    Build a term -> category lookup table.

    Notes duplicate terms that appear in multiple categories.

    Parameters
    ----------
    risk_dictionary : dict
        The RISK_DICTIONARY.

    Returns
    -------
    pd.DataFrame
        Columns: term, category. May have multiple rows per term if
        the term appears in multiple categories.
    """
    rows = []
    for category, terms in risk_dictionary.items():
        for term in terms:
            rows.append({'term': term, 'category': category})
    return pd.DataFrame(rows)


# =============================================================================
# MATRIX BUILDING
# =============================================================================

def build_matrices(
    texts_df: pd.DataFrame,
    risk_dictionary: dict,
    text_column: str = 'text',
    verbose: bool = False,
) -> tuple:
    """
    Build term-level and category-level document matrices.

    Parameters
    ----------
    texts_df : pd.DataFrame
        The corpus with columns: file, text, actor, year.
    risk_dictionary : dict
        The RISK_DICTIONARY.
    text_column : str
        Column containing document text.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (term_matrix, category_matrix) with metadata columns.
    """
    # Collect all unique terms (preserving original casing)
    all_terms = []
    seen = set()
    for category, terms in risk_dictionary.items():
        for term in terms:
            if term not in seen:
                all_terms.append(term)
                seen.add(term)

    term_rows = []
    category_rows = []

    n_docs = len(texts_df)
    for idx, row in texts_df.iterrows():
        if verbose and idx % 50 == 0:
            print(f"  Processing document {idx}/{n_docs}...")

        text = str(row.get(text_column, ''))
        filename = str(row.get('file', ''))
        actor = str(row.get('actor', 'unknown'))
        year = row.get('year', None)
        entity = extract_entity(filename)

        # Map year to wave
        wave = map_year_to_wave(year) if year is not None else None

        # Count terms
        term_counts = count_terms_per_document(text, risk_dictionary)
        category_counts = aggregate_to_categories(term_counts, risk_dictionary)

        # Build metadata
        metadata = {
            'file': filename,
            'actor': actor,
            'entity': entity,
            'year': year,
            'wave': wave,
        }

        # Term-level row
        term_row = {**metadata}
        for term in all_terms:
            term_row[term] = term_counts.get(term, 0)
        term_rows.append(term_row)

        # Category-level row
        cat_row = {**metadata}
        for category in risk_dictionary.keys():
            cat_row[f'risk_{category}'] = category_counts.get(category, 0)
        cat_row['total_risk_mentions'] = sum(category_counts.values())
        category_rows.append(cat_row)

    term_matrix = pd.DataFrame(term_rows)
    category_matrix = pd.DataFrame(category_rows)

    return term_matrix, category_matrix


# =============================================================================
# OUTPUT
# =============================================================================

def save_outputs(
    term_matrix: pd.DataFrame,
    category_matrix: pd.DataFrame,
    term_metadata: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save all outputs to the output directory.

    Parameters
    ----------
    term_matrix : pd.DataFrame
        Term-level document matrix.
    category_matrix : pd.DataFrame
        Category-level document matrix.
    term_metadata : pd.DataFrame
        Term -> category lookup.
    output_dir : Path
        Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    term_path = output_dir / 'term_document_matrix.csv'
    term_matrix.to_csv(term_path, index=False, encoding='utf-8')
    print(f"  Saved: {term_path} ({term_matrix.shape[0]} docs × {term_matrix.shape[1]} cols)")

    cat_path = output_dir / 'category_document_matrix.csv'
    category_matrix.to_csv(cat_path, index=False, encoding='utf-8')
    print(f"  Saved: {cat_path} ({category_matrix.shape[0]} docs × {category_matrix.shape[1]} cols)")

    meta_path = output_dir / 'term_metadata.csv'
    term_metadata.to_csv(meta_path, index=False, encoding='utf-8')
    print(f"  Saved: {meta_path} ({len(term_metadata)} term-category mappings)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build term-level and category-level document matrices'
    )

    parser.add_argument(
        '--texts',
        type=Path,
        required=True,
        help='Path to merged parquet file with texts'
    )

    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Column name containing text (default: text)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./results/term_document_matrix'),
        help='Output directory'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TERM-DOCUMENT MATRIX BUILDER")
    print("=" * 60)

    # Load data
    print(f"\nLoading texts from: {args.texts}")
    texts_df = pd.read_parquet(args.texts)
    print(f"  Loaded {len(texts_df)} documents")
    print(f"  Columns: {list(texts_df.columns)}")

    if 'actor' in texts_df.columns:
        print(f"  Actors: {texts_df['actor'].value_counts().to_dict()}")
    if 'year' in texts_df.columns:
        print(f"  Years: {sorted(texts_df['year'].dropna().unique())}")

    # Build original (non-lemmatized) matrices
    print(f"\n{'='*60}")
    print("STAGE 1: Building ORIGINAL (non-lemmatized) matrices")
    print(f"{'='*60}")

    print(f"\nOriginal risk dictionary: {len(RISK_DICTIONARY_ORIGINAL)} categories")
    term_metadata_original = build_term_metadata(RISK_DICTIONARY_ORIGINAL)
    n_unique_terms_orig = term_metadata_original['term'].nunique()
    n_total_mappings_orig = len(term_metadata_original)
    n_duplicates_orig = n_total_mappings_orig - n_unique_terms_orig
    print(f"  {n_unique_terms_orig} unique terms, {n_duplicates_orig} duplicates across categories")

    if n_duplicates_orig > 0:
        dup_terms = term_metadata_original[term_metadata_original.duplicated(subset='term', keep=False)]
        for term in dup_terms['term'].unique():
            cats = dup_terms[dup_terms['term'] == term]['category'].tolist()
            print(f"    Duplicate: '{term}' in {cats}")

    print(f"\nBuilding original matrices...")
    term_matrix_original, category_matrix_original = build_matrices(
        texts_df, RISK_DICTIONARY_ORIGINAL,
        text_column=args.text_column,
        verbose=args.verbose,
    )

    # Build lemmatized matrices
    print(f"\n{'='*60}")
    print("STAGE 2: Building LEMMATIZED matrices")
    print(f"{'='*60}")

    # Get lemmatized dictionary (will create it if needed)
    risk_dict_lemmatized = get_risk_dictionary(lemmatize=True, output_dir=args.output)

    print(f"\nLemmatized risk dictionary: {len(risk_dict_lemmatized)} categories")
    term_metadata = build_term_metadata(risk_dict_lemmatized)
    n_unique_terms = term_metadata['term'].nunique()
    n_total_mappings = len(term_metadata)
    n_duplicates = n_total_mappings - n_unique_terms
    print(f"  {n_unique_terms} unique terms, {n_duplicates} duplicates across categories")
    print(f"  Reduction from original: {n_unique_terms_orig - n_unique_terms} terms ({(1 - n_unique_terms / n_unique_terms_orig) * 100:.1f}%)")

    if n_duplicates > 0:
        dup_terms = term_metadata[term_metadata.duplicated(subset='term', keep=False)]
        for term in dup_terms['term'].unique():
            cats = dup_terms[dup_terms['term'] == term]['category'].tolist()
            print(f"    Duplicate: '{term}' in {cats}")

    print(f"\nBuilding lemmatized matrices...")
    term_matrix, category_matrix = build_matrices(
        texts_df, risk_dict_lemmatized,
        text_column=args.text_column,
        verbose=args.verbose,
    )

    # Summary statistics
    metadata_cols = ['file', 'actor', 'entity', 'year', 'wave']
    term_cols = [c for c in term_matrix.columns if c not in metadata_cols]
    print(f"\nTerm matrix: {term_matrix.shape[0]} documents × {len(term_cols)} terms")
    print(f"  Non-zero entries: {(term_matrix[term_cols] > 0).sum().sum()}")
    print(f"  Sparsity: {1 - (term_matrix[term_cols] > 0).sum().sum() / (len(term_cols) * len(term_matrix)):.1%}")

    # Wave distribution
    wave_ranges = {
        0: 'pre-2015',
        1: '2015-2018',
        2: '2019-2022',
        3: '≥ 2023',
    }
    print(f"\nWave distribution:")
    wave_stats = term_matrix.groupby('wave').agg({'year': ['min', 'max', 'count']})
    for wave in sorted(term_matrix['wave'].unique()):
        wave_range = wave_ranges.get(wave, 'unknown')
        count = len(term_matrix[term_matrix['wave'] == wave])
        year_min = term_matrix[term_matrix['wave'] == wave]['year'].min()
        year_max = term_matrix[term_matrix['wave'] == wave]['year'].max()
        print(f"  Wave {wave} ({wave_range}): {count} documents (years {year_min}-{year_max})")

    # Entity extraction summary
    entity_counts = term_matrix.groupby('actor')['entity'].nunique()
    print(f"\nEntities extracted:")
    for actor, count in entity_counts.items():
        print(f"  {actor}: {count} unique entities")

    unknowns = term_matrix[term_matrix['entity'] == 'unknown']
    if len(unknowns) > 0:
        print(f"\n  WARNING: {len(unknowns)} documents with unknown entity:")
        for _, row in unknowns.iterrows():
            print(f"    {row['file']}")

    # Save
    print(f"\nSaving outputs to: {args.output}")

    # Save original (non-lemmatized) matrices
    print(f"\n  Saving ORIGINAL matrices...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    term_path_original = output_dir / 'term_document_matrix_original.csv'
    term_matrix_original.to_csv(term_path_original, index=False, encoding='utf-8')
    print(f"    Saved: {term_path_original} ({term_matrix_original.shape[0]} docs × {term_matrix_original.shape[1]} cols)")

    cat_path_original = output_dir / 'category_document_matrix_original.csv'
    category_matrix_original.to_csv(cat_path_original, index=False, encoding='utf-8')
    print(f"    Saved: {cat_path_original} ({category_matrix_original.shape[0]} docs × {category_matrix_original.shape[1]} cols)")

    meta_path_original = output_dir / 'term_metadata_original.csv'
    term_metadata_original.to_csv(meta_path_original, index=False, encoding='utf-8')
    print(f"    Saved: {meta_path_original} ({len(term_metadata_original)} term-category mappings)")

    # Save lemmatized matrices (default files)
    print(f"\n  Saving LEMMATIZED matrices (default)...")
    save_outputs(term_matrix, category_matrix, term_metadata, args.output)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"\nOutput files created:")
    print(f"  Original matrices: *_original.csv")
    print(f"  Lemmatized matrices: *.csv (default)")
    print(f"  Lemma mapping: lemma_mapping.json\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
