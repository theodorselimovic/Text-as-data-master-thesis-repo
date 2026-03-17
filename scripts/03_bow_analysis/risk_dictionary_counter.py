#!/usr/bin/env python3
"""
Risk Dictionary Counter

Counts occurrences of risk terms from a predefined dictionary in RSA documents.
Builds term-level and category-level document matrices.

Supports three input modes:
1. Raw text: Searches for terms using regex in raw text
2. Pre-lemmatized: Counts lemmas in token lists (from preprocessing_bow.py with Stanza)
3. Pre-stemmed: Counts stemmed tokens including n-grams (from preprocessing_bow.py with stemming)

When using pre-processed input (--use-lemmas or --use-stems), the dictionary terms
are also processed identically for consistent matching.

Input formats:
    - Document-level: parquet with 'file', 'text' (or 'tokens'), 'actor' columns
    - Sentence-level: parquet from preprocessing_bow.py with 'doc_id', 'tokens' columns
      (will be aggregated to document level)

Outputs:
    - term_document_matrix.csv: one column per risk term (~100 terms)
    - category_document_matrix.csv: one column per risk category (8 categories)
    - term_metadata.csv: term -> category lookup table

Usage:
    # From raw text (document-level)
    python risk_dictionary_counter.py \\
        --input data/raw/pdf_texts.parquet \\
        --output ./results/risk_matrices/

    # From pre-stemmed corpus (recommended for speed + n-gram matching)
    python risk_dictionary_counter.py \\
        --input data/processed/bow_corpus_stemmed.parquet \\
        --output ./results/risk_matrices/ \\
        --use-stems

    # From pre-lemmatized corpus (legacy Stanza mode)
    python risk_dictionary_counter.py \\
        --input data/processed/bow_corpus.parquet \\
        --output ./results/risk_matrices/ \\
        --use-lemmas

    # With verbose output
    python risk_dictionary_counter.py \\
        --input data/processed/bow_corpus_stemmed.parquet \\
        --output ./results/risk_matrices/ \\
        --use-stems --verbose

Requirements:
    pip install pandas pyarrow nltk
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

# Import stemmer for --use-stems mode
from nltk.stem.snowball import SnowballStemmer

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
# DICTIONARY STEMMING
# =============================================================================

def stem_risk_dictionary(risk_dict: dict) -> dict:
    """
    Stem all terms in risk dictionary, including n-gram joining for multi-word terms.

    Multi-word terms like "organiserad brottslighet" become "organiser_brottslig"
    to match the n-gram format from preprocessing_bow.py.

    Parameters
    ----------
    risk_dict : dict
        The RISK_DICTIONARY mapping category -> list of terms.

    Returns
    -------
    dict
        Stemmed dictionary with same structure. Multi-word terms joined with underscore.
    """
    stemmer = SnowballStemmer('swedish')
    stemmed = {}

    for category, terms in risk_dict.items():
        stemmed_terms = set()
        for term in terms:
            words = term.lower().split()
            stemmed_words = [stemmer.stem(w) for w in words]
            if len(stemmed_words) == 1:
                stemmed_terms.add(stemmed_words[0])
            else:
                # Multi-word: add as joined n-gram
                stemmed_terms.add('_'.join(stemmed_words))
        stemmed[category] = list(stemmed_terms)

    return stemmed


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
    year = int(year)  # Handle string years
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
    Count each individual risk term in a document (raw text mode).

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


def count_terms_from_tokens(tokens: list, risk_dictionary: dict) -> dict:
    """
    Count each individual risk term from a list of lemmatized tokens.

    For use with pre-lemmatized corpus from preprocessing_bow.py.
    The dictionary terms should also be lemmatized for consistent matching.

    Parameters
    ----------
    tokens : list
        List of lemmatized tokens (lowercase).
    risk_dictionary : dict
        The RISK_DICTIONARY mapping category -> list of (lemmatized) terms.

    Returns
    -------
    dict
        {term: count} for every term in the dictionary.
    """
    from collections import Counter

    # Handle empty tokens or non-iterable
    if tokens is None or (hasattr(tokens, '__len__') and len(tokens) == 0):
        # Return zeros for all terms
        term_counts = {}
        for category, terms in risk_dictionary.items():
            for term in terms:
                term_counts[term] = 0
        return term_counts

    # Convert numpy arrays to list if needed
    if not isinstance(tokens, list):
        tokens = list(tokens)

    # Count all tokens
    token_counts = Counter(tokens)

    # Extract counts for dictionary terms
    term_counts = {}
    for category, terms in risk_dictionary.items():
        for term in terms:
            # Terms in dictionary should already be lemmatized and lowercase
            term_counts[term] = token_counts.get(term.lower(), 0)

    return term_counts


def aggregate_sentences_to_documents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentence-level data to document-level.

    Combines tokens from all sentences in a document and extracts
    metadata from the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level DataFrame with 'doc_id' and 'tokens' columns.

    Returns
    -------
    pd.DataFrame
        Document-level DataFrame with combined tokens.
    """
    logger.info("Aggregating sentences to documents...")

    # Group by document
    doc_groups = df.groupby('doc_id')

    rows = []
    for doc_id, group in doc_groups:
        # Combine all tokens from sentences
        all_tokens = []
        for tokens in group['tokens']:
            # Handle both lists and numpy arrays
            if hasattr(tokens, '__iter__') and not isinstance(tokens, str):
                all_tokens.extend(list(tokens))

        # Get metadata from first row
        first_row = group.iloc[0]

        row = {
            'file': doc_id,
            'tokens': all_tokens,
            'actor': first_row.get('actor_type', 'unknown'),
            'year': first_row.get('year', None),
            'municipality': first_row.get('municipality', None),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    logger.info(f"  Aggregated {len(df)} sentences -> {len(result)} documents")

    return result


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
    use_lemmas: bool = False,
    verbose: bool = False,
) -> tuple:
    """
    Build term-level and category-level document matrices.

    Parameters
    ----------
    texts_df : pd.DataFrame
        The corpus with columns: file, text/tokens, actor, year.
    risk_dictionary : dict
        The RISK_DICTIONARY.
    text_column : str
        Column containing document text (ignored if use_lemmas=True).
    use_lemmas : bool
        If True, count from 'tokens' column instead of raw text.
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

        filename = str(row.get('file', ''))
        actor = str(row.get('actor', 'unknown'))
        year = row.get('year', None)
        entity = extract_entity(filename)

        # Map year to wave
        wave = map_year_to_wave(year) if year is not None else None

        # Count terms - either from tokens or raw text
        if use_lemmas:
            tokens = row.get('tokens', [])
            term_counts = count_terms_from_tokens(tokens, risk_dictionary)
        else:
            text = str(row.get(text_column, ''))
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
        description='Count risk dictionary terms and build document matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Path to input parquet file (raw texts or preprocessed bow_corpus)'
    )

    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Column name containing text (default: text, ignored with --use-lemmas)'
    )

    parser.add_argument(
        '--use-lemmas',
        action='store_true',
        help='Use pre-lemmatized tokens from preprocessing_bow.py with Stanza (reads "tokens" column)'
    )

    parser.add_argument(
        '--use-stems',
        action='store_true',
        help='Use pre-stemmed tokens with n-grams from preprocessing_bow.py (reads "tokens" column, recommended)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./results/risk_matrices'),
        help='Output directory'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RISK DICTIONARY COUNTER")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Detect input type and prepare data
    is_sentence_level = 'doc_id' in df.columns and 'sentence_id' in df.columns

    # Validate mutually exclusive options
    if args.use_lemmas and args.use_stems:
        print("ERROR: --use-lemmas and --use-stems are mutually exclusive")
        return 1

    use_tokens = args.use_lemmas or args.use_stems

    if use_tokens:
        mode_name = "Pre-stemmed tokens (with n-grams)" if args.use_stems else "Pre-lemmatized tokens"
        print(f"\nMode: {mode_name}")
        if 'tokens' not in df.columns:
            print("ERROR: --use-lemmas/--use-stems requires 'tokens' column (from preprocessing_bow.py)")
            return 1

        # Aggregate sentence-level to document-level if needed
        if is_sentence_level:
            df = aggregate_sentences_to_documents(df)
    else:
        print(f"\nMode: Raw text (regex matching)")
        if is_sentence_level:
            # For raw text mode with sentence-level input, concatenate sentences
            print("  Aggregating sentences to documents...")
            df = df.groupby('doc_id').agg({
                'sentence_text': ' '.join,
                'actor_type': 'first',
                'year': 'first',
                'municipality': 'first',
            }).reset_index()
            df = df.rename(columns={'doc_id': 'file', 'sentence_text': 'text', 'actor_type': 'actor'})
            print(f"  Aggregated to {len(df)} documents")

    # Show data summary
    actor_col = 'actor' if 'actor' in df.columns else 'actor_type'
    if actor_col in df.columns:
        print(f"  Actors: {df[actor_col].value_counts().to_dict()}")
    if 'year' in df.columns:
        years = df['year'].dropna().unique()
        if len(years) > 0:
            print(f"  Years: {sorted([int(y) for y in years if pd.notna(y)])}")

    # Get risk dictionary (lemmatized if using pre-lemmatized corpus)
    print(f"\n{'='*60}")
    print("BUILDING RISK TERM MATRICES")
    print(f"{'='*60}")

    if args.use_stems:
        # Use stemmed dictionary to match stemmed tokens with n-grams
        risk_dict = stem_risk_dictionary(RISK_DICTIONARY_ORIGINAL)
        dict_type = "stemmed"
        # Show some example transformations
        print("\n  Sample stemmed terms:")
        for cat, terms in list(risk_dict.items())[:2]:
            orig_terms = RISK_DICTIONARY_ORIGINAL[cat][:3]
            stem_terms = terms[:3]
            print(f"    {cat}: {orig_terms} -> {stem_terms}")
    elif args.use_lemmas:
        # Use lemmatized dictionary to match lemmatized tokens
        risk_dict = get_risk_dictionary(lemmatize=True, output_dir=args.output)
        dict_type = "lemmatized"
    else:
        # Use original dictionary for raw text matching
        risk_dict = RISK_DICTIONARY_ORIGINAL
        dict_type = "original"

    print(f"\nRisk dictionary ({dict_type}): {len(risk_dict)} categories")
    term_metadata = build_term_metadata(risk_dict)
    n_unique_terms = term_metadata['term'].nunique()
    n_total_mappings = len(term_metadata)
    n_duplicates = n_total_mappings - n_unique_terms
    print(f"  {n_unique_terms} unique terms, {n_duplicates} duplicates across categories")

    if n_duplicates > 0:
        dup_terms = term_metadata[term_metadata.duplicated(subset='term', keep=False)]
        for term in dup_terms['term'].unique():
            cats = dup_terms[dup_terms['term'] == term]['category'].tolist()
            print(f"    Duplicate: '{term}' in {cats}")

    print(f"\nBuilding matrices...")
    term_matrix, category_matrix = build_matrices(
        df, risk_dict,
        text_column=args.text_column,
        use_lemmas=use_tokens,  # True for both --use-lemmas and --use-stems
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

    # Save outputs
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print(f"{'='*60}")
    print(f"\nOutput directory: {args.output}")

    save_outputs(term_matrix, category_matrix, term_metadata, args.output)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"\nOutput files:")
    print(f"  term_document_matrix.csv - per-term counts")
    print(f"  category_document_matrix.csv - per-category counts")
    print(f"  term_metadata.csv - term to category mapping")
    print(f"\nThese can be used by:")
    print(f"  - risk_persistence_analysis.py (term_document_matrix.csv)")
    print(f"  - risk_clustering_analysis.py (category_document_matrix.csv)\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
