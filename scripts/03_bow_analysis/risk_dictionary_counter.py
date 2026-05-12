#!/usr/bin/env python3
"""
Risk Dictionary Counter

Counts occurrences of canonical risk terms in RSA documents using the
RISK_TERMS dictionary. Variants are collapsed to canonical names.

Input: Pre-stemmed corpus from preprocessing_bow.py (bow_corpus_stemmed.parquet)

Outputs:
    - term_document_matrix.csv: one column per canonical risk (~100 risks)
    - category_document_matrix.csv: risks aggregated by MSB category
    - term_metadata.csv: risk → category mapping

Usage:
    python risk_dictionary_counter.py \\
        --input data/processed/bow_corpus_stemmed.parquet \\
        --output results/01_bow_analysis/term_matrices/

    # Include pre-2015 documents
    python risk_dictionary_counter.py \\
        --input data/processed/bow_corpus_stemmed.parquet \\
        --output results/01_bow_analysis/term_matrices/ \\
        --min-year 0
"""

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.stem.snowball import SnowballStemmer

# Import dictionaries
sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionaries import RISK_TERMS, RISK_TO_CATEGORY, CATEGORY_NAMES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filename patterns
RSA_PATTERN = re.compile(
    r"^RSA\s+(?P<entity>.+?)\s+(?P<year>(?:19|20)\d{2})"
    r"(?:\s+(?P<maskad>[Mm]askad|[Mm]askerad))?\s*\.pdf$",
    re.IGNORECASE,
)
NRSB_PATTERN = re.compile(
    r"^(?P<prefix>\w+)\s+(?P<entity>.+?)\s+(?P<year>(?:19|20)\d{2})"
    r"(?:\s+(?P<maskad>[Mm]askad))?\s*\.pdf$",
    re.IGNORECASE,
)


# =============================================================================
# DICTIONARY STEMMING
# =============================================================================

def stem_dictionary() -> dict:
    """
    Stem all variants in RISK_TERMS dictionary.

    Returns
    -------
    dict
        {canonical_risk: set of stemmed variants}
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / '02_preprocessing'))
    from preprocessing_bow import SWEDISH_STOPWORDS

    stemmer = SnowballStemmer('swedish')
    stemmed_stopwords = {stemmer.stem(sw) for sw in SWEDISH_STOPWORDS}

    result = {}
    for canonical, variants in RISK_TERMS.items():
        stemmed_forms = set()
        for term in variants:
            words = term.lower().split()
            stemmed_words = [
                stemmer.stem(w) for w in words
                if w not in SWEDISH_STOPWORDS and stemmer.stem(w) not in stemmed_stopwords
            ]
            if stemmed_words:
                if len(stemmed_words) == 1:
                    stemmed_forms.add(stemmed_words[0])
                else:
                    stemmed_forms.add('_'.join(stemmed_words))
        if stemmed_forms:
            result[canonical] = stemmed_forms

    return result


# =============================================================================
# HELPERS
# =============================================================================

def extract_entity(filename: str) -> str:
    """Extract entity name from RSA filename."""
    for pattern in [RSA_PATTERN, NRSB_PATTERN]:
        match = pattern.match(filename)
        if match:
            return match.group('entity').strip()
    return 'unknown'


def map_year_to_wave(year: int) -> int:
    """Map year to wave: 0=pre-2015, 1=2015-18, 2=2019-22, 3=2023+."""
    year = int(year)
    if year < 2015:
        return 0
    elif year <= 2018:
        return 1
    elif year <= 2022:
        return 2
    return 3


def aggregate_sentences_to_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentence-level data to document-level."""
    logger.info("Aggregating sentences to documents...")

    rows = []
    for doc_id, group in df.groupby('doc_id'):
        all_tokens = []
        for tokens in group['tokens']:
            if hasattr(tokens, '__iter__') and not isinstance(tokens, str):
                all_tokens.extend(list(tokens))

        first = group.iloc[0]
        rows.append({
            'file': doc_id,
            'tokens': all_tokens,
            'actor': first.get('actor_type', 'unknown'),
            'year': first.get('year', None),
        })

    result = pd.DataFrame(rows)
    logger.info(f"  Aggregated to {len(result)} documents")
    return result


def aggregate_sentences_to_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentence-level data to paragraph-level for character counting.

    Returns DataFrame with one row per paragraph containing tokens and char count.
    """
    logger.info("Aggregating sentences to paragraphs...")

    rows = []
    for (doc_id, para_id), group in df.groupby(['doc_id', 'paragraph_id']):
        all_tokens = []
        for tokens in group['tokens']:
            if hasattr(tokens, '__iter__') and not isinstance(tokens, str):
                all_tokens.extend(list(tokens))

        para_text = ' '.join(group['sentence_text'].fillna(''))
        char_count = len(para_text)

        first = group.iloc[0]
        rows.append({
            'doc_id': doc_id,
            'paragraph_id': para_id,
            'tokens': all_tokens,
            'char_count': char_count,
            'actor': first.get('actor_type', 'unknown'),
            'year': first.get('year', None),
        })

    result = pd.DataFrame(rows)
    logger.info(f"  Aggregated to {len(result)} paragraphs")
    return result


# =============================================================================
# COUNTING
# =============================================================================

def count_risks_in_document(tokens: list, stemmed_dict: dict) -> dict:
    """
    Count canonical risks in a document's tokens.

    Parameters
    ----------
    tokens : list
        Stemmed tokens from the document
    stemmed_dict : dict
        {canonical_risk: set of stemmed variants}

    Returns
    -------
    dict
        {canonical_risk: count}
    """
    if not tokens:
        return {risk: 0 for risk in stemmed_dict}

    if not isinstance(tokens, list):
        tokens = list(tokens)

    token_counts = Counter(tokens)

    risk_counts = {}
    for canonical, variants in stemmed_dict.items():
        count = sum(token_counts.get(v, 0) for v in variants)
        risk_counts[canonical] = count

    return risk_counts


def count_risk_characters_in_document(
    paragraphs: pd.DataFrame,
    stemmed_dict: dict,
) -> dict:
    """
    Count total characters in paragraphs mentioning each risk.

    For each risk, finds all paragraphs mentioning it and sums their character
    counts. Paragraphs mentioning multiple variants of the same risk are counted
    once.

    Parameters
    ----------
    paragraphs : pd.DataFrame
        Paragraph-level data for a single document with 'tokens' and 'char_count'
    stemmed_dict : dict
        {canonical_risk: set of stemmed variants}

    Returns
    -------
    dict
        {canonical_risk: total_characters}
    """
    from collections import defaultdict

    risk_paragraphs = defaultdict(set)
    para_chars = {}

    for idx, row in paragraphs.iterrows():
        tokens = row.get('tokens', [])
        if not tokens:
            continue

        para_id = row['paragraph_id']
        para_chars[para_id] = row['char_count']

        token_set = set(tokens) if not isinstance(tokens, set) else tokens

        for canonical, variants in stemmed_dict.items():
            if token_set & variants:
                risk_paragraphs[canonical].add(para_id)

    char_counts = {}
    for canonical in stemmed_dict:
        paras = risk_paragraphs.get(canonical, set())
        char_counts[canonical] = sum(para_chars.get(p, 0) for p in paras)

    return char_counts


def build_character_matrix(
    paragraphs_df: pd.DataFrame,
    stemmed_dict: dict,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build character document matrix: characters devoted to each risk per document.

    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        Paragraph-level data with 'doc_id', 'tokens', 'char_count', 'actor', 'year'
    stemmed_dict : dict
        {canonical_risk: set of stemmed variants}

    Returns
    -------
    pd.DataFrame
        Document × risk matrix with character counts
    """
    logger.info("Building character matrix...")

    canonical_risks = sorted(stemmed_dict.keys())
    rows = []

    doc_ids = paragraphs_df['doc_id'].unique()
    for i, doc_id in enumerate(doc_ids):
        if verbose and i % 100 == 0:
            logger.info(f"  Processing document {i}/{len(doc_ids)}...")

        doc_paras = paragraphs_df[paragraphs_df['doc_id'] == doc_id]
        first = doc_paras.iloc[0]

        meta = {
            'file': doc_id,
            'actor': first.get('actor', 'unknown'),
            'entity': extract_entity(str(doc_id)),
            'year': first.get('year'),
            'wave': map_year_to_wave(first['year']) if pd.notna(first.get('year')) else None,
        }

        char_counts = count_risk_characters_in_document(doc_paras, stemmed_dict)
        rows.append({**meta, **char_counts})

    result = pd.DataFrame(rows)
    logger.info(f"  Built matrix: {len(result)} documents × {len(canonical_risks)} risks")
    return result


def build_matrices(df: pd.DataFrame, stemmed_dict: dict, verbose: bool = False):
    """
    Build term and category document matrices.

    Parameters
    ----------
    df : pd.DataFrame
        Document-level data with 'file', 'tokens', 'actor', 'year' columns
    stemmed_dict : dict
        {canonical_risk: set of stemmed variants}
    verbose : bool
        Print progress

    Returns
    -------
    tuple
        (term_matrix, category_matrix) DataFrames
    """
    canonical_risks = sorted(stemmed_dict.keys())
    categories = sorted(set(RISK_TO_CATEGORY.values()))

    term_rows = []
    cat_rows = []

    for idx, row in df.iterrows():
        if verbose and idx % 100 == 0:
            logger.info(f"  Processing document {idx}/{len(df)}...")

        # Metadata
        meta = {
            'file': row.get('file', ''),
            'actor': row.get('actor', 'unknown'),
            'entity': extract_entity(str(row.get('file', ''))),
            'year': row.get('year'),
            'wave': map_year_to_wave(row['year']) if pd.notna(row.get('year')) else None,
        }

        # Count risks
        risk_counts = count_risks_in_document(row.get('tokens', []), stemmed_dict)

        # Term row
        term_row = {**meta, **risk_counts}
        term_rows.append(term_row)

        # Category row
        cat_counts = {cat: 0 for cat in categories}
        for risk, count in risk_counts.items():
            cat = RISK_TO_CATEGORY.get(risk)
            if cat:
                cat_counts[cat] += count

        cat_row = {**meta}
        for cat in categories:
            cat_row[f'risk_{cat}'] = cat_counts[cat]
        cat_row['total_risk_mentions'] = sum(cat_counts.values())
        cat_rows.append(cat_row)

    return pd.DataFrame(term_rows), pd.DataFrame(cat_rows)


def build_term_metadata(stemmed_dict: dict) -> pd.DataFrame:
    """Build term → category metadata table."""
    rows = []
    for canonical in sorted(stemmed_dict.keys()):
        category = RISK_TO_CATEGORY.get(canonical, 'unknown')
        rows.append({
            'term': canonical,
            'category': category,
            'category_name': CATEGORY_NAMES.get(category, category),
        })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Count risk terms and build document matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input', '-i', type=Path, required=True,
        help='Path to bow_corpus_stemmed.parquet'
    )
    parser.add_argument(
        '--output', '-o', type=Path,
        default=Path('./results/01_bow_analysis/term_matrices'),
        help='Output directory'
    )
    parser.add_argument(
        '--min-year', type=int, default=2015,
        help='Minimum year to include (default: 2015). Use 0 for all.'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print progress'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RISK DICTIONARY COUNTER")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} rows")

    # Filter by year
    if args.min_year > 0 and 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        n_before = len(df)
        df = df[df['year'] >= args.min_year]
        n_filtered = n_before - len(df)
        if n_filtered > 0:
            print(f"  Filtered {n_filtered} rows before {args.min_year} ({len(df)} remaining)")

    # Aggregate sentences to documents (for mention matrix)
    # Also keep paragraph-level for character matrix
    paragraphs_df = None
    if 'doc_id' in df.columns and 'sentence_id' in df.columns:
        paragraphs_df = aggregate_sentences_to_paragraphs(df)
        df = aggregate_sentences_to_documents(df)

    # Show summary
    actor_col = 'actor' if 'actor' in df.columns else 'actor_type'
    if actor_col in df.columns:
        print(f"  Actors: {df[actor_col].value_counts().to_dict()}")

    # Stem dictionary
    print(f"\n{'=' * 60}")
    print("BUILDING MATRICES")
    print("=" * 60)

    print("\nStemming dictionary...")
    stemmed_dict = stem_dictionary()
    print(f"  {len(stemmed_dict)} canonical risks")

    # Sample
    for risk in list(stemmed_dict.keys())[:3]:
        variants = list(stemmed_dict[risk])[:3]
        print(f"    {risk}: {variants}")

    # Build matrices
    print("\nCounting risks...")
    term_matrix, cat_matrix = build_matrices(df, stemmed_dict, args.verbose)

    # Build character matrix (if paragraph data available)
    char_matrix = None
    if paragraphs_df is not None:
        print("\nCounting characters per risk...")
        char_matrix = build_character_matrix(paragraphs_df, stemmed_dict, args.verbose)

    # Stats
    risk_cols = [c for c in term_matrix.columns if c not in ['file', 'actor', 'entity', 'year', 'wave']]
    n_nonzero = (term_matrix[risk_cols] > 0).sum().sum()
    total_cells = len(term_matrix) * len(risk_cols)
    sparsity = 1 - (n_nonzero / total_cells) if total_cells > 0 else 0

    print(f"\nTerm matrix: {len(term_matrix)} documents × {len(risk_cols)} risks")
    print(f"  Non-zero: {n_nonzero}, Sparsity: {sparsity:.1%}")

    # Wave distribution
    if 'wave' in term_matrix.columns:
        print("\nWave distribution:")
        wave_labels = {1: '2015-2018', 2: '2019-2022', 3: '2023+'}
        for wave in sorted(term_matrix['wave'].dropna().unique()):
            wave = int(wave)
            if wave > 0:
                n = len(term_matrix[term_matrix['wave'] == wave])
                print(f"  Wave {wave} ({wave_labels.get(wave, '?')}): {n} documents")

    # Build metadata
    term_metadata = build_term_metadata(stemmed_dict)

    # Save
    print(f"\n{'=' * 60}")
    print("SAVING")
    print("=" * 60)

    args.output.mkdir(parents=True, exist_ok=True)

    term_path = args.output / 'term_document_matrix.csv'
    term_matrix.to_csv(term_path, index=False)
    print(f"  {term_path} ({len(term_matrix)} × {len(term_matrix.columns)})")

    cat_path = args.output / 'category_document_matrix.csv'
    cat_matrix.to_csv(cat_path, index=False)
    print(f"  {cat_path} ({len(cat_matrix)} × {len(cat_matrix.columns)})")

    meta_path = args.output / 'term_metadata.csv'
    term_metadata.to_csv(meta_path, index=False)
    print(f"  {meta_path} ({len(term_metadata)} terms)")

    if char_matrix is not None:
        char_path = args.output / 'character_document_matrix.csv'
        char_matrix.to_csv(char_path, index=False)
        print(f"  {char_path} ({len(char_matrix)} × {len(char_matrix.columns)})")

    print(f"\n{'=' * 60}")
    print("DONE")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
