#!/usr/bin/env python3
"""
Stratified Sampling for Hand-Coding

Creates a reproducible stratified sample of sentences or paragraphs from the
RSA corpus for hand-coding theoretical mechanisms (legitimacy, functional
aptness, spaces of equivalence, complexity empowerment).

Implements two-stage stratified sampling:
    1. Stage 1: Sample documents stratified by (actor_type, wave)
    2. Stage 2: Sample units (sentences or paragraphs) within selected documents

Train/test split is performed at document level to prevent data leakage.

Input:
    - Sentence-level parquet (preferred): Pre-segmented sentences with paragraph_id
    - Merged corpus parquet (fallback): Full document text, segments on-the-fly

Output:
    - sample_train.csv: Training set (70%)
    - sample_test.csv: Test set (30%)
    - sampling_report.json: Metadata and diagnostics

Usage:
    # Sample sentences (default)
    python stratified_sample.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/sampling/ \\
        --n-units 500 \\
        --seed 42

    # Sample paragraphs instead
    python stratified_sample.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/sampling/ \\
        --unit paragraph \\
        --n-units 500 \\
        --seed 42 \\
        --verbose

Requirements:
    pip install pandas pyarrow stanza
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)

# RSA filename parsing pattern (from term_document_matrix.py)
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

ACTOR_TRANSLATIONS = {
    'kommun': 'Municipality',
    'lansstyrelse': 'Prefecture',
    'MCF': 'MCF',
}

# Coding columns to add to output
CODING_COLUMNS = [
    'mechanism_legitimacy',
    'mechanism_functional',
    'mechanism_equivalence',
    'mechanism_complexity',
    'coder_notes',
]

# Unit type for generic references
UNIT_SENTENCE = 'sentence'
UNIT_PARAGRAPH = 'paragraph'


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def map_year_to_wave(year) -> Optional[int]:
    """
    Map publication year to wave number.

    Wave mapping:
        Wave 0: pre-2015
        Wave 1: 2015-2018
        Wave 2: 2019-2022
        Wave 3: >= 2023

    Parameters
    ----------
    year : int or float
        Publication year.

    Returns
    -------
    int or None
        Wave number (0, 1, 2, 3), or None if year is missing.
    """
    if pd.isna(year):
        return None
    year = int(year)
    if year < 2015:
        return 0
    elif 2015 <= year <= 2018:
        return 1
    elif 2019 <= year <= 2022:
        return 2
    else:  # year >= 2023
        return 3


def extract_entity(filename: str) -> str:
    """
    Extract entity name from RSA filename.

    Parameters
    ----------
    filename : str
        The PDF filename.

    Returns
    -------
    str
        The entity name, or 'unknown' if parsing fails.
    """
    match = RSA_FILENAME_PATTERN.match(filename)
    if match:
        return match.group('entity').strip()

    match = NON_RSA_PATTERN.match(filename)
    if match:
        return match.group('entity').strip()

    logger.warning(f"Could not parse entity from filename: {filename}")
    return 'unknown'


def translate_actor(actor: str) -> str:
    """Translate actor names from Swedish to English."""
    return ACTOR_TRANSLATIONS.get(actor, actor)


# =============================================================================
# DATA LOADING
# =============================================================================

def detect_input_format(df: pd.DataFrame) -> str:
    """
    Detect whether input is sentence-level or document-level.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded parquet dataframe.

    Returns
    -------
    str
        'sentence' if sentence-level, 'document' if document-level.
    """
    # Sentence-level has sentence_text/sentence_id columns
    if 'sentence_text' in df.columns and 'sentence_id' in df.columns:
        return 'sentence'
    # Document-level has text column with full documents
    elif 'text' in df.columns:
        return 'document'
    else:
        raise ValueError(
            f"Unknown input format. Columns: {list(df.columns)}. "
            "Expected either 'sentence_text' (sentence-level) or 'text' (document-level)."
        )


def load_data(input_path: Path, verbose: bool = False) -> tuple:
    """
    Load parquet file and detect format.

    Parameters
    ----------
    input_path : Path
        Path to parquet file.
    verbose : bool
        Print detailed information.

    Returns
    -------
    tuple of (pd.DataFrame, str)
        (Dataframe, format type: 'sentence' or 'document')
    """
    print(f"\nLoading data from: {input_path}")
    df = pd.read_parquet(input_path)
    input_format = detect_input_format(df)

    print(f"  Format detected: {input_format}-level")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    return df, input_format


def add_metadata(df: pd.DataFrame, input_format: str) -> pd.DataFrame:
    """
    Add missing metadata columns (actor_type, entity, wave).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    input_format : str
        'sentence' or 'document'.

    Returns
    -------
    pd.DataFrame
        Dataframe with added metadata.
    """
    df = df.copy()

    # Handle different column naming conventions
    # Sentence-level uses doc_id, document-level uses file
    if 'doc_id' in df.columns and 'file' not in df.columns:
        df['file'] = df['doc_id']
    elif 'file' not in df.columns:
        raise ValueError("Cannot determine document identifier column")

    # Add entity if missing
    if 'entity' not in df.columns:
        if 'municipality' in df.columns:
            df['entity'] = df['municipality']
        else:
            df['entity'] = df['file'].apply(extract_entity)

    # Add actor_type if missing
    if 'actor_type' not in df.columns and 'actor' not in df.columns:
        # Infer from filename patterns
        def infer_actor(filename):
            if 'Län' in filename or 'län' in filename:
                return 'länsstyrelse'
            elif 'MCF' in filename or 'NRSB' in filename:
                return 'MCF'
            else:
                return 'kommun'
        df['actor_type'] = df['file'].apply(infer_actor)
    elif 'actor' in df.columns and 'actor_type' not in df.columns:
        df['actor_type'] = df['actor']

    # Ensure year is numeric
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Add wave
    if 'wave' not in df.columns:
        df['wave'] = df['year'].apply(map_year_to_wave)

    return df


# =============================================================================
# PARAGRAPH AGGREGATION
# =============================================================================

def aggregate_to_paragraphs(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Aggregate sentence-level data to paragraph-level.

    Combines all sentences within each (doc_id, paragraph_id) into a single
    paragraph text, preserving metadata from the first sentence.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level dataframe with columns: doc_id, paragraph_id, sentence_text.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Paragraph-level dataframe with paragraph_text column.
    """
    if 'paragraph_id' not in df.columns:
        raise ValueError(
            "Input must have 'paragraph_id' column for paragraph sampling. "
            "Run preprocessing_bert.py with paragraph tracking enabled."
        )

    print("\n  Aggregating sentences to paragraphs...")
    n_sentences = len(df)
    n_paragraphs_before = df.groupby(['file', 'paragraph_id']).ngroups

    # Group by document and paragraph, concatenate sentences
    grouped = df.groupby(['file', 'paragraph_id'], sort=False)

    paragraphs = []
    for (file_id, para_id), group in grouped:
        # Sort sentences by sentence_id if available
        if 'sentence_id' in group.columns:
            group = group.sort_values('sentence_id')

        # Concatenate sentence texts with space
        para_text = ' '.join(group['sentence_text'].astype(str))
        word_count = len(para_text.split())

        # Get metadata from first sentence
        first_row = group.iloc[0]

        paragraphs.append({
            'file': file_id,
            'paragraph_id': para_id,
            'paragraph_text': para_text,
            'word_count': word_count,
            'n_sentences': len(group),
            'entity': first_row.get('entity', ''),
            'actor_type': first_row.get('actor_type', first_row.get('actor', '')),
            'year': first_row.get('year'),
            'wave': first_row.get('wave', map_year_to_wave(first_row.get('year'))),
        })

    result = pd.DataFrame(paragraphs)
    print(f"  Aggregated {n_sentences} sentences into {len(result)} paragraphs")

    return result


# =============================================================================
# SENTENCE SEGMENTATION
# =============================================================================

def segment_sentences(
    df: pd.DataFrame,
    min_words: int = 5,
    max_words: int = 300,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Segment document-level text into sentences using Stanza.

    Parameters
    ----------
    df : pd.DataFrame
        Document-level dataframe with 'text' column.
    min_words : int
        Minimum words per sentence.
    max_words : int
        Maximum words per sentence.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Sentence-level dataframe.
    """
    try:
        import stanza
    except ImportError:
        raise ImportError("stanza is required for sentence segmentation. Install with: pip install stanza")

    print("\n  Initializing Stanza Swedish tokenizer...")
    try:
        nlp = stanza.Pipeline('sv', processors='tokenize', verbose=False)
    except Exception:
        print("  Downloading Swedish model...")
        stanza.download('sv', processors='tokenize')
        nlp = stanza.Pipeline('sv', processors='tokenize', verbose=False)

    print("  Segmenting sentences...")
    sentences = []
    n_docs = len(df)

    for idx, row in df.iterrows():
        if verbose and idx % 50 == 0:
            print(f"    Processing document {idx + 1}/{n_docs}...")

        text = str(row.get('text', ''))
        if not text.strip():
            continue

        doc = nlp(text)
        for sent_idx, sentence in enumerate(doc.sentences):
            sent_text = sentence.text.strip()
            word_count = len(sent_text.split())

            # Filter by word count
            if word_count < min_words or word_count > max_words:
                continue

            # Check alphabetic ratio (filter OCR artifacts)
            alpha_chars = sum(1 for c in sent_text if c.isalpha())
            if len(sent_text) > 0 and alpha_chars / len(sent_text) < 0.5:
                continue

            sentences.append({
                'file': row['file'],
                'entity': row.get('entity', ''),
                'actor_type': row.get('actor_type', row.get('actor', '')),
                'year': row['year'],
                'wave': row.get('wave', map_year_to_wave(row['year'])),
                'sentence_id': sent_idx + 1,
                'sentence_text': sent_text,
                'word_count': word_count,
            })

    result = pd.DataFrame(sentences)
    print(f"  Segmented into {len(result)} sentences from {n_docs} documents")

    return result


# =============================================================================
# STRATIFIED SAMPLING
# =============================================================================

def compute_allocation(
    df: pd.DataFrame,
    n_units: int,
    units_per_doc: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Compute proportional allocation by (actor_type, wave).

    Parameters
    ----------
    df : pd.DataFrame
        Unit-level dataframe (sentences or paragraphs).
    n_units : int
        Target total units.
    units_per_doc : int
        Max units per document.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    pd.DataFrame
        Allocation table with columns: actor_type, wave, n_docs, n_units_target.
    """
    # Count documents per stratum
    doc_counts = df.groupby(['actor_type', 'wave'])['file'].nunique().reset_index()
    doc_counts.columns = ['actor_type', 'wave', 'n_docs']

    total_docs = doc_counts['n_docs'].sum()
    doc_counts['proportion'] = doc_counts['n_docs'] / total_docs

    # Allocate units proportionally
    doc_counts['n_units_target'] = (doc_counts['proportion'] * n_units).astype(int)

    # Adjust for rounding (distribute remainder randomly)
    remainder = n_units - doc_counts['n_units_target'].sum()
    if remainder > 0:
        indices = rng.choice(len(doc_counts), size=remainder, replace=False)
        doc_counts.loc[doc_counts.index[indices], 'n_units_target'] += 1

    # Estimate documents needed per stratum
    doc_counts['n_docs_needed'] = np.ceil(
        doc_counts['n_units_target'] / units_per_doc
    ).astype(int)

    return doc_counts


def sample_documents(
    df: pd.DataFrame,
    allocation: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Stage 1: Sample documents stratified by (actor_type, wave).

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level dataframe.
    allocation : pd.DataFrame
        Allocation table from compute_allocation().
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    pd.DataFrame
        Sampled documents (unique files with metadata).
    """
    sampled_docs = []

    for _, row in allocation.iterrows():
        actor = row['actor_type']
        wave = row['wave']
        n_needed = row['n_docs_needed']

        stratum_docs = df[(df['actor_type'] == actor) & (df['wave'] == wave)]['file'].unique()

        if len(stratum_docs) == 0:
            logger.warning(f"No documents in stratum ({actor}, wave {wave})")
            continue

        # Sample documents (with replacement if not enough)
        n_sample = min(n_needed, len(stratum_docs))
        selected = rng.choice(stratum_docs, size=n_sample, replace=False)

        for doc in selected:
            sampled_docs.append({
                'file': doc,
                'actor_type': actor,
                'wave': wave,
            })

    return pd.DataFrame(sampled_docs)


def sample_units(
    df: pd.DataFrame,
    sampled_docs: pd.DataFrame,
    allocation: pd.DataFrame,
    units_per_doc: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Stage 2: Sample units (sentences or paragraphs) within selected documents.

    Parameters
    ----------
    df : pd.DataFrame
        Unit-level dataframe (sentences or paragraphs).
    sampled_docs : pd.DataFrame
        Sampled documents from stage 1.
    allocation : pd.DataFrame
        Allocation table.
    units_per_doc : int
        Max units per document.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    pd.DataFrame
        Sampled units.
    """
    sampled_units = []

    for _, alloc_row in allocation.iterrows():
        actor = alloc_row['actor_type']
        wave = alloc_row['wave']
        target = alloc_row['n_units_target']

        # Get documents for this stratum
        stratum_docs = sampled_docs[
            (sampled_docs['actor_type'] == actor) & (sampled_docs['wave'] == wave)
        ]['file'].tolist()

        if not stratum_docs:
            continue

        # Sample units from each document
        stratum_units = []
        for doc_file in stratum_docs:
            doc_units = df[df['file'] == doc_file].copy()
            if len(doc_units) == 0:
                continue

            n_sample = min(units_per_doc, len(doc_units))
            selected_idx = rng.choice(len(doc_units), size=n_sample, replace=False)
            stratum_units.append(doc_units.iloc[selected_idx])

        if not stratum_units:
            continue

        stratum_df = pd.concat(stratum_units, ignore_index=True)

        # If we have more than target, subsample
        if len(stratum_df) > target:
            selected_idx = rng.choice(len(stratum_df), size=target, replace=False)
            stratum_df = stratum_df.iloc[selected_idx]

        sampled_units.append(stratum_df)

    if not sampled_units:
        raise ValueError("No units sampled. Check input data.")

    return pd.concat(sampled_units, ignore_index=True)


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Split sample at document level (prevent data leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Sampled sentences.
    train_ratio : float
        Proportion for training set.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'split' column ('train' or 'test').
    """
    df = df.copy()

    # Get unique documents
    unique_docs = df['file'].unique()
    n_train = int(len(unique_docs) * train_ratio)

    # Randomly assign documents to train/test
    rng.shuffle(unique_docs)
    train_docs = set(unique_docs[:n_train])

    df['split'] = df['file'].apply(lambda x: 'train' if x in train_docs else 'test')

    return df


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_output(df: pd.DataFrame, unit_type: str) -> pd.DataFrame:
    """
    Format output with sample IDs and coding columns.

    Parameters
    ----------
    df : pd.DataFrame
        Sampled units with split column.
    unit_type : str
        'sentence' or 'paragraph'.

    Returns
    -------
    pd.DataFrame
        Formatted output ready for hand-coding.
    """
    df = df.copy()

    # Generate sample IDs
    df = df.reset_index(drop=True)
    prefix = 'P' if unit_type == UNIT_PARAGRAPH else 'S'
    df['sample_id'] = [f"{prefix}{i+1:04d}" for i in range(len(df))]

    # Rename 'file' to 'doc_id' if present (avoid duplicate if both exist)
    if 'file' in df.columns:
        if 'doc_id' in df.columns:
            df = df.drop(columns=['doc_id'])
        df = df.rename(columns={'file': 'doc_id'})

    # Add empty coding columns
    for col in CODING_COLUMNS:
        df[col] = ''

    # Select and order columns based on unit type
    if unit_type == UNIT_PARAGRAPH:
        output_cols = [
            'sample_id',
            'doc_id',
            'entity',
            'actor_type',
            'year',
            'wave',
            'paragraph_id',
            'paragraph_text',
            'word_count',
            'n_sentences',
            'split',
        ] + CODING_COLUMNS
    else:
        output_cols = [
            'sample_id',
            'doc_id',
            'entity',
            'actor_type',
            'year',
            'wave',
            'sentence_id',
            'paragraph_id',
            'sentence_text',
            'word_count',
            'split',
        ] + CODING_COLUMNS

    # Keep only columns that exist (deduplicate)
    seen = set()
    output_cols = [c for c in output_cols if c in df.columns and c not in seen and not seen.add(c)]

    return df[output_cols]


def generate_report(
    df: pd.DataFrame,
    allocation: pd.DataFrame,
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    unit_type: str,
) -> dict:
    """
    Generate JSON report with sampling metadata and diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Final sampled units.
    allocation : pd.DataFrame
        Allocation table.
    input_path : Path
        Input file path.
    output_dir : Path
        Output directory.
    args : argparse.Namespace
        Command line arguments.
    unit_type : str
        'sentence' or 'paragraph'.

    Returns
    -------
    dict
        Report dictionary.
    """
    report = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'input_file': str(input_path),
            'output_dir': str(output_dir),
            'seed': args.seed,
            'unit_type': unit_type,
            'n_units_requested': args.n_units,
            'n_units_sampled': len(df),
            'units_per_doc': args.units_per_doc,
            'train_ratio': args.train_ratio,
            'min_words': args.min_words,
        },
        'sample_statistics': {
            f'total_{unit_type}s': len(df),
            'total_documents': int(df['doc_id'].nunique()),
            f'train_{unit_type}s': len(df[df['split'] == 'train']),
            f'test_{unit_type}s': len(df[df['split'] == 'test']),
            'train_documents': int(df[df['split'] == 'train']['doc_id'].nunique()),
            'test_documents': int(df[df['split'] == 'test']['doc_id'].nunique()),
        },
        'stratification': {
            'by_actor': df.groupby('actor_type').size().to_dict(),
            'by_wave': {int(k): v for k, v in df.groupby('wave').size().to_dict().items()},
            'by_actor_wave': {
                f"{actor}_wave{wave}": count
                for (actor, wave), count in df.groupby(['actor_type', 'wave']).size().to_dict().items()
            },
        },
        'allocation_table': allocation.to_dict(orient='records'),
        'word_count_stats': {
            'mean': float(df['word_count'].mean()),
            'median': float(df['word_count'].median()),
            'min': int(df['word_count'].min()),
            'max': int(df['word_count'].max()),
        },
    }

    # Add warnings
    warnings = []
    actors_present = set(df['actor_type'].unique())
    expected_actors = {'kommun', 'lansstyrelse', 'MCF'}
    missing_actors = expected_actors - actors_present
    if missing_actors:
        warnings.append(
            f"Missing actor types: {missing_actors}. "
            "Consider regenerating corpus with all actors."
        )

    if len(df) < args.n_units:
        warnings.append(
            f"Could only sample {len(df)} {unit_type}s (requested {args.n_units})."
        )

    report['warnings'] = warnings

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create stratified sample of RSA sentences or paragraphs for hand-coding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Sample sentences (default)
    python stratified_sample.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/sampling/ \\
        --n-units 500 --seed 42

    # Sample paragraphs instead
    python stratified_sample.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/sampling/ \\
        --unit paragraph \\
        --n-units 500 --seed 42 --verbose
"""
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to sentence parquet OR merged corpus parquet'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./results/sampling'),
        help='Output directory (default: results/sampling/)'
    )

    parser.add_argument(
        '--unit',
        choices=[UNIT_SENTENCE, UNIT_PARAGRAPH],
        default=UNIT_SENTENCE,
        help='Sampling unit: sentence or paragraph (default: sentence)'
    )

    parser.add_argument(
        '--n-units',
        type=int,
        default=500,
        help='Total units to sample (default: 500)'
    )

    parser.add_argument(
        '--units-per-doc',
        type=int,
        default=10,
        help='Maximum units per document (default: 10)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Train split ratio (default: 0.7)'
    )

    parser.add_argument(
        '--min-words',
        type=int,
        default=5,
        help='Minimum words per unit (default: 5)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress messages'
    )

    args = parser.parse_args()
    unit_type = args.unit

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    print("=" * 60)
    print("STRATIFIED SAMPLING FOR HAND-CODING")
    print("=" * 60)

    # Initialize random number generator
    rng = np.random.default_rng(args.seed)
    print(f"\nRandom seed: {args.seed}")
    print(f"Sampling unit: {unit_type}")

    # Load data
    df, input_format = load_data(args.input, args.verbose)

    # Add metadata
    print("\nAdding metadata...")
    df = add_metadata(df, input_format)

    # Segment sentences if document-level
    if input_format == 'document':
        print("\nDocument-level input detected. Segmenting sentences...")
        df = segment_sentences(
            df,
            min_words=args.min_words,
            verbose=args.verbose,
        )
    else:
        # Filter by min_words for sentence-level input
        print(f"\nFiltering sentences (min {args.min_words} words)...")
        original_count = len(df)
        df = df[df['word_count'] >= args.min_words].copy()
        print(f"  Kept {len(df)}/{original_count} sentences")

    # If sampling paragraphs, aggregate sentences
    if unit_type == UNIT_PARAGRAPH:
        df = aggregate_to_paragraphs(df, args.verbose)
        # Filter paragraphs by min_words
        print(f"\nFiltering paragraphs (min {args.min_words} words)...")
        original_count = len(df)
        df = df[df['word_count'] >= args.min_words].copy()
        print(f"  Kept {len(df)}/{original_count} paragraphs")

    # Report corpus statistics
    print("\nCorpus statistics:")
    print(f"  Total {unit_type}s: {len(df)}")
    print(f"  Total documents: {df['file'].nunique()}")
    print(f"\n  By actor type:")
    for actor in sorted(df['actor_type'].unique()):
        n_units = len(df[df['actor_type'] == actor])
        n_doc = df[df['actor_type'] == actor]['file'].nunique()
        print(f"    {translate_actor(actor)}: {n_units} {unit_type}s, {n_doc} documents")
    print(f"\n  By wave:")
    wave_labels = {0: 'pre-2015', 1: '2015-2018', 2: '2019-2022', 3: '≥2023'}
    for wave in sorted(df['wave'].unique()):
        n_units = len(df[df['wave'] == wave])
        print(f"    Wave {wave} ({wave_labels.get(wave, '?')}): {n_units} {unit_type}s")

    # Compute allocation
    print(f"\n{'='*60}")
    print("STAGE 1: Computing stratified allocation")
    print(f"{'='*60}")
    allocation = compute_allocation(
        df,
        n_units=args.n_units,
        units_per_doc=args.units_per_doc,
        rng=rng,
    )
    print("\nAllocation table:")
    print(allocation.to_string(index=False))

    # Sample documents
    print(f"\n{'='*60}")
    print("STAGE 2: Sampling documents")
    print(f"{'='*60}")
    sampled_docs = sample_documents(df, allocation, rng)
    print(f"\nSampled {len(sampled_docs)} documents")

    # Sample units
    print(f"\n{'='*60}")
    print(f"STAGE 3: Sampling {unit_type}s within documents")
    print(f"{'='*60}")
    sampled = sample_units(
        df, sampled_docs, allocation,
        units_per_doc=args.units_per_doc,
        rng=rng,
    )
    print(f"\nSampled {len(sampled)} {unit_type}s")

    # Split train/test
    print(f"\n{'='*60}")
    print("STAGE 4: Train/test split (document-level)")
    print(f"{'='*60}")
    sampled = split_train_test(sampled, args.train_ratio, rng)
    n_train = len(sampled[sampled['split'] == 'train'])
    n_test = len(sampled[sampled['split'] == 'test'])
    print(f"\nTrain: {n_train} {unit_type}s ({n_train/len(sampled):.1%})")
    print(f"Test:  {n_test} {unit_type}s ({n_test/len(sampled):.1%})")

    # Format output
    print(f"\n{'='*60}")
    print("STAGE 5: Formatting output")
    print(f"{'='*60}")
    output_df = format_output(sampled, unit_type)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save train/test splits
    train_df = output_df[output_df['split'] == 'train'].copy()
    test_df = output_df[output_df['split'] == 'test'].copy()

    train_path = args.output / 'sample_train.csv'
    test_path = args.output / 'sample_test.csv'
    full_path = args.output / 'sample_full.csv'

    train_df.to_csv(train_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    output_df.to_csv(full_path, index=False, encoding='utf-8')

    print(f"\nSaved: {train_path} ({len(train_df)} rows)")
    print(f"Saved: {test_path} ({len(test_df)} rows)")
    print(f"Saved: {full_path} ({len(output_df)} rows)")

    # Generate and save report
    report = generate_report(output_df, allocation, args.input, args.output, args, unit_type)
    report_path = args.output / 'sampling_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {report_path}")

    # Print warnings
    if report['warnings']:
        print("\nWarnings:")
        for warning in report['warnings']:
            print(f"  - {warning}")

    # Final summary
    print(f"\n{'='*60}")
    print("SAMPLING COMPLETE")
    print(f"{'='*60}")
    n_docs = output_df['doc_id'].nunique()
    print(f"\nSample size: {len(output_df)} {unit_type}s from {n_docs} documents")
    print(f"Train/test split: {n_train}/{n_test} ({args.train_ratio:.0%}/{1-args.train_ratio:.0%})")
    print(f"\nOutput directory: {args.output}")
    print(f"\nFiles for hand-coding:")
    print(f"  - sample_train.csv: {n_train} {unit_type}s (for training BERT)")
    print(f"  - sample_test.csv: {n_test} {unit_type}s (for evaluation)")
    print(f"\nCoding columns to fill:")
    for col in CODING_COLUMNS:
        print(f"  - {col}")

    print(f"\n{'='*60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
