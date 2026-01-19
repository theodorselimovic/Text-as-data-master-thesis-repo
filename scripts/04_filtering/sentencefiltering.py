#!/usr/bin/env python3
"""
Sentence Filtering and Vectorization for RSA Documents

This script filters lemmatized sentences to those containing expanded terms,
then creates sentence embeddings using FastText. The output is used for
co-occurrence analysis and other downstream tasks.

Pipeline:
    1. Load expanded terms from vectoranalysis.py output
    2. Load lemmatized sentences
    3. Filter to sentences containing target terms
    4. Assign categories to each sentence
    5. Vectorize sentences using FastText embeddings
    6. Expand to category-sentence pairs
    7. Save results in multiple formats

Output Files:
    - sentence_vectors_with_metadata.parquet: Complete dataset with vectors
    - sentence_vectors_metadata.csv: Metadata without vectors (for inspection)
    - sentence_vectors.npy: Just vectors as numpy array
    - sentence_vectors_index.csv: Index mapping for numpy array

Usage:
    python sentencefiltering.py \\
        --expanded-terms data/expanded_terms/expanded_terms_lemmatized_complete.csv \\
        --sentences data/processed/sentences_lemmatized.parquet \\
        --model-path /path/to/cc.sv.300.bin \\
        --output-dir data/vectors

Requirements:
    - FastText Swedish model (cc.sv.300.bin)
    - Python packages: numpy, pandas, fasttext, scikit-learn

Author: Swedish Risk Analysis Text-as-Data Project
Version: 2.0
Date: 2025-01-04
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import fasttext
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default file paths
DEFAULT_EXPANDED_TERMS = 'data/expanded_terms/expanded_terms_lemmatized_complete.csv'
DEFAULT_SENTENCES = 'data/processed/sentences_lemmatized.parquet'
DEFAULT_OUTPUT_DIR = 'data/vectors'

# Vectorization method
AGGREGATION_METHOD = 'mean'  # 'mean' (average) or 'sum' (addition)

# Vector quality threshold
VECTOR_NORM_THRESHOLD = 0.01  # Minimum norm for meaningful vector


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the script.
    
    Parameters:
    -----------
    verbose : bool
        If True, set logging level to DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """
    Load expanded terms and sentence data.
    
    This class handles loading the expanded terms from vectoranalysis.py
    and the lemmatized sentences from preprocessing.
    """
    
    @staticmethod
    def load_expanded_terms(filepath: Path) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        Load expanded terms and create category lookup.
        
        Parameters:
        -----------
        filepath : Path
            Path to expanded_terms_lemmatized_complete.csv
            
        Returns:
        --------
        Tuple[Dict[str, Set[str]], Set[str]]
            (category_lemmas dict, all_target_lemmas set)
        """
        logging.info("=" * 80)
        logging.info("LOADING EXPANDED TERMS")
        logging.info("=" * 80)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Expanded terms file not found: {filepath}")
        
        df_expanded = pd.read_csv(filepath)
        
        # Create lookup dictionary: category -> set of lemmas
        category_lemmas = {}
        all_target_lemmas = set()
        
        for category in df_expanded['category'].unique():
            lemmas = set(df_expanded[df_expanded['category'] == category]['lemma'].tolist())
            category_lemmas[category] = lemmas
            all_target_lemmas.update(lemmas)
            logging.info(f"  {category}: {len(lemmas)} terms")
        
        logging.info(f"\nTotal unique target terms: {len(all_target_lemmas)}")
        
        return category_lemmas, all_target_lemmas
    
    @staticmethod
    def load_sentences(filepath: Path) -> pd.DataFrame:
        """
        Load lemmatized sentences.
        
        Parameters:
        -----------
        filepath : Path
            Path to sentences_lemmatized.parquet
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with sentence data
        """
        logging.info("\n" + "=" * 80)
        logging.info("LOADING SENTENCE DATA")
        logging.info("=" * 80)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Sentences file not found: {filepath}")
        
        df_sentences = pd.read_parquet(filepath)
        
        logging.info(f"Loaded {len(df_sentences):,} sentences")
        logging.info(f"Columns: {df_sentences.columns.tolist()}")
        
        return df_sentences


# =============================================================================
# SENTENCE FILTER
# =============================================================================

class SentenceFilter:
    """
    Filter sentences containing target terms and assign categories.
    
    This class identifies which sentences contain expanded terms and
    assigns appropriate categories to each sentence.
    """
    
    @staticmethod
    def find_target_terms(
        sentence_text: str,
        all_target_lemmas: Set[str]
    ) -> List[str]:
        """
        Find which target terms appear in a sentence.
        
        Parameters:
        -----------
        sentence_text : str
            Lemmatized sentence text
        all_target_lemmas : Set[str]
            Set of all target lemmas
            
        Returns:
        --------
        List[str]
            List of target terms found in sentence
        """
        words = sentence_text.split()
        return [word for word in words if word in all_target_lemmas]
    
    @staticmethod
    def get_categories_for_terms(
        terms: List[str],
        category_lemmas: Dict[str, Set[str]]
    ) -> List[str]:
        """
        Get all categories that these terms belong to.
        
        Parameters:
        -----------
        terms : List[str]
            List of terms found in sentence
        category_lemmas : Dict[str, Set[str]]
            Mapping from category to set of lemmas
            
        Returns:
        --------
        List[str]
            List of categories for these terms
        """
        categories = set()
        for term in terms:
            for category, lemmas in category_lemmas.items():
                if term in lemmas:
                    categories.add(category)
        return list(categories)
    
    def filter_and_categorize(
        self,
        df_sentences: pd.DataFrame,
        all_target_lemmas: Set[str],
        category_lemmas: Dict[str, Set[str]]
    ) -> pd.DataFrame:
        """
        Filter sentences and assign categories.
        
        Parameters:
        -----------
        df_sentences : pd.DataFrame
            All sentences
        all_target_lemmas : Set[str]
            All target lemmas
        category_lemmas : Dict[str, Set[str]]
            Category to lemmas mapping
            
        Returns:
        --------
        pd.DataFrame
            Filtered sentences with target terms and categories
        """
        logging.info("\n" + "=" * 80)
        logging.info("FILTERING SENTENCES WITH TARGET TERMS")
        logging.info("=" * 80)
        
        # Find target terms in each sentence
        logging.info("Identifying sentences with target terms...")
        df_sentences['target_terms'] = df_sentences['sentence_text'].apply(
            lambda x: self.find_target_terms(x, all_target_lemmas)
        )
        
        # Filter to sentences with at least one target term
        df_with_targets = df_sentences[
            df_sentences['target_terms'].apply(len) > 0
        ].copy()
        
        pct = len(df_with_targets) / len(df_sentences) * 100
        logging.info(
            f"Sentences containing target terms: {len(df_with_targets):,} "
            f"({pct:.2f}%)"
        )
        
        # Assign categories
        df_with_targets['categories'] = df_with_targets['target_terms'].apply(
            lambda terms: self.get_categories_for_terms(terms, category_lemmas)
        )
        
        # Show distribution
        logging.info("\nSentences by number of categories:")
        category_counts = df_with_targets['categories'].apply(len).value_counts()
        for n_cats in sorted(category_counts.index):
            logging.info(f"  {n_cats} categories: {category_counts[n_cats]:,} sentences")
        
        return df_with_targets


# =============================================================================
# SENTENCE VECTORIZER
# =============================================================================

class SentenceVectorizer:
    """
    Convert sentences to vectors using FastText embeddings.
    
    This class handles loading the FastText model and vectorizing sentences
    by aggregating word embeddings.
    """
    
    def __init__(self, model_path: Path):
        """
        Initialize vectorizer.
        
        Parameters:
        -----------
        model_path : Path
            Path to FastText .bin model
        """
        self.model_path = model_path
        self.model = None
        self.dimension = None
        
    def load_model(self) -> None:
        """
        Load FastText model from disk.
        
        Raises:
        -------
        FileNotFoundError
            If model file doesn't exist
        RuntimeError
            If model loading fails
        """
        logging.info("\n" + "=" * 80)
        logging.info("LOADING FASTTEXT MODEL")
        logging.info("=" * 80)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logging.info("Loading FastText model (this may take a few minutes)...")
        start_time = time.time()
        
        try:
            self.model = fasttext.load_model(str(self.model_path))
            self.dimension = self.model.get_dimension()
            
            elapsed = time.time() - start_time
            logging.info(f"✓ Model loaded in {elapsed:.1f}s")
            logging.info(f"  Dimension: {self.dimension}")
            logging.info("  Has subword information - can handle OOV words!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def vectorize_sentence(
        self,
        sentence_text: str,
        method: str = AGGREGATION_METHOD
    ) -> Dict:
        """
        Convert a sentence to a single vector by aggregating word vectors.
        
        Parameters:
        -----------
        sentence_text : str
            The lemmatized sentence
        method : str
            'mean' (average) or 'sum' (addition)
            
        Returns:
        --------
        Dict
            Dictionary with keys:
            - 'vector': np.ndarray
            - 'words_found': int
            - 'words_total': int
            - 'coverage': float
            
        References:
        -----------
        Arora et al. (2017) "A Simple but Tough-to-Beat Baseline for Sentence Embeddings"
        
        Note:
        -----
        FastText can generate vectors for most words using subword info, but very
        unusual artifacts may still fail to produce meaningful vectors.
        We check vector norms to identify such cases.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        words = sentence_text.split()
        vectors = []
        words_with_vectors = 0
        
        for word in words:
            # FastText can get a vector for any word (uses subword info)
            vector = self.model.get_word_vector(word)
            
            # Check if the vector is meaningful (not just zeros or near-zeros)
            if np.linalg.norm(vector) > VECTOR_NORM_THRESHOLD:
                vectors.append(vector)
                words_with_vectors += 1
        
        if len(vectors) == 0:
            # No meaningful vectors - return zero vector
            return {
                'vector': np.zeros(self.dimension),
                'words_found': 0,
                'words_total': len(words),
                'coverage': 0.0
            }
        
        vectors_array = np.array(vectors)
        
        # Aggregate vectors
        if method == 'mean':
            sentence_vector = np.mean(vectors_array, axis=0)
        elif method == 'sum':
            sentence_vector = np.sum(vectors_array, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        coverage = words_with_vectors / len(words) if len(words) > 0 else 0.0
        
        return {
            'vector': sentence_vector,
            'words_found': words_with_vectors,
            'words_total': len(words),
            'coverage': coverage
        }
    
    def vectorize_dataframe(
        self,
        df: pd.DataFrame,
        method: str = AGGREGATION_METHOD
    ) -> pd.DataFrame:
        """
        Vectorize all sentences in a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'sentence_text' column
        method : str
            Aggregation method ('mean' or 'sum')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added columns:
            - sentence_vector
            - words_found
            - coverage
        """
        logging.info("\n" + "=" * 80)
        logging.info("VECTORIZING SENTENCES")
        logging.info("=" * 80)
        logging.info(f"Aggregation method: '{method}'")
        logging.info(f"Vectorizing {len(df):,} sentences...")
        logging.info("This may take a few minutes...\n")
        
        start_time = time.time()
        
        # Vectorize all sentences
        vectorization_results = df['sentence_text'].apply(
            lambda x: self.vectorize_sentence(x, method=method)
        )
        
        # Unpack results into separate columns
        df = df.copy()
        df['sentence_vector'] = vectorization_results.apply(lambda x: x['vector'])
        df['words_found'] = vectorization_results.apply(lambda x: x['words_found'])
        df['coverage'] = vectorization_results.apply(lambda x: x['coverage'])
        
        elapsed = time.time() - start_time
        logging.info(f"✓ Vectorization complete in {elapsed:.1f}s")
        logging.info(f"  Successfully vectorized {len(df):,} sentences")
        
        return df
    
    def free_memory(self) -> None:
        """Delete model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            logging.info("✓ Model deleted, memory freed")


# =============================================================================
# DATA PROCESSOR
# =============================================================================

class DataProcessor:
    """
    Process vectorized sentences for analysis.
    
    This class handles expanding sentences to category-sentence pairs,
    computing statistics, and saving results.
    """
    
    @staticmethod
    def print_statistics(df: pd.DataFrame) -> None:
        """
        Print summary statistics about the vectorized data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Vectorized sentences
        """
        logging.info("\n" + "=" * 80)
        logging.info("SUMMARY STATISTICS")
        logging.info("=" * 80)
        
        logging.info(f"\nTotal vectorized sentences: {len(df):,}")
        logging.info(f"Average coverage: {df['coverage'].mean():.2%}")
        logging.info(f"Median coverage: {df['coverage'].median():.2%}")
        
        # Coverage distribution
        logging.info("\nCoverage distribution:")
        logging.info(f"  Min:  {df['coverage'].min():.2%}")
        logging.info(f"  25%:  {df['coverage'].quantile(0.25):.2%}")
        logging.info(f"  50%:  {df['coverage'].median():.2%}")
        logging.info(f"  75%:  {df['coverage'].quantile(0.75):.2%}")
        logging.info(f"  Max:  {df['coverage'].max():.2%}")
        
        # Low coverage sentences
        low_coverage = df[df['coverage'] < 0.8]
        pct_low = len(low_coverage) / len(df) * 100
        logging.info(f"\nSentences with coverage < 80%: {len(low_coverage):,} ({pct_low:.2f}%)")
        
        if len(low_coverage) > 0:
            logging.info("  Example low coverage sentences:")
            for idx in range(min(3, len(low_coverage))):
                example = low_coverage.iloc[idx]
                cov = example['coverage']
                found = example['words_found']
                total = example.get('word_count', found)
                text = example['sentence_text'][:80]
                logging.info(f"    Coverage: {cov:.2%} | Words: {found}/{total}")
                logging.info(f"    Text: {text}...")
        
        # Temporal distribution
        logging.info("\nSentences by year:")
        year_counts = df.groupby('year').size().sort_index()
        for year, count in year_counts.items():
            logging.info(f"  {year}: {count:,}")
        
        # Municipal distribution
        logging.info("\nSentences by municipality (top 10):")
        muni_counts = df.groupby('municipality').size().sort_values(ascending=False).head(10)
        for muni, count in muni_counts.items():
            logging.info(f"  {muni}: {count:,}")
    
    @staticmethod
    def expand_to_category_pairs(df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand sentences to category-sentence pairs.
        
        A sentence containing terms from multiple categories will appear
        once for each category.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Sentences with 'categories' column
            
        Returns:
        --------
        pd.DataFrame
            Expanded DataFrame with one row per category-sentence pair
        """
        logging.info("\n" + "=" * 80)
        logging.info("EXPANDING TO CATEGORY-SENTENCE PAIRS")
        logging.info("=" * 80)
        
        rows_expanded = []
        for _, row in df.iterrows():
            for category in row['categories']:
                rows_expanded.append({
                    'doc_id': row['doc_id'],
                    'municipality': row['municipality'],
                    'year': row['year'],
                    'maskad': row['maskad'],
                    'sentence_id': row['sentence_id'],
                    'category': category,
                    'target_terms': ', '.join(row['target_terms']),
                    'sentence_text': row['sentence_text'],
                    'sentence_vector': row['sentence_vector'],
                    'word_count': row['word_count'],
                    'words_found': row['words_found'],
                    'coverage': row['coverage']
                })
        
        df_expanded = pd.DataFrame(rows_expanded)
        
        logging.info(f"Total category-sentence pairs: {len(df_expanded):,}")
        logging.info("\nPairs by category:")
        category_counts = df_expanded.groupby('category').size()
        for category in sorted(category_counts.index):
            logging.info(f"  {category}: {category_counts[category]:,}")
        
        return df_expanded
    
    @staticmethod
    def save_results(df: pd.DataFrame, output_dir: Path) -> None:
        """
        Save results in multiple formats.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Expanded category-sentence pairs with vectors
        output_dir : Path
            Directory to save output files
        """
        logging.info("\n" + "=" * 80)
        logging.info("SAVING RESULTS")
        logging.info("=" * 80)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Metadata CSV (without vectors)
        metadata_path = output_dir / 'sentence_vectors_metadata.csv'
        df_metadata = df.drop(columns=['sentence_vector'])
        df_metadata.to_csv(metadata_path, index=False, encoding='utf-8')
        logging.info(f"✓ Saved metadata to: {metadata_path}")
        
        # 2. Full data with vectors (Parquet)
        parquet_path = output_dir / 'sentence_vectors_with_metadata.parquet'
        df.to_parquet(parquet_path, index=False)
        logging.info(f"✓ Saved full data with vectors to: {parquet_path}")
        
        # 3. Just vectors as numpy array
        npy_path = output_dir / 'sentence_vectors.npy'
        sentence_vectors = np.stack(df['sentence_vector'].values)
        np.save(npy_path, sentence_vectors)
        logging.info(f"✓ Saved sentence vectors to: {npy_path}")
        logging.info(f"  Shape: {sentence_vectors.shape} (sentences × vector_dim)")
        
        # 4. Index mapping
        index_path = output_dir / 'sentence_vectors_index.csv'
        df.reset_index(drop=True)[['doc_id', 'municipality', 'year', 'category']].to_csv(
            index_path,
            index=True,
            encoding='utf-8'
        )
        logging.info(f"✓ Saved index mapping to: {index_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    expanded_terms_path: Path,
    sentences_path: Path,
    model_path: Path,
    output_dir: Path,
    aggregation_method: str = AGGREGATION_METHOD,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run the complete sentence filtering and vectorization pipeline.
    
    Parameters:
    -----------
    expanded_terms_path : Path
        Path to expanded_terms_lemmatized_complete.csv
    sentences_path : Path
        Path to sentences_lemmatized.parquet
    model_path : Path
        Path to FastText .bin model
    output_dir : Path
        Directory for output files
    aggregation_method : str
        'mean' or 'sum' for vector aggregation
    verbose : bool
        Enable verbose logging
        
    Returns:
    --------
    pd.DataFrame
        Final expanded category-sentence pairs with vectors
    """
    setup_logging(verbose)
    
    logging.info("=" * 80)
    logging.info("SENTENCE FILTERING AND VECTORIZATION PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Expanded terms: {expanded_terms_path}")
    logging.info(f"Sentences: {sentences_path}")
    logging.info(f"Model: {model_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Aggregation method: {aggregation_method}")
    
    # Step 1: Load data
    category_lemmas, all_target_lemmas = DataLoader.load_expanded_terms(
        expanded_terms_path
    )
    df_sentences = DataLoader.load_sentences(sentences_path)
    
    # Step 2: Filter and categorize
    sentence_filter = SentenceFilter()
    df_with_targets = sentence_filter.filter_and_categorize(
        df_sentences,
        all_target_lemmas,
        category_lemmas
    )
    
    # Step 3: Vectorize
    vectorizer = SentenceVectorizer(model_path)
    vectorizer.load_model()
    df_vectorized = vectorizer.vectorize_dataframe(df_with_targets, aggregation_method)
    vectorizer.free_memory()
    
    # Step 4: Statistics
    processor = DataProcessor()
    processor.print_statistics(df_vectorized)
    
    # Step 5: Expand to category pairs
    df_final = processor.expand_to_category_pairs(df_vectorized)
    
    # Step 6: Save results
    processor.save_results(df_final, output_dir)
    
    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("PIPELINE COMPLETE!")
    logging.info("=" * 80)
    logging.info("\nOutput files:")
    logging.info(f"  1. {output_dir}/sentence_vectors_with_metadata.parquet")
    logging.info(f"  2. {output_dir}/sentence_vectors_metadata.csv")
    logging.info(f"  3. {output_dir}/sentence_vectors.npy")
    logging.info(f"  4. {output_dir}/sentence_vectors_index.csv")
    logging.info("\nNext steps:")
    logging.info("  - Run data_diagnostic.py to verify data quality")
    logging.info("  - Run cooccurrence_analysis.py for chi-square tests")
    
    return df_final


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns:
    --------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Filter sentences and create embeddings for RSA analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (with defaults)
    python sentencefiltering.py --model-path /path/to/cc.sv.300.bin
    
    # Custom paths
    python sentencefiltering.py \\
        --expanded-terms data/expanded_terms/terms.csv \\
        --sentences data/processed/sentences.parquet \\
        --model-path /path/to/model.bin \\
        --output-dir results/vectors
    
    # Use sum aggregation instead of mean
    python sentencefiltering.py \\
        --model-path /path/to/model.bin \\
        --aggregation sum

Output:
    Creates 4 files in output directory:
    1. sentence_vectors_with_metadata.parquet - Full dataset
    2. sentence_vectors_metadata.csv - Metadata only
    3. sentence_vectors.npy - Just vectors (300-dim)
    4. sentence_vectors_index.csv - Index mapping
        """
    )
    
    parser.add_argument(
        '--expanded-terms',
        type=Path,
        default=Path(DEFAULT_EXPANDED_TERMS),
        help=f'Path to expanded terms CSV (default: {DEFAULT_EXPANDED_TERMS})'
    )
    
    parser.add_argument(
        '--sentences',
        type=Path,
        default=Path(DEFAULT_SENTENCES),
        help=f'Path to lemmatized sentences (default: {DEFAULT_SENTENCES})'
    )
    
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Path to FastText .bin model (cc.sv.300.bin)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--aggregation',
        type=str,
        choices=['mean', 'sum'],
        default=AGGREGATION_METHOD,
        help=f'Vector aggregation method (default: {AGGREGATION_METHOD})'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main() -> int:
    """
    Main entry point for command-line execution.
    
    Returns:
    --------
    int
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        run_pipeline(
            expanded_terms_path=args.expanded_terms,
            sentences_path=args.sentences,
            model_path=args.model_path,
            output_dir=args.output_dir,
            aggregation_method=args.aggregation,
            verbose=args.verbose
        )
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure all input files exist:", file=sys.stderr)
        print(f"  - Expanded terms: {args.expanded_terms}", file=sys.stderr)
        print(f"  - Sentences: {args.sentences}", file=sys.stderr)
        print(f"  - Model: {args.model_path}", file=sys.stderr)
        return 1
        
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
        
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        logging.exception("Unexpected error during processing")
        return 1


if __name__ == '__main__':
    sys.exit(main())
