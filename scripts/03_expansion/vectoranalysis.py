#!/usr/bin/env python3
"""
Vector Analysis: Seed Term Expansion for RSA Documents

This script expands seed terms for 6 categories using FastText Swedish embeddings.
It finds similar words, lemmatizes them, and creates a comprehensive term list for
filtering sentences in the RSA text analysis pipeline.

Categories:
    - Risk: Core risk discourse
    - Accountability: Responsibility and obligations
    - Complexity: Difficulty and local uniqueness
    - Efficiency: Effectiveness and rationalization
    - Equality: Comparability and equivalence
    - Agency: Institutional actors

Output:
    expanded_terms_lemmatized_complete.csv - CSV file with columns:
        - category: Category name
        - word: Original word from FastText
        - lemma: Lemmatized form
        - similarity_score: Cosine similarity to seed terms

Usage:
    python vectoranalysis.py --model-path /path/to/cc.sv.300.bin --output expanded_terms.csv
    
    # With custom parameters
    python vectoranalysis.py --model-path ./model.bin --top-n 100 --output terms.csv

Requirements:
    - FastText Swedish model (cc.sv.300.bin, ~7GB)
    - Stanza Swedish language model
    - Python packages: numpy, pandas, fasttext, stanza

Author: Swedish Risk Analysis Text-as-Data Project
Version: 2.0
Date: 2025-01-02
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import fasttext
import numpy as np
import pandas as pd
import stanza

# =============================================================================
# CONFIGURATION
# =============================================================================

# Seed terms for each category (Version 2.0)
SEED_TERMS = {
    'risk': [
        'risk', 'riskanalys', 'riskbedömning', 'sårbarhet',
        'kritiska', 'beroenden', 'krisberedskap', 'samhällsviktig', 'verksamhet'
    ],
    'accountability': [
        'åtagande', 'ansvar', 'skyldighet', 'förpliktelse', 'ansvarsområde'
    ],
    'complexity': [
        'komplex', 'svår', 'komplicerad', 'utmaning', 'otydlig',
        'annorlunda', 'unik'
    ],
    'efficiency': [
        'effektiv', 'effektivering', 'effektivitet', 'rationell',
        'nyttig', 'ändamålsenlig', 'verkningsfull'
    ],
    'equality': [
        'jämförbar', 'ekvivalent', 'motsvarande', 'likvärdig', 'utbytbar'
    ],
    'agency': [
        'kommun', 'stat', 'länsstyrelse', 'region',
        'näringsliv', 'civilsamhälle', 'förening'
    ]
}

# Default parameters
DEFAULT_TOP_N = 50
DEFAULT_OUTPUT_FILE = 'expanded_terms_lemmatized_complete.csv'

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
# FASTTEXT MODEL LOADER
# =============================================================================

class FastTextModel:
    """
    Wrapper for loading and using FastText models.
    
    This class handles model loading and provides methods for finding
    similar words using cosine similarity in the embedding space.
    """
    
    def __init__(self, model_path: Path):
        """
        Initialize FastText model loader.
        
        Parameters:
        -----------
        model_path : Path
            Path to FastText .bin model file
        """
        self.model_path = model_path
        self.model = None
        self.dimension = None
        
    def load(self) -> None:
        """
        Load FastText model from disk.
        
        Raises:
        -------
        FileNotFoundError
            If model file doesn't exist
        RuntimeError
            If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logging.info("Loading FastText model...")
        logging.info(f"  Path: {self.model_path}")
        logging.info("  This may take 5-10 minutes for a 7GB file...")
        
        start_time = time.time()
        
        try:
            self.model = fasttext.load_model(str(self.model_path))
            self.dimension = self.model.get_dimension()
            
            elapsed = time.time() - start_time
            logging.info(f"✓ Model loaded successfully in {elapsed:.1f}s")
            logging.info(f"  Vector dimension: {self.dimension}")
            logging.info(f"  Has subword information: Yes")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def get_similar_words(self, word: str, top_n: int = 50) -> Dict[str, float]:
        """
        Find words similar to a given word.
        
        Parameters:
        -----------
        word : str
            Seed word to find neighbors for
        top_n : int
            Number of similar words to return
            
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping similar words to similarity scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # FastText returns: [(similarity, word), ...]
            neighbors = self.model.get_nearest_neighbors(word, k=top_n)
            return {word: similarity for similarity, word in neighbors}
            
        except Exception as e:
            logging.warning(f"Failed to get neighbors for '{word}': {e}")
            return {}
    
    def free_memory(self) -> None:
        """
        Delete model and free memory.
        
        Call this when done with the model to free ~7GB of RAM.
        """
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            logging.info("✓ Model deleted, memory freed")


# =============================================================================
# TERM EXPANDER
# =============================================================================

class TermExpander:
    """
    Expand seed terms using FastText embeddings.
    
    For each seed term in a category, this class finds the most similar
    words in the embedding space and combines them into an expanded
    term list for that category.
    """
    
    def __init__(self, model: FastTextModel):
        """
        Initialize term expander.
        
        Parameters:
        -----------
        model : FastTextModel
            Loaded FastText model
        """
        self.model = model
        self.expanded_terms = {}
        
    def expand_category(
        self,
        category: str,
        seed_terms: List[str],
        top_n: int = 50
    ) -> Dict[str, float]:
        """
        Expand seed terms for a single category.
        
        Parameters:
        -----------
        category : str
            Category name
        seed_terms : List[str]
            List of seed words for this category
        top_n : int
            Number of similar words to find per seed term
            
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping expanded words to their maximum similarity score
        """
        logging.info("=" * 70)
        logging.info(f"Expanding category: {category.upper()}")
        logging.info(f"Seed terms: {', '.join(seed_terms)}")
        logging.info("=" * 70)
        
        similar_words = {}
        
        for seed_term in seed_terms:
            logging.info(f"  Processing '{seed_term}'...")
            
            neighbors = self.model.get_similar_words(seed_term, top_n)
            
            # Combine neighbors, keeping maximum similarity if word appears multiple times
            for word, similarity in neighbors.items():
                if word in similar_words:
                    similar_words[word] = max(similar_words[word], similarity)
                else:
                    similar_words[word] = similarity
            
            logging.info(f"    Found {len(neighbors)} similar words")
        
        # Show top 15 terms
        sorted_terms = sorted(similar_words.items(), key=lambda x: x[1], reverse=True)[:15]
        logging.info(f"\n  Top 15 expanded terms for {category}:")
        for word, score in sorted_terms:
            logging.info(f"    {word:30s} {score:.4f}")
        
        logging.info("")
        
        self.expanded_terms[category] = similar_words
        return similar_words
    
    def expand_all(self, seed_terms_dict: Dict[str, List[str]], top_n: int = 50) -> None:
        """
        Expand seed terms for all categories.
        
        Parameters:
        -----------
        seed_terms_dict : Dict[str, List[str]]
            Dictionary mapping category names to lists of seed terms
        top_n : int
            Number of similar words to find per seed term
        """
        logging.info("\n" + "=" * 80)
        logging.info("EXPANDING ALL CATEGORIES")
        logging.info("=" * 80)
        
        for category, seeds in seed_terms_dict.items():
            self.expand_category(category, seeds, top_n)
        
        logging.info("=" * 80)
        logging.info("EXPANSION COMPLETE")
        logging.info("=" * 80)
        
        # Print summary
        total_terms = sum(len(terms) for terms in self.expanded_terms.values())
        logging.info(f"\nTotal expanded terms: {total_terms}")
        logging.info("\nTerms by category:")
        for category, terms in self.expanded_terms.items():
            logging.info(f"  {category:15s}: {len(terms):4d} terms")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert expanded terms to a pandas DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: category, word, similarity_score
        """
        rows = []
        for category, terms in self.expanded_terms.items():
            for word, similarity in terms.items():
                rows.append({
                    'category': category,
                    'word': word,
                    'similarity_score': similarity
                })
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['category', 'similarity_score'], ascending=[True, False])
        return df


# =============================================================================
# LEMMATIZER
# =============================================================================

class SwedishLemmatizer:
    """
    Lemmatize Swedish words using Stanza.
    
    This class handles initialization of the Stanza Swedish pipeline
    and provides methods for lemmatizing individual words and DataFrames.
    """
    
    def __init__(self):
        """
        Initialize Swedish lemmatizer.
        
        The Stanza pipeline is loaded lazily (on first use) to avoid
        loading it if not needed.
        """
        self.pipeline = None
        
    def load(self) -> None:
        """
        Load Stanza Swedish pipeline.
        
        The pipeline includes tokenization, POS tagging, and lemmatization.
        Note: Does NOT include MWT (multi-word tokenization) as it's not
        available for Swedish.
        """
        logging.info("\n" + "=" * 80)
        logging.info("LOADING STANZA SWEDISH MODEL")
        logging.info("=" * 80)
        logging.info("This may take a minute...")
        
        try:
            self.pipeline = stanza.Pipeline(
                'sv',
                processors='tokenize,pos,lemma',
                use_gpu=False,
                verbose=False
            )
            logging.info("✓ Stanza model loaded successfully\n")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Stanza model: {e}") from e
    
    def lemmatize_word(self, word: str) -> str:
        """
        Lemmatize a single word.
        
        Parameters:
        -----------
        word : str
            Word to lemmatize
            
        Returns:
        --------
        str
            Lemmatized word (lowercase)
        """
        if self.pipeline is None:
            self.load()
        
        try:
            doc = self.pipeline(word)
            if doc.sentences and doc.sentences[0].words:
                return doc.sentences[0].words[0].lemma.lower()
            return word.lower()
            
        except Exception:
            # If lemmatization fails, return lowercase word
            return word.lower()
    
    def lemmatize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lemmatized column to DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'word' column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added 'lemma' column
        """
        if self.pipeline is None:
            self.load()
        
        logging.info("=" * 80)
        logging.info("LEMMATIZING EXPANDED TERMS")
        logging.info("=" * 80)
        logging.info(f"Lemmatizing {len(df)} terms...")
        logging.info("This will take a few minutes...\n")
        
        start_time = time.time()
        
        # Lemmatize all words
        df['lemma'] = df['word'].apply(self.lemmatize_word)
        
        elapsed = time.time() - start_time
        logging.info(f"✓ Lemmatization complete in {elapsed:.1f}s")
        
        # Remove duplicates (same lemma within same category)
        before = len(df)
        df = df.drop_duplicates(subset=['category', 'lemma'])
        after = len(df)
        
        logging.info(f"\nDeduplication:")
        logging.info(f"  Before: {before:4d} terms")
        logging.info(f"  After:  {after:4d} unique lemmas")
        logging.info(f"  Removed: {before - after:4d} duplicates")
        
        logging.info("\nUnique lemmas by category:")
        for category, count in df.groupby('category')['lemma'].nunique().items():
            logging.info(f"  {category:15s}: {count:4d} lemmas")
        
        return df


# =============================================================================
# OUTPUT WRITER
# =============================================================================

class OutputWriter:
    """
    Write expanded terms to disk.
    
    This class handles saving the expanded and lemmatized terms to CSV
    format for use in downstream analysis.
    """
    
    @staticmethod
    def save_csv(df: pd.DataFrame, output_path: Path) -> None:
        """
        Save DataFrame to CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with expanded terms
        output_path : Path
            Output file path
        """
        logging.info("\n" + "=" * 80)
        logging.info("SAVING RESULTS")
        logging.info("=" * 80)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logging.info(f"✓ Saved expanded terms to: {output_path}")
        logging.info(f"  Total terms: {len(df)}")
        logging.info(f"  Categories: {df['category'].nunique()}")
        
    @staticmethod
    def print_summary(df: pd.DataFrame, top_n: int = 20) -> None:
        """
        Print summary of expanded terms.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with expanded terms
        top_n : int
            Number of top terms to display per category
        """
        logging.info("\n" + "=" * 80)
        logging.info("FINAL RESULTS SUMMARY")
        logging.info("=" * 80)
        
        for category in sorted(df['category'].unique()):
            cat_data = df[df['category'] == category]
            logging.info(f"\n{category.upper()}: {len(cat_data)} unique lemmas")
            logging.info("─" * 70)
            logging.info(f"Top {top_n} terms by similarity:")
            
            top_terms = cat_data.nlargest(top_n, 'similarity_score')
            for i, (_, row) in enumerate(top_terms.iterrows(), 1):
                logging.info(
                    f"  {i:2d}. {row['lemma']:25s} "
                    f"(from: {row['word']:25s}) "
                    f"{row['similarity_score']:.4f}"
                )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    model_path: Path,
    output_path: Path,
    seed_terms: Dict[str, List[str]] = None,
    top_n: int = DEFAULT_TOP_N,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run the complete term expansion pipeline.
    
    This function orchestrates the entire process:
    1. Load FastText model
    2. Expand seed terms
    3. Lemmatize expanded terms
    4. Save results
    
    Parameters:
    -----------
    model_path : Path
        Path to FastText .bin model
    output_path : Path
        Path for output CSV file
    seed_terms : Dict[str, List[str]], optional
        Custom seed terms (uses defaults if None)
    top_n : int
        Number of similar words per seed term
    verbose : bool
        Enable verbose logging
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with expanded and lemmatized terms
    """
    setup_logging(verbose)
    
    if seed_terms is None:
        seed_terms = SEED_TERMS
    
    logging.info("=" * 80)
    logging.info("VECTOR ANALYSIS: SEED TERM EXPANSION")
    logging.info("=" * 80)
    logging.info(f"Model path: {model_path}")
    logging.info(f"Output path: {output_path}")
    logging.info(f"Top-N per seed: {top_n}")
    logging.info(f"Categories: {len(seed_terms)}")
    logging.info(f"Total seed terms: {sum(len(v) for v in seed_terms.values())}")
    
    # Step 1: Load FastText model
    model = FastTextModel(model_path)
    model.load()
    
    # Step 2: Expand seed terms
    expander = TermExpander(model)
    expander.expand_all(seed_terms, top_n)
    df_expanded = expander.to_dataframe()
    
    # Free model memory (no longer needed)
    model.free_memory()
    
    # Step 3: Lemmatize
    lemmatizer = SwedishLemmatizer()
    df_final = lemmatizer.lemmatize_dataframe(df_expanded)
    
    # Step 4: Save results
    OutputWriter.save_csv(df_final, output_path)
    OutputWriter.print_summary(df_final, top_n=20)
    
    # Next steps
    logging.info("\n" + "=" * 80)
    logging.info("NEXT STEPS")
    logging.info("=" * 80)
    logging.info("\n1. ✓ Term expansion complete")
    logging.info("\n2. → Run sentencefiltering.py with this file:")
    logging.info(f"     Input: {output_path}")
    logging.info("     Output: sentence_vectors_with_metadata.parquet")
    logging.info("\n3. → Run cooccurrence_analysis.py for chi-square tests")
    logging.info("\n4. → Run correspondence analysis for 2D mapping")
    
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
        description='Expand seed terms using FastText embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python vectoranalysis.py --model-path ./cc.sv.300.bin
    
    # Custom output file
    python vectoranalysis.py --model-path ./model.bin --output my_terms.csv
    
    # Find more similar words per seed term
    python vectoranalysis.py --model-path ./model.bin --top-n 100
    
    # Verbose output
    python vectoranalysis.py --model-path ./model.bin --verbose

Output:
    CSV file with columns:
        - category: Category name (risk, accountability, etc.)
        - word: Original word from FastText
        - lemma: Lemmatized form
        - similarity_score: Cosine similarity to seed terms (0-1)

Seed Terms (Version 2.0):
    Risk: 9 terms (risk, riskanalys, ...)
    Accountability: 5 terms (åtagande, ansvar, ...)
    Complexity: 7 terms (komplex, svår, ...)
    Efficiency: 7 terms (effektiv, effektivering, ...)
    Equality: 5 terms (jämförbar, ekvivalent, ...)
    Agency: 7 terms (kommun, stat, ...)
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Path to FastText .bin model file (cc.sv.300.bin)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
        help=f'Output CSV file (default: {DEFAULT_OUTPUT_FILE})'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=DEFAULT_TOP_N,
        help=f'Number of similar words per seed term (default: {DEFAULT_TOP_N})'
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
            model_path=args.model_path,
            output_path=args.output,
            top_n=args.top_n,
            verbose=args.verbose
        )
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure the FastText model exists at the specified path.")
        return 1
        
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
        
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        logging.exception("Unexpected error during processing")
        return 1


if __name__ == '__main__':
    sys.exit(main())
