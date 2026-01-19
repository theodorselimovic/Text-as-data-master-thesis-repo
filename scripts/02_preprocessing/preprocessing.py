#!/usr/bin/env python3
"""
Text Preprocessing and Lemmatization for RSA Documents

This script processes raw text extracted from Swedish RSA PDFs, performing:
1. Document metadata parsing (municipality, year, redaction status)
2. Sentence segmentation
3. Lemmatization using Stanza
4. Stopword removal
5. Output to structured parquet format

The script is designed to work with R's readtext package output format,
taking RDS files and converting them to lemmatized sentence-level data
ready for downstream analysis.

Input Format (from R readtext or Python PDF extraction):
    File with columns:
    - file: PDF filename (e.g., "RSA Ale 2015 Maskad.pdf")
    - text: Extracted text content
    
    Supported formats:
    - RDS (from R readtext package)
    - CSV (from pdf_reader_enhanced.py or other sources)
    - Parquet (from pdf_reader_enhanced.py or other sources)

Output Format:
    Parquet file with columns:
    - doc_id: Document identifier (filename)
    - municipality: Swedish municipality name
    - year: Publication year (4-digit string or 'unknown')
    - maskad: Whether document is redacted (boolean)
    - sentence_id: Sentence number within document
    - sentence_text: Lemmatized sentence (stopwords removed)
    - word_count: Number of words in lemmatized sentence

Usage:
    # From PDF extraction tool
    python readingtexts.py --input pdf_texts.parquet --output sentences_lemmatized.parquet
    
    # From R readtext
    python readingtexts.py --input readtext_success.rds --output sentences_lemmatized.parquet
    
    # With custom stopwords
    python readingtexts.py --input pdf_texts.parquet --output output.parquet --stopwords custom_stopwords.txt

Requirements:
    - Stanza Swedish language model
    - Python packages: pandas, pyreadr, stanza, nltk

Author: Swedish Risk Analysis Text-as-Data Project
Version: 2.0
Date: 2025-01-04
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import pyreadr
import stanza
from nltk.corpus import stopwords
import nltk

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default file paths
DEFAULT_INPUT_FILE = 'readtext_success.rds'
DEFAULT_OUTPUT_FILE = 'data/processed/sentences_lemmatized.parquet'

# Custom stopwords for RSA documents (OCR artifacts and domain-specific terms)
CUSTOM_STOPWORDS = {
    "samt",           # "as well as" (very common filler)
    "â”‚",              # OCR artifact (vertical bar)
    "_____________________________________________________________________________",  # OCR artifact
    "â–ª",              # OCR artifact (bullet point)
    "underbilaga",    # "appendix" (administrative boilerplate)
    "rsa",            # Document type abbreviation
    "2023-2026"       # Specific year range (too specific)
}

# RSA filename parsing pattern
# Format: RSA [Municipality] [Year] [Maskad].pdf
RSA_FILENAME_PATTERN = re.compile(
    r"^RSA\s+(?P<municipality>.+?)\s+(?P<year>(?:19|20)\d{2})(?:\s+(?P<maskad>[Mm]askad))?\s*\.pdf$",
    re.IGNORECASE,
)

# Fallback year pattern for non-standard filenames
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


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
# STOPWORD LOADER
# =============================================================================

class StopwordManager:
    """
    Manage Swedish stopwords including NLTK and custom domain-specific terms.
    
    This class handles loading and combining stopwords from multiple sources:
    - NLTK Swedish stopwords (standard linguistic stopwords)
    - Custom stopwords (OCR artifacts, domain-specific terms)
    - Optional user-provided stopword file
    """
    
    def __init__(self, custom_stopwords: Optional[Set[str]] = None):
        """
        Initialize stopword manager.
        
        Parameters:
        -----------
        custom_stopwords : Set[str], optional
            Additional custom stopwords to include
        """
        self.stopwords = set()
        self.custom_stopwords = custom_stopwords or CUSTOM_STOPWORDS
        
    def load_nltk_stopwords(self) -> None:
        """
        Load Swedish stopwords from NLTK.
        
        Downloads the stopwords corpus if not already available.
        """
        logging.info("Loading NLTK Swedish stopwords...")
        try:
            swedish_stopwords = set(stopwords.words('swedish'))
            self.stopwords.update(swedish_stopwords)
            logging.info(f"  Loaded {len(swedish_stopwords)} NLTK stopwords")
        except LookupError:
            logging.info("  NLTK stopwords not found, downloading...")
            nltk.download('stopwords', quiet=True)
            swedish_stopwords = set(stopwords.words('swedish'))
            self.stopwords.update(swedish_stopwords)
            logging.info(f"  Loaded {len(swedish_stopwords)} NLTK stopwords")
    
    def add_custom_stopwords(self) -> None:
        """Add custom domain-specific stopwords."""
        logging.info(f"Adding {len(self.custom_stopwords)} custom stopwords")
        self.stopwords.update(self.custom_stopwords)
    
    def load_from_file(self, filepath: Path) -> None:
        """
        Load additional stopwords from a text file.
        
        Parameters:
        -----------
        filepath : Path
            Path to text file with one stopword per line
        """
        if not filepath.exists():
            logging.warning(f"Stopword file not found: {filepath}")
            return
        
        logging.info(f"Loading stopwords from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            file_stopwords = {line.strip().lower() for line in f if line.strip()}
        self.stopwords.update(file_stopwords)
        logging.info(f"  Loaded {len(file_stopwords)} stopwords from file")
    
    def get_stopwords(self) -> Set[str]:
        """
        Get the complete set of stopwords.
        
        Returns:
        --------
        Set[str] : All stopwords combined
        """
        if not self.stopwords:
            self.load_nltk_stopwords()
            self.add_custom_stopwords()
        return self.stopwords


# =============================================================================
# DOCUMENT METADATA PARSER
# =============================================================================

class DocumentMetadataParser:
    """
    Parse metadata from RSA document filenames.
    
    Extracts:
    - Municipality name
    - Publication year
    - Redaction status (maskad)
    """
    
    @staticmethod
    def parse_filename(filename: str) -> Dict[str, any]:
        """
        Parse RSA document filename to extract metadata.
        
        Expected format: RSA [municipality] [year] [Maskad].pdf
        
        Parameters:
        -----------
        filename : str
            PDF filename to parse
            
        Returns:
        --------
        Dict with keys:
            - municipality: str (municipality name or 'unknown')
            - year: str (4-digit year or 'unknown')
            - maskad: bool (redaction status)
            
        Examples:
        ---------
        >>> parse_filename("RSA Ale 2015 Maskad.pdf")
        {'municipality': 'Ale', 'year': '2015', 'maskad': True}
        
        >>> parse_filename("RSA GnosjÃ¶ 2019.pdf")
        {'municipality': 'GnosjÃ¶', 'year': '2019', 'maskad': False}
        """
        # Try structured pattern first
        match = RSA_FILENAME_PATTERN.match(filename)
        if match:
            return {
                'municipality': match.group('municipality').strip(),
                'year': match.group('year'),
                'maskad': match.group('maskad') is not None
            }
        
        # Fallback: extract what we can
        name_without_ext = Path(filename).stem
        is_masked = "maskad" in name_without_ext.lower()
        
        # Remove 'RSA' prefix and 'Maskad' suffix
        clean_name = re.sub(r"^RSA\s*", "", name_without_ext, flags=re.IGNORECASE)
        clean_name = re.sub(r"\s*[Mm]askad\s*$", "", clean_name)
        
        # Extract year
        year_match = YEAR_PATTERN.search(clean_name)
        year = year_match.group(0) if year_match else "unknown"
        
        # Remaining text is municipality
        municipality = YEAR_PATTERN.sub("", clean_name).strip()
        municipality = municipality if municipality else "unknown"
        
        return {
            'municipality': municipality,
            'year': year,
            'maskad': is_masked
        }


# =============================================================================
# STANZA LEMMATIZER
# =============================================================================

class SwedishLemmatizer:
    """
    Swedish text lemmatizer using Stanza.
    
    Handles:
    - Sentence segmentation
    - Part-of-speech tagging
    - Lemmatization
    - Stopword removal
    """
    
    def __init__(self, stopwords: Set[str], remove_stopwords: bool = True):
        """
        Initialize lemmatizer.
        
        Parameters:
        -----------
        stopwords : Set[str]
            Set of stopwords to remove
        remove_stopwords : bool
            Whether to remove stopwords during lemmatization
        """
        self.stopwords = stopwords
        self.remove_stopwords = remove_stopwords
        self.pipeline = None
        
    def load_stanza_pipeline(self) -> None:
        """
        Load Stanza Swedish pipeline.
        
        Includes tokenization, POS tagging, and lemmatization.
        Does NOT include MWT (multi-word tokenization) as it's not available for Swedish.
        """
        logging.info("=" * 80)
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
            logging.info("âœ“ Stanza model loaded successfully\n")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Stanza model: {e}") from e
    
    def lemmatize_sentence(self, sentence) -> tuple[str, int]:
        """
        Lemmatize a single sentence and optionally remove stopwords.
        
        Parameters:
        -----------
        sentence : stanza.models.common.doc.Sentence
            Stanza sentence object
            
        Returns:
        --------
        tuple[str, int] : (lemmatized_text, word_count)
        """
        lemmatized_words = []
        
        for word in sentence.words:
            lemma = word.lemma.lower()
            original = word.text
            
            # Skip punctuation-only tokens
            if re.match(r'^[^\w\s]+$', original):
                continue
            
            # Check if it's a stopword
            if self.remove_stopwords:
                if lemma in self.stopwords or original.lower() in self.stopwords:
                    continue
            
            lemmatized_words.append(lemma)
        
        lemmatized_text = ' '.join(lemmatized_words)
        word_count = len(lemmatized_words)
        
        return lemmatized_text, word_count
    
    def process_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, any]
    ) -> List[Dict]:
        """
        Process a document into lemmatized sentences.
        
        Parameters:
        -----------
        doc_id : str
            Document identifier (filename)
        text : str
            Raw text content
        metadata : Dict
            Document metadata (municipality, year, maskad)
            
        Returns:
        --------
        List[Dict] : List of sentence records with lemmatized text
        """
        if pd.isna(text) or text == '':
            return []
        
        if self.pipeline is None:
            self.load_stanza_pipeline()
        
        # Process text with Stanza
        doc = self.pipeline(text)
        sentences_data = []
        
        for sent_idx, sentence in enumerate(doc.sentences, 1):
            lemmatized_text, word_count = self.lemmatize_sentence(sentence)
            
            # Only keep non-empty sentences
            if lemmatized_text.strip():
                sentence_record = {
                    'doc_id': doc_id,
                    'municipality': metadata['municipality'],
                    'year': metadata['year'],
                    'maskad': metadata['maskad'],
                    'sentence_id': sent_idx,
                    'sentence_text': lemmatized_text,
                    'word_count': word_count
                }
                sentences_data.append(sentence_record)
        
        return sentences_data


# =============================================================================
# DATA PROCESSOR
# =============================================================================

class CorpusProcessor:
    """
    Main processor for converting R readtext output to lemmatized sentences.
    
    Orchestrates:
    1. Loading RDS file
    2. Parsing document metadata
    3. Lemmatizing sentences
    4. Saving to parquet
    """
    
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        stopwords: Set[str],
        remove_stopwords: bool = True
    ):
        """
        Initialize corpus processor.
        
        Parameters:
        -----------
        input_path : Path
            Path to input RDS file
        output_path : Path
            Path for output parquet file
        stopwords : Set[str]
            Set of stopwords to remove
        remove_stopwords : bool
            Whether to remove stopwords
        """
        self.input_path = input_path
        self.output_path = output_path
        self.stopwords = stopwords
        self.remove_stopwords = remove_stopwords
        
        self.df_input = None
        self.df_output = None
        
        self.lemmatizer = SwedishLemmatizer(stopwords, remove_stopwords)
        self.metadata_parser = DocumentMetadataParser()
    
    def load_input_file(self) -> pd.DataFrame:
        """
        Load input file (RDS, CSV, or Parquet).
        
        Supports:
        - RDS files from R readtext
        - CSV files from pdf_reader_enhanced.py
        - Parquet files from pdf_reader_enhanced.py
        
        Returns:
        --------
        pd.DataFrame with columns:
            - file: PDF filename
            - text: Extracted text content
        """
        logging.info("=" * 80)
        logging.info("LOADING INPUT FILE")
        logging.info("=" * 80)
        logging.info(f"Input file: {self.input_path}")
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Determine file type and read accordingly
        suffix = self.input_path.suffix.lower()
        
        if suffix == '.rds':
            logging.info("Format: RDS (from R readtext)")
            # Read RDS file
            rds_data = pyreadr.read_r(str(self.input_path))
            # Get first (and typically only) dataframe from RDS
            df = rds_data[list(rds_data.keys())[0]]
            
        elif suffix == '.csv':
            logging.info("Format: CSV")
            df = pd.read_csv(self.input_path)
            
        elif suffix == '.parquet':
            logging.info("Format: Parquet")
            df = pd.read_parquet(self.input_path)
            
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .rds, .csv, .parquet"
            )
        
        logging.info(f"Loaded {len(df)} documents")
        logging.info(f"Columns: {df.columns.tolist()}\n")
        
        # Validate required columns
        if 'file' not in df.columns or 'text' not in df.columns:
            raise ValueError(
                f"Input file must contain 'file' and 'text' columns. "
                f"Found: {df.columns.tolist()}"
            )
        
        self.df_input = df
        return df
    
    def process_all_documents(self) -> pd.DataFrame:
        """
        Process all documents into sentence-level data.
        
        Returns:
        --------
        pd.DataFrame : Sentence-level data with lemmatization
        """
        logging.info("=" * 80)
        logging.info("PROCESSING DOCUMENTS")
        logging.info("=" * 80)
        logging.info(f"Processing {len(self.df_input)} documents...")
        logging.info("(This may take a while for large corpora)\n")
        
        all_sentences = []
        start_time = time.time()
        
        for idx, row in self.df_input.iterrows():
            # Progress indicator
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(self.df_input) - idx - 1) / rate if rate > 0 else 0
                logging.info(
                    f"Progress: {idx + 1}/{len(self.df_input)} "
                    f"({rate:.1f} docs/sec, ETA: {eta/60:.1f} min)"
                )
            
            doc_id = row['file']
            text = row['text']
            
            # Parse metadata
            metadata = self.metadata_parser.parse_filename(doc_id)
            
            # Process document
            try:
                sentences = self.lemmatizer.process_document(doc_id, text, metadata)
                all_sentences.extend(sentences)
            except Exception as e:
                logging.error(f"Error processing {doc_id}: {e}")
                continue
        
        elapsed = time.time() - start_time
        logging.info(f"\nâœ“ Processing complete in {elapsed:.1f}s")
        logging.info(f"  Average: {len(self.df_input) / elapsed:.2f} docs/sec")
        
        # Create dataframe
        self.df_output = pd.DataFrame(all_sentences)
        return self.df_output
    
    def print_statistics(self) -> None:
        """Print summary statistics about the processed corpus."""
        logging.info("\n" + "=" * 80)
        logging.info("STATISTICS")
        logging.info("=" * 80)
        
        logging.info(f"Original documents: {len(self.df_input)}")
        logging.info(f"Total sentences extracted: {len(self.df_output)}")
        logging.info(
            f"Average sentences per document: "
            f"{len(self.df_output) / len(self.df_input):.1f}"
        )
        logging.info(
            f"Average words per sentence: "
            f"{self.df_output['word_count'].mean():.1f}"
        )
        logging.info(
            f"Median words per sentence: "
            f"{self.df_output['word_count'].median():.0f}"
        )
        logging.info(f"Min words per sentence: {self.df_output['word_count'].min()}")
        logging.info(f"Max words per sentence: {self.df_output['word_count'].max()}")
        
        # Word count distribution
        logging.info("\nWord count distribution:")
        quartiles = self.df_output['word_count'].quantile([0.25, 0.5, 0.75, 0.95])
        logging.info(f"  25th percentile: {quartiles[0.25]:.0f} words")
        logging.info(f"  50th percentile: {quartiles[0.50]:.0f} words")
        logging.info(f"  75th percentile: {quartiles[0.75]:.0f} words")
        logging.info(f"  95th percentile: {quartiles[0.95]:.0f} words")
        
        # Temporal distribution
        if 'year' in self.df_output.columns:
            logging.info("\nSentences by year:")
            year_counts = self.df_output.groupby('year').size().sort_index()
            for year, count in year_counts.items():
                logging.info(f"  {year}: {count:,}")
        
        # Municipality distribution
        if 'municipality' in self.df_output.columns:
            n_muni = self.df_output['municipality'].nunique()
            logging.info(f"\nTotal municipalities: {n_muni}")
            logging.info("\nTop 10 municipalities by sentences:")
            top_muni = (
                self.df_output
                .groupby('municipality')
                .size()
                .sort_values(ascending=False)
                .head(10)
            )
            for muni, count in top_muni.items():
                logging.info(f"  {muni}: {count:,}")
    
    def print_examples(self, n_examples: int = 5) -> None:
        """
        Print example sentences.
        
        Parameters:
        -----------
        n_examples : int
            Number of example sentences to print
        """
        logging.info("\n" + "=" * 80)
        logging.info("EXAMPLE SENTENCES")
        logging.info("=" * 80)
        
        # Get random sample
        sample = self.df_output.sample(n=min(n_examples, len(self.df_output)))
        
        for idx, (_, row) in enumerate(sample.iterrows(), 1):
            logging.info(f"\nExample {idx}:")
            logging.info(f"  Document: {row['doc_id']}")
            logging.info(
                f"  Municipality: {row['municipality']}, "
                f"Year: {row['year']}, "
                f"Maskad: {row['maskad']}"
            )
            logging.info(f"  Sentence {row['sentence_id']} ({row['word_count']} words):")
            text_preview = row['sentence_text'][:200]
            if len(row['sentence_text']) > 200:
                text_preview += "..."
            logging.info(f"  {text_preview}")
    
    def save_output(self) -> None:
        """Save processed sentences to parquet file."""
        logging.info("\n" + "=" * 80)
        logging.info("SAVING OUTPUT")
        logging.info("=" * 80)
        
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        self.df_output.to_parquet(self.output_path, index=False)
        
        logging.info(f"âœ“ Saved to: {self.output_path}")
        logging.info(f"  Rows: {len(self.df_output):,}")
        logging.info(f"  Columns: {list(self.df_output.columns)}")
        logging.info(f"  File size: {self.output_path.stat().st_size / 1024 / 1024:.1f} MB")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    input_path: Path,
    output_path: Path,
    stopword_file: Optional[Path] = None,
    remove_stopwords: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.
    
    Parameters:
    -----------
    input_path : Path
        Path to input RDS file from R readtext
    output_path : Path
        Path for output parquet file
    stopword_file : Path, optional
        Optional file with additional stopwords
    remove_stopwords : bool
        Whether to remove stopwords during lemmatization
    verbose : bool
        Enable verbose logging
        
    Returns:
    --------
    pd.DataFrame : Processed sentence-level data
    """
    setup_logging(verbose)
    
    logging.info("=" * 80)
    logging.info("TEXT PREPROCESSING AND LEMMATIZATION PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Input:  {input_path}")
    logging.info(f"Output: {output_path}")
    logging.info(f"Remove stopwords: {remove_stopwords}")
    logging.info("")
    
    # Step 1: Load stopwords
    stopword_manager = StopwordManager()
    stopword_manager.load_nltk_stopwords()
    stopword_manager.add_custom_stopwords()
    
    if stopword_file:
        stopword_manager.load_from_file(stopword_file)
    
    all_stopwords = stopword_manager.get_stopwords()
    logging.info(f"Total stopwords: {len(all_stopwords)}\n")
    
    # Step 2: Initialize processor
    processor = CorpusProcessor(
        input_path=input_path,
        output_path=output_path,
        stopwords=all_stopwords,
        remove_stopwords=remove_stopwords
    )
    
    # Step 3: Load input data
    processor.load_input_file()
    
    # Step 4: Process all documents
    df_output = processor.process_all_documents()
    
    # Step 5: Print statistics
    processor.print_statistics()
    processor.print_examples()
    
    # Step 6: Save output
    processor.save_output()
    
    # Final summary
    logging.info("\n" + "=" * 80)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 80)
    logging.info("\nNext steps:")
    logging.info("  1. âœ“ Preprocessing complete")
    logging.info("  2. â†’ Run vectoranalysis.py to expand seed terms")
    logging.info("  3. â†’ Run sentencefiltering.py to create filtered dataset")
    logging.info("  4. â†’ Run cooccurrence_analysis.py for statistical tests")
    
    return df_output


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns:
    --------
    argparse.ArgumentParser : Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Preprocess and lemmatize Swedish RSA documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From R readtext RDS file
    python readingtexts.py --input readtext_success.rds
    
    # From PDF extraction (Parquet format)
    python readingtexts.py --input pdf_texts.parquet
    
    # From PDF extraction (CSV format)
    python readingtexts.py --input pdf_texts.csv
    
    # Custom output location
    python readingtexts.py --input pdf_texts.parquet --output data/processed/sentences.parquet
    
    # With custom stopwords
    python readingtexts.py --input pdf_texts.parquet --stopwords custom_stopwords.txt
    
    # Keep stopwords (no removal)
    python readingtexts.py --input pdf_texts.parquet --no-remove-stopwords
    
    # Verbose output
    python readingtexts.py --input pdf_texts.parquet --verbose

Input Format:
    Any file (RDS, CSV, or Parquet) with columns:
    - file: PDF filename (e.g., "RSA Ale 2015 Maskad.pdf")
    - text: Extracted text content
    
    Sources:
    - R's readtext package (RDS format)
    - pdf_reader_enhanced.py (Parquet or CSV format)
    - Any custom extraction tool with the same format

Output Format:
    Parquet file with columns:
    - doc_id: Document identifier
    - municipality: Municipality name
    - year: Publication year
    - maskad: Redaction status (boolean)
    - sentence_id: Sentence number
    - sentence_text: Lemmatized sentence
    - word_count: Number of words

Stopword Removal:
    By default, removes Swedish stopwords from NLTK plus custom stopwords
    for OCR artifacts and domain-specific terms. Use --no-remove-stopwords
    to keep all words.
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        default=Path(DEFAULT_INPUT_FILE),
        help=f'Input file (RDS/CSV/Parquet) with "file" and "text" columns (default: {DEFAULT_INPUT_FILE})'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
        help=f'Output parquet file (default: {DEFAULT_OUTPUT_FILE})'
    )
    
    parser.add_argument(
        '--stopwords',
        type=Path,
        default=None,
        help='Optional file with additional stopwords (one per line)'
    )
    
    parser.add_argument(
        '--no-remove-stopwords',
        action='store_true',
        help='Keep stopwords in lemmatized text'
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
    int : Exit code (0 for success, non-zero for errors)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            stopword_file=args.stopwords,
            remove_stopwords=not args.no_remove_stopwords,
            verbose=args.verbose
        )
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure the input RDS file exists.", file=sys.stderr)
        return 1
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure Stanza Swedish model is installed:", file=sys.stderr)
        print("  python -c \"import stanza; stanza.download('sv')\"", file=sys.stderr)
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
