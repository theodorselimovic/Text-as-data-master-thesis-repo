#!/usr/bin/env python3
"""
BoW Preprocessing Script (Stage 2)

Takes the BERT corpus (sentence-segmented, cleaned text) and prepares it for
bag-of-words analysis by adding:
1. Stemming (using NLTK Snowball Swedish stemmer)
2. Stopword removal (Swedish stopwords)
3. Lowercasing
4. N-gram generation (unigrams, bigrams, trigrams)

Stemming vs Lemmatization:
    Stemming is used instead of lemmatization because:
    1. Speed: Rule-based stemming is ~100x faster than neural lemmatization
    2. Consistency: Same input always produces same output (no model variance)
    3. Dictionary matching: When both corpus and dictionary are stemmed identically,
       matching is reliable even if stems are non-words

    N-grams (bigrams, trigrams) capture multi-word dictionary phrases like
    "organiserad brottslighet" → "organiser_brottslig".

Input:
    BERT corpus parquet with columns:
    - doc_id, municipality, year, actor_type, sentence_id, paragraph_id
    - sentence_text (original surface form)

Output:
    Parquet file with additional columns:
    - tokens: List of stemmed tokens including n-grams (e.g., ["klimat", "förändring", "klimat_förändring"])
    - tokens_text: Space-joined tokens (for easy viewing)
    - token_count: Total token count

Usage:
    python preprocessing_bow.py \\
        --input data/processed/bert_corpus.parquet \\
        --output data/processed/bow_corpus_stemmed.parquet

    # Keep stopwords (for certain analyses)
    python preprocessing_bow.py \\
        --input data/processed/bert_corpus.parquet \\
        --output data/processed/bow_corpus_stemmed.parquet \\
        --keep-stopwords

    # Adjust max n-gram size (default: 3 for trigrams)
    python preprocessing_bow.py \\
        --input data/processed/bert_corpus.parquet \\
        --output data/processed/bow_corpus_stemmed.parquet \\
        --max-ngram 2

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-03-17
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams

# =============================================================================
# Configuration
# =============================================================================

# Swedish stopwords (comprehensive list)
# Source: NLTK Swedish stopwords + common additions
SWEDISH_STOPWORDS = {
    # Articles and pronouns
    'en', 'ett', 'den', 'det', 'de', 'denna', 'dette', 'dessa',
    'jag', 'du', 'han', 'hon', 'den', 'det', 'vi', 'ni', 'dem',
    'min', 'din', 'sin', 'vår', 'er', 'deras',
    'mitt', 'ditt', 'sitt', 'vårt', 'ert',
    'mina', 'dina', 'sina', 'våra', 'era',
    'sig', 'man',

    # Conjunctions and prepositions
    'och', 'eller', 'men', 'så', 'som', 'att', 'om', 'när', 'där',
    'här', 'var', 'vart', 'hur', 'varför', 'vad', 'vem', 'vilken', 'vilket', 'vilka',
    'av', 'på', 'i', 'till', 'med', 'för', 'från', 'vid', 'under', 'över',
    'mellan', 'genom', 'efter', 'före', 'utan', 'inom', 'mot', 'hos',
    'ur', 'åt', 'enligt', 'utav', 'omkring', 'bakom', 'framför',

    # Verbs (common auxiliaries and copulas)
    'är', 'var', 'varit', 'vara', 'blir', 'blev', 'blivit', 'bli', 'bliva',
    'ha', 'har', 'hade', 'haft', 'kan', 'kunde', 'kunna', 'kunnat',
    'ska', 'skall', 'skulle', 'vill', 'ville', 'måste', 'må', 'måtte',
    'bör', 'borde', 'få', 'får', 'fick', 'fått', 'göra', 'gör', 'gjorde', 'gjort',

    # Adverbs
    'inte', 'ej', 'icke', 'aldrig', 'alltid', 'också', 'även', 'bara', 'endast',
    'ju', 'nog', 'väl', 'dock', 'redan', 'ännu', 'sedan', 'fortfarande',
    'mycket', 'mer', 'mest', 'lite', 'litet', 'mindre', 'minst',
    'helt', 'helt', 'ganska', 'rätt', 'riktigt', 'verkligen',
    'nu', 'då', 'än', 'snart', 'ofta', 'sällan', 'ibland',
    'upp', 'ned', 'ner', 'ut', 'in', 'bort', 'hem', 'hit', 'dit',

    # Other common words
    'alla', 'allt', 'allting', 'ingen', 'inget', 'ingenting', 'någon', 'något', 'några',
    'varje', 'annan', 'annat', 'andra', 'sådan', 'sådant', 'sådana',
    'samma', 'själv', 'själva', 'egen', 'eget', 'egna',
    'båda', 'bägge', 'vardera', 'samtliga',
    'många', 'få', 'flera', 'flesta', 'enda',

    # Numbers (as words)
    'ett', 'två', 'tre', 'fyra', 'fem', 'sex', 'sju', 'åtta', 'nio', 'tio',
    'första', 'andra', 'tredje',

    # Common filler words in formal Swedish text
    'samt', 'dels', 'dels', 'respektive', 'dvs', 'bl', 'a', 'etc', 'osv',
    'ex', 't', 'ex', 'ca', 's', 'k', 'tex', 'bla',
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# =============================================================================
# Text Processing
# =============================================================================

class BoWPreprocessor:
    """
    Preprocessor for bag-of-words analysis.

    Uses Snowball Swedish stemmer and generates n-grams for multi-word phrase matching.
    """

    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        keep_stopwords: bool = False,
        min_token_length: int = 2,
        max_ngram: int = 3
    ):
        """
        Initialize the preprocessor.

        Parameters:
        -----------
        stopwords : Set[str], optional
            Custom stopword set. Uses default Swedish stopwords if not provided.
        keep_stopwords : bool
            If True, don't remove stopwords (default: False)
        min_token_length : int
            Minimum token length to keep (default: 2)
        max_ngram : int
            Maximum n-gram size (default: 3 for trigrams)
        """
        self.stopwords = stopwords if stopwords is not None else SWEDISH_STOPWORDS
        self.keep_stopwords = keep_stopwords
        self.min_token_length = min_token_length
        self.max_ngram = max_ngram
        self.stemmer = SnowballStemmer('swedish')

        # Pre-stem stopwords for faster lookup
        self.stemmed_stopwords = {self.stemmer.stem(sw) for sw in self.stopwords}

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning before tokenization."""
        if not isinstance(text, str):
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        return text.strip()

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: split on non-alphanumeric characters.

        This is faster than NLTK word_tokenize and sufficient for stemming.
        """
        # Split on anything that's not a letter (including Swedish chars)
        tokens = re.findall(r'[a-zåäöA-ZÅÄÖ]+', text.lower())
        return tokens

    def _is_valid_token(self, token: str) -> bool:
        """Check if token should be kept."""
        # Must meet minimum length
        if len(token) < self.min_token_length:
            return False

        # Must contain at least one letter
        if not re.search(r'[a-zåäö]', token, re.IGNORECASE):
            return False

        return True

    def stem_token(self, token: str) -> str:
        """Stem a single token."""
        return self.stemmer.stem(token.lower())

    def generate_ngrams(self, tokens: List[str], max_n: int = 3) -> List[str]:
        """
        Generate n-grams from stemmed tokens.

        Parameters:
        -----------
        tokens : List[str]
            List of stemmed unigrams
        max_n : int
            Maximum n-gram size

        Returns:
        --------
        List[str] : Combined list of unigrams and joined n-grams
                   e.g., ["klimat", "förändring", "klimat_förändring"]
        """
        if not tokens:
            return []

        all_grams = list(tokens)  # Start with unigrams

        for n in range(2, max_n + 1):
            if len(tokens) >= n:
                n_grams = ['_'.join(gram) for gram in ngrams(tokens, n)]
                all_grams.extend(n_grams)

        return all_grams

    def process_sentence(self, text: str) -> List[str]:
        """
        Process a sentence: tokenize, stem, filter, generate n-grams.

        Parameters:
        -----------
        text : str
            Input sentence text

        Returns:
        --------
        List[str] : List of stemmed tokens including n-grams
        """
        text = self._clean_text(text)
        if not text:
            return []

        # Tokenize
        raw_tokens = self._tokenize(text)

        # Filter and stem
        stemmed_tokens = []
        for token in raw_tokens:
            if not self._is_valid_token(token):
                continue

            stem = self.stem_token(token)

            # Skip stopwords (check both original and stemmed form)
            if not self.keep_stopwords:
                if token in self.stopwords or stem in self.stemmed_stopwords:
                    continue

            stemmed_tokens.append(stem)

        # Generate n-grams
        all_tokens = self.generate_ngrams(stemmed_tokens, self.max_ngram)

        return all_tokens

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'sentence_text',
        batch_size: int = 10000
    ) -> pd.DataFrame:
        """
        Process entire DataFrame, adding stemmed tokens and n-grams.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with text column
        text_column : str
            Name of column containing text
        batch_size : int
            Process in batches for progress reporting

        Returns:
        --------
        pd.DataFrame : DataFrame with 'tokens', 'tokens_text', 'token_count' columns added
        """
        total = len(df)
        logging.info(f"Processing {total:,} rows...")

        tokens_list = []
        start_time = time.time()

        for i, text in enumerate(df[text_column]):
            tokens = self.process_sentence(text)
            tokens_list.append(tokens)

            # Progress reporting
            if (i + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate
                logging.info(
                    f"  Processed {i + 1:,}/{total:,} "
                    f"({(i + 1) / total * 100:.1f}%) - "
                    f"Rate: {rate:.0f} rows/sec - "
                    f"ETA: {remaining / 60:.1f} min"
                )

        # Add new columns
        df = df.copy()
        df['tokens'] = tokens_list
        df['tokens_text'] = df['tokens'].apply(lambda x: ' '.join(x))
        df['token_count'] = df['tokens'].apply(len)

        elapsed = time.time() - start_time
        logging.info(f"Processing complete in {elapsed / 60:.1f} minutes")
        logging.info(f"  Rate: {total / elapsed:.0f} rows/sec")

        return df


# =============================================================================
# Main
# =============================================================================

def load_stopwords(filepath: Path) -> Set[str]:
    """Load stopwords from file (one per line)."""
    stopwords = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word and not word.startswith('#'):
                stopwords.add(word)
    return stopwords


def main():
    parser = argparse.ArgumentParser(
        description="BoW preprocessing: stemming, stopword removal, and n-gram generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input parquet file (BERT corpus)'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output parquet file'
    )

    parser.add_argument(
        '--text-column',
        type=str,
        default='sentence_text',
        help='Column containing text (default: sentence_text)'
    )

    parser.add_argument(
        '--keep-stopwords',
        action='store_true',
        help='Keep stopwords (default: remove them)'
    )

    parser.add_argument(
        '--stopwords-file',
        type=Path,
        help='Custom stopwords file (one word per line)'
    )

    parser.add_argument(
        '--min-token-length',
        type=int,
        default=2,
        help='Minimum token length to keep (default: 2)'
    )

    parser.add_argument(
        '--max-ngram',
        type=int,
        default=3,
        help='Maximum n-gram size (default: 3 for trigrams)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Validate input
    if not args.input.exists():
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Load data
    logging.info("=" * 70)
    logging.info("BOW PREPROCESSING (STEMMING + N-GRAMS)")
    logging.info("=" * 70)
    logging.info(f"Input: {args.input}")
    logging.info(f"Output: {args.output}")
    logging.info(f"Keep stopwords: {args.keep_stopwords}")
    logging.info(f"Min token length: {args.min_token_length}")
    logging.info(f"Max n-gram size: {args.max_ngram}")

    logging.info("\nLoading data...")
    df = pd.read_parquet(args.input)
    logging.info(f"Loaded {len(df):,} rows")
    logging.info(f"Columns: {list(df.columns)}")

    # Load custom stopwords if provided
    stopwords = None
    if args.stopwords_file:
        if args.stopwords_file.exists():
            stopwords = load_stopwords(args.stopwords_file)
            logging.info(f"Loaded {len(stopwords)} custom stopwords")
        else:
            logging.warning(f"Stopwords file not found: {args.stopwords_file}")

    # Process
    logging.info("\n" + "=" * 70)
    logging.info("STEMMING & N-GRAM GENERATION")
    logging.info("=" * 70)

    preprocessor = BoWPreprocessor(
        stopwords=stopwords,
        keep_stopwords=args.keep_stopwords,
        min_token_length=args.min_token_length,
        max_ngram=args.max_ngram
    )

    df = preprocessor.process_dataframe(df, text_column=args.text_column)

    # Summary statistics
    logging.info("\n" + "=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)

    total_tokens = df['token_count'].sum()
    avg_tokens = df['token_count'].mean()
    empty_rows = (df['token_count'] == 0).sum()

    logging.info(f"Total tokens (incl. n-grams): {total_tokens:,}")
    logging.info(f"Average tokens per sentence: {avg_tokens:.1f}")
    logging.info(f"Empty rows (no tokens): {empty_rows:,} ({empty_rows / len(df) * 100:.1f}%)")

    # Count n-grams
    all_tokens = [t for tokens in df['tokens'] for t in tokens]
    unigrams = sum(1 for t in all_tokens if '_' not in t)
    bigrams = sum(1 for t in all_tokens if t.count('_') == 1)
    trigrams = sum(1 for t in all_tokens if t.count('_') == 2)
    logging.info(f"\nToken breakdown:")
    logging.info(f"  Unigrams: {unigrams:,}")
    logging.info(f"  Bigrams:  {bigrams:,}")
    logging.info(f"  Trigrams: {trigrams:,}")

    # Show sample
    logging.info("\nSample output:")
    sample = df[df['token_count'] > 0].head(3)
    for _, row in sample.iterrows():
        logging.info(f"  Original: {row[args.text_column][:80]}...")
        # Show unigrams and some n-grams
        tokens = row['tokens']
        unigrams_sample = [t for t in tokens if '_' not in t][:5]
        ngrams_sample = [t for t in tokens if '_' in t][:3]
        logging.info(f"  Unigrams: {unigrams_sample}")
        if ngrams_sample:
            logging.info(f"  N-grams:  {ngrams_sample}")
        logging.info("")

    # Save output
    logging.info("=" * 70)
    logging.info("SAVING OUTPUT")
    logging.info("=" * 70)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    file_size = args.output.stat().st_size / 1024 / 1024
    logging.info(f"Saved to: {args.output}")
    logging.info(f"File size: {file_size:.1f} MB")
    logging.info(f"Columns: {list(df.columns)}")

    logging.info("\n" + "=" * 70)
    logging.info("DONE")
    logging.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
