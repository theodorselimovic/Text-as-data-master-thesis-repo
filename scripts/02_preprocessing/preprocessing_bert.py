#!/usr/bin/env python3
"""
Light Preprocessing for BERT Corpus

Produces a clean, sentence-segmented corpus suitable for BERT fine-tuning.
Does NOT lemmatize, remove stopwords, or lowercase. Focuses on:
1. OCR artifact cleanup (mojibake repair, line artifact removal)
2. Introductory chapter removal for municipal RSAs (Chapters 1-2)
3. Sentence segmentation (Stanza tokenize-only, no POS/lemma)
4. Data quality assessment with per-document scoring

This is "Stage 1" of a two-stage preprocessing pipeline. The output
can be consumed directly by the BERT fine-tuning pipeline, or passed
to a future "Stage 2" BOW preprocessing script that adds lemmatization
and stopword removal.

Input Format:
    Parquet/CSV file with columns:
    - file: PDF filename (e.g., "RSA Ale 2015 Maskad.pdf")
    - text: Extracted text content
    - actor: (optional) Actor type (kommun/lansstyrelse/MCF)

Output Format:
    Parquet file with columns:
    - doc_id: Document identifier (filename)
    - municipality: Swedish municipality name
    - year: Publication year (4-digit string or 'unknown')
    - maskad: Whether document is redacted (boolean)
    - actor_type: Actor type (kommun/lansstyrelse/MCF/unknown)
    - sentence_id: Sentence number within document (1-indexed)
    - sentence_text: Cleaned sentence (original surface form)
    - word_count: Number of words in sentence
    - doc_quality: Per-document quality score (0.0-1.0)

    JSON quality report with per-document metrics.

Usage:
    # Basic usage
    python preprocessing_bert.py \\
        --input data/merged/pdf_texts_all_actors.parquet \\
        --output data/processed/bert_corpus.parquet

    # Without intro chapter removal
    python preprocessing_bert.py \\
        --input data/merged/pdf_texts_all_actors.parquet \\
        --output data/processed/bert_corpus.parquet \\
        --no-remove-intro-chapters

    # With quality filtering
    python preprocessing_bert.py \\
        --input data/merged/pdf_texts_all_actors.parquet \\
        --output data/processed/bert_corpus.parquet \\
        --min-quality-score 0.3

    # Verbose output
    python preprocessing_bert.py \\
        --input data/merged/pdf_texts_all_actors.parquet \\
        --output data/processed/bert_corpus.parquet \\
        --verbose

Requirements:
    - Stanza Swedish language model
    - Python packages: pandas, stanza, pyarrow

Author: Swedish Risk Analysis Text-as-Data Project
Version: 1.0
Date: 2025-02-10
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import stanza


# =============================================================================
# CONFIGURATION
# =============================================================================

# Mojibake mapping: UTF-8 bytes mis-decoded as Latin-1 (Windows-1252)
MOJIBAKE_MAP = {
    # Smart quotes and dashes
    'â\x80\x99': '\u2019',     # right single quote
    'â\x80\x9c': '\u201c',     # left double quote
    'â\x80\x9d': '\u201d',     # right double quote
    'â\x80\x93': '\u2013',     # en dash
    'â\x80\x94': '\u2014',     # em dash
    'â\x80\xa2': '\u2022',     # bullet
    'â\x80\xa6': '\u2026',     # ellipsis
    'â\x80\x98': '\u2018',     # left single quote
    # Swedish characters
    'Ã¥': 'å', 'Ã\x85': 'Å',
    'Ã¤': 'ä', 'Ã\x84': 'Ä',
    'Ã¶': 'ö', 'Ã\x96': 'Ö',
    'Ã©': 'é', 'Ã¼': 'ü',
    # Garbage artifacts (remove entirely)
    'â\x94\x82': '',           # box-drawing vertical bar
    'â\x96ª': '',              # black small square
}

# String-form mojibake patterns (for text that's already string, not raw bytes)
MOJIBAKE_MAP_STR = {
    'â€™': '\u2019',
    'â€œ': '\u201c',
    '\u00e2\u0080\u009d': '\u201d',
    'â€"': '\u2013',
    'â€"': '\u2014',
    'â€¢': '\u2022',
    'â€¦': '\u2026',
    'â€˜': '\u2018',
    'Ã¥': 'å', 'Ã…': 'Å',
    'Ã¤': 'ä', 'Ã„': 'Ä',
    'Ã¶': 'ö', 'Ã–': 'Ö',
    'Ã©': 'é', 'Ã¼': 'ü',
    'â"‚': '',
    'â–ª': '',
}

# Line-level artifact patterns
ARTIFACT_PATTERNS = [
    re.compile(r'^\s*\d{1,3}\s*$', re.MULTILINE),          # Page numbers
    re.compile(r'^_{5,}$', re.MULTILINE),                    # Underscore lines
    re.compile(r'^-{5,}$', re.MULTILINE),                    # Dash lines
    re.compile(r'^={5,}$', re.MULTILINE),                    # Equals lines
    re.compile(r'^\s*[\u2500-\u257F]+\s*$', re.MULTILINE),  # Box-drawing lines
    re.compile(r'^\s*\.{5,}\s*$', re.MULTILINE),             # Dot leader lines
    # Page headers/footers: "Risk- och sårbarhetsanalys 2023-2025" or
    # "Risk-och sårbarhetsanalys Sida X (Y)" or "X (Y)" page indicators
    re.compile(
        r'^\s*[Rr]isk-?\s*och\s+sårbarhetsanalys\s+.*$', re.MULTILINE
    ),
    re.compile(r'^\s*\d{1,3}\s*\(\d{1,3}\)\s*$', re.MULTILINE),  # "X (Y)"
    re.compile(r'^\s*Sida?\s+\d{1,3}\s*\(\d{1,3}\)\s*$', re.MULTILINE),
]

# Chapter 3 detection regex — precise, matching the MSB-mandated heading
# "Identifierad samhällsviktig verksamhet inom kommunens geografiska område"
# Some municipalities insert extra text (e.g. "och dess kritiska beroenden")
# between "verksamhet" and "inom", so we allow optional intervening words.
CHAPTER_3_PATTERN = re.compile(
    r'(?:^|\n)\s*3[\.\s]\s*'
    r'[Ii]dentifierade?\s+'
    r'samhällsviktig[at]?\s+'
    r'verksamhet(?:er)?'
    r'(?:\s+\S+){0,6}\s+'    # allow up to 6 intervening words
    r'inom\s+kommunens\s+geografiska\s+område',
    re.IGNORECASE,
)

# RSA filename parsing pattern
RSA_FILENAME_PATTERN = re.compile(
    r"^RSA\s+(?P<municipality>.+?)\s+(?P<year>(?:19|20)\d{2})"
    r"(?:\s+(?P<maskad>[Mm]askad))?\s*\.pdf$",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

# Quality thresholds
MIN_SENTENCE_WORDS = 3
MAX_SENTENCE_WORDS = 300
MIN_ALPHA_RATIO = 0.5
SHORT_SENTENCE_THRESHOLD = 5
QUALITY_FLAG_OK = 0.6
QUALITY_FLAG_LOW = 0.3


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


# =============================================================================
# DOCUMENT METADATA PARSER
# =============================================================================

class DocumentMetadataParser:
    """Parse metadata from RSA document filenames.

    Extracts municipality name, publication year, and redaction status.
    """

    @staticmethod
    def parse_filename(filename: str) -> Dict[str, any]:
        """Parse RSA document filename to extract metadata.

        Expected format: RSA [municipality] [year] [Maskad].pdf

        Returns dict with keys: municipality, year, maskad.
        """
        match = RSA_FILENAME_PATTERN.match(filename)
        if match:
            return {
                'municipality': match.group('municipality').strip(),
                'year': match.group('year'),
                'maskad': match.group('maskad') is not None,
            }

        # Fallback extraction
        name_without_ext = Path(filename).stem
        is_masked = "maskad" in name_without_ext.lower()

        clean_name = re.sub(r"^RSA\s*", "", name_without_ext, flags=re.IGNORECASE)
        clean_name = re.sub(r"\s*[Mm]askad\s*$", "", clean_name)

        year_match = YEAR_PATTERN.search(clean_name)
        year = year_match.group(0) if year_match else "unknown"

        municipality = YEAR_PATTERN.sub("", clean_name).strip()
        municipality = municipality if municipality else "unknown"

        return {
            'municipality': municipality,
            'year': year,
            'maskad': is_masked,
        }


# =============================================================================
# TEXT CLEANING
# =============================================================================

class TextCleaner:
    """Clean raw OCR/PDF-extracted text.

    Handles mojibake repair, OCR artifact removal, and whitespace
    normalisation. Does NOT remove stopwords, lemmatize, or change case.
    """

    def __init__(self):
        self.stats = {
            'mojibake_replacements': 0,
            'artifact_lines_removed': 0,
            'encoding_rescues': 0,
        }

    def fix_mojibake(self, text: str) -> str:
        """Replace known mojibake sequences with correct Unicode."""
        for bad, good in MOJIBAKE_MAP_STR.items():
            if bad in text:
                count = text.count(bad)
                self.stats['mojibake_replacements'] += count
                text = text.replace(bad, good)

        # Fallback: attempt encode/decode rescue for remaining â-sequences
        if 'â' in text or 'Ã' in text:
            try:
                rescued = text.encode('latin-1').decode('utf-8')
                self.stats['encoding_rescues'] += 1
                text = rescued
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass  # Not Latin-1 encoded; leave as-is

        return text

    def remove_ocr_artifacts(self, text: str) -> str:
        """Remove non-text line artifacts from OCR output."""
        for pattern in ARTIFACT_PATTERNS:
            matches = pattern.findall(text)
            self.stats['artifact_lines_removed'] += len(matches)
            text = pattern.sub('', text)
        return text

    def normalise_whitespace(self, text: str) -> str:
        """Normalise whitespace without destroying paragraph structure.

        Does NOT attempt hyphen-rejoin — Swedish compound constructions
        (e.g. "risk- och") are too easily corrupted.
        """
        # Collapse multiple spaces (not newlines) to single space
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Strip trailing whitespace per line
        text = re.sub(r' +\n', '\n', text)
        # Collapse 3+ consecutive newlines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def clean_document(self, text: str) -> str:
        """Apply all cleaning steps in sequence."""
        if not text:
            return ''
        text = self.fix_mojibake(text)
        text = self.remove_ocr_artifacts(text)
        text = self.normalise_whitespace(text)
        return text

    def get_stats(self) -> Dict[str, int]:
        """Return cleaning statistics."""
        return dict(self.stats)


# =============================================================================
# CHAPTER REMOVAL
# =============================================================================

class ChapterRemover:
    """Remove introductory chapters from municipal RSA documents.

    Detects the Chapter 3 heading ("Identifierad samhällsviktig verksamhet
    inom kommunens geografiska område") and discards everything before it.
    Only applied to kommun actor documents.
    """

    def __init__(self):
        self.pattern = CHAPTER_3_PATTERN
        self.stats = {
            'documents_trimmed': 0,
            'total_chars_removed': 0,
            'detection_failed': 0,
            'safety_guard_triggered': 0,
        }

    def find_chapter3_start(self, text: str) -> Optional[int]:
        """Find the character position where Chapter 3 begins.

        Returns the position of the start of the matched heading line,
        or None if no match is found.
        """
        match = self.pattern.search(text)
        if match is None:
            return None

        # Back up to the start of the line containing the match
        line_start = text.rfind('\n', 0, match.start())
        return line_start + 1 if line_start >= 0 else 0

    def remove_intro_chapters(self, text: str, doc_id: str = '') -> str:
        """Remove introductory chapters before Chapter 3.

        Returns text starting from Chapter 3 heading. If Chapter 3 is
        not found or safety guards trigger, returns original text with
        a logged warning.
        """
        pos = self.find_chapter3_start(text)

        if pos is None:
            self.stats['detection_failed'] += 1
            logging.warning(
                f"  Chapter 3 not detected in {doc_id}; keeping full text"
            )
            return text

        # Safety guard: Chapter 3 should be in the first half of the document
        if pos > len(text) * 0.5:
            self.stats['safety_guard_triggered'] += 1
            logging.warning(
                f"  Chapter 3 found at position {pos}/{len(text)} "
                f"(>{50}%) in {doc_id}; skipping trim"
            )
            return text

        removed_chars = pos
        self.stats['documents_trimmed'] += 1
        self.stats['total_chars_removed'] += removed_chars
        logging.debug(
            f"  Trimmed {removed_chars} chars "
            f"({removed_chars / len(text) * 100:.1f}%) from {doc_id}"
        )
        return text[pos:]

    def get_stats(self) -> Dict[str, int]:
        """Return chapter removal statistics."""
        return dict(self.stats)


# =============================================================================
# DATA QUALITY
# =============================================================================

@dataclass
class DocumentQuality:
    """Quality metrics for a single document."""

    doc_id: str
    total_chars_raw: int
    total_chars_cleaned: int
    chars_retained_pct: float
    total_sentences: int
    sentences_after_filtering: int
    mean_sentence_length: float
    short_sentence_pct: float
    mojibake_density: float
    non_swedish_char_ratio: float
    quality_score: float
    quality_flag: str
    chapter_trimmed: bool
    chapter_trim_failed: bool


class QualityAssessor:
    """Assess text quality at document and sentence level.

    Computes per-document quality scores and filters artifact sentences.
    Does NOT remove documents — only flags them.
    """

    # Characters considered "Swedish text" for quality assessment
    SWEDISH_CHARS: Set[str] = set(
        'abcdefghijklmnopqrstuvwxyzåäö'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ'
        '0123456789 .,;:!?()-/&%\n\t"\''
    )

    def __init__(self):
        self.document_qualities: List[DocumentQuality] = []

    def compute_mojibake_density(self, raw_text: str) -> float:
        """Count mojibake sequences per 1000 characters in raw text."""
        if not raw_text:
            return 0.0
        count = 0
        for pattern in MOJIBAKE_MAP_STR:
            count += raw_text.count(pattern)
        return (count / len(raw_text)) * 1000

    def compute_non_swedish_ratio(self, text: str) -> float:
        """Ratio of characters not in Swedish alphabet + common punctuation."""
        if not text:
            return 0.0
        non_swedish = sum(1 for c in text if c not in self.SWEDISH_CHARS)
        return non_swedish / len(text)

    def filter_sentences(
        self, sentences: List[Dict]
    ) -> List[Dict]:
        """Filter out artifact sentences based on structural quality.

        Removes sentences that are too short, too long, or mostly
        non-alphabetic (table fragments, OCR noise).
        """
        filtered = []
        for s in sentences:
            word_count = s['word_count']

            if word_count < MIN_SENTENCE_WORDS:
                continue
            if word_count > MAX_SENTENCE_WORDS:
                continue

            text = s['sentence_text']
            if text:
                alpha_count = sum(c.isalpha() for c in text)
                alpha_ratio = alpha_count / len(text)
                if alpha_ratio < MIN_ALPHA_RATIO:
                    continue

            filtered.append(s)
        return filtered

    def assess_document(
        self,
        doc_id: str,
        raw_text: str,
        cleaned_text: str,
        sentences_before: List[Dict],
        sentences_after: List[Dict],
        chapter_trimmed: bool,
        chapter_trim_failed: bool,
    ) -> DocumentQuality:
        """Compute quality metrics for a document."""
        total_chars_raw = len(raw_text) if raw_text else 0
        total_chars_cleaned = len(cleaned_text) if cleaned_text else 0
        chars_retained = (
            total_chars_cleaned / total_chars_raw
            if total_chars_raw > 0
            else 0.0
        )

        total_sents = len(sentences_before)
        filtered_sents = len(sentences_after)

        # Sentence length stats
        word_counts = [s['word_count'] for s in sentences_after]
        mean_len = sum(word_counts) / len(word_counts) if word_counts else 0.0
        short_pct = (
            sum(1 for w in word_counts if w < SHORT_SENTENCE_THRESHOLD)
            / len(word_counts)
            if word_counts
            else 0.0
        )

        mojibake_dens = self.compute_mojibake_density(raw_text)
        non_swedish = self.compute_non_swedish_ratio(cleaned_text)

        # Composite quality score (0.0–1.0)
        score = (
            0.4 * min(chars_retained, 1.0)
            + 0.2 * (1.0 - min(short_pct, 1.0))
            + 0.2 * max(1.0 - mojibake_dens / 10.0, 0.0)
            + 0.2 * (1.0 - min(non_swedish, 1.0))
        )
        score = max(0.0, min(1.0, score))

        if score >= QUALITY_FLAG_OK:
            flag = "ok"
        elif score >= QUALITY_FLAG_LOW:
            flag = "low_quality"
        else:
            flag = "very_low_quality"

        quality = DocumentQuality(
            doc_id=doc_id,
            total_chars_raw=total_chars_raw,
            total_chars_cleaned=total_chars_cleaned,
            chars_retained_pct=round(chars_retained, 4),
            total_sentences=total_sents,
            sentences_after_filtering=filtered_sents,
            mean_sentence_length=round(mean_len, 1),
            short_sentence_pct=round(short_pct, 4),
            mojibake_density=round(mojibake_dens, 4),
            non_swedish_char_ratio=round(non_swedish, 4),
            quality_score=round(score, 4),
            quality_flag=flag,
            chapter_trimmed=chapter_trimmed,
            chapter_trim_failed=chapter_trim_failed,
        )
        self.document_qualities.append(quality)
        return quality

    def write_quality_report(self, output_path: Path) -> None:
        """Write quality report as JSON."""
        qualities = self.document_qualities

        # Corpus summary
        total_docs = len(qualities)
        total_sents = sum(q.sentences_after_filtering for q in qualities)
        mean_score = (
            sum(q.quality_score for q in qualities) / total_docs
            if total_docs > 0
            else 0.0
        )
        low_quality = sum(
            1 for q in qualities if q.quality_flag != "ok"
        )
        trimmed = sum(1 for q in qualities if q.chapter_trimmed)
        trim_failed = sum(1 for q in qualities if q.chapter_trim_failed)

        report = {
            "corpus_summary": {
                "total_documents": total_docs,
                "total_sentences": total_sents,
                "mean_quality_score": round(mean_score, 4),
                "low_quality_documents": low_quality,
                "documents_with_chapter_trimming": trimmed,
                "chapter_trimming_failures": trim_failed,
            },
            "documents": [asdict(q) for q in qualities],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logging.info(f"Quality report saved to: {output_path}")


# =============================================================================
# SENTENCE SEGMENTER
# =============================================================================

class SentenceSegmenter:
    """Sentence segmentation using Stanza tokenizer.

    Uses only the 'tokenize' processor (no POS tagging, no lemmatization)
    for speed. Preserves original surface form.
    """

    def __init__(self):
        self.pipeline = None

    def load_pipeline(self) -> None:
        """Load Stanza Swedish pipeline with tokenize only."""
        logging.info("Loading Stanza Swedish tokenizer (no lemmatization)...")
        self.pipeline = stanza.Pipeline(
            'sv',
            processors='tokenize',
            use_gpu=False,
            verbose=False,
        )
        logging.info("Stanza tokenizer loaded.")

    def segment_document(self, text: str) -> List[Dict]:
        """Segment text into sentences.

        Returns list of dicts with sentence_id, sentence_text, word_count.
        """
        if self.pipeline is None:
            self.load_pipeline()

        if not text or not text.strip():
            return []

        doc = self.pipeline(text)
        sentences = []
        for idx, sent in enumerate(doc.sentences, 1):
            sent_text = sent.text
            word_count = len(sent.words)
            sentences.append({
                'sentence_id': idx,
                'sentence_text': sent_text,
                'word_count': word_count,
            })
        return sentences


# =============================================================================
# CORPUS PROCESSOR
# =============================================================================

class BertCorpusProcessor:
    """Main orchestrator for BERT corpus preprocessing.

    Pipeline: load → clean → trim chapters → segment → assess → save.
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        remove_intro_chapters: bool = True,
        quality_report_path: Optional[Path] = None,
        min_quality_score: float = 0.0,
        verbose: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.remove_intro_chapters = remove_intro_chapters
        self.quality_report_path = quality_report_path or output_path.with_suffix(
            '.quality.json'
        )
        self.min_quality_score = min_quality_score

        self.cleaner = TextCleaner()
        self.chapter_remover = ChapterRemover()
        self.quality_assessor = QualityAssessor()
        self.segmenter = SentenceSegmenter()
        self.metadata_parser = DocumentMetadataParser()

        self.df_input = None
        self.df_output = None

    def load_input(self) -> pd.DataFrame:
        """Load input file (parquet or CSV)."""
        logging.info("=" * 80)
        logging.info("LOADING INPUT FILE")
        logging.info("=" * 80)
        logging.info(f"Input file: {self.input_path}")

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        suffix = self.input_path.suffix.lower()
        if suffix == '.parquet':
            df = pd.read_parquet(self.input_path)
        elif suffix == '.csv':
            df = pd.read_csv(self.input_path)
        elif suffix == '.rds':
            import pyreadr
            rds_data = pyreadr.read_r(str(self.input_path))
            df = rds_data[list(rds_data.keys())[0]]
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported: .parquet, .csv, .rds"
            )

        logging.info(f"Loaded {len(df)} documents")
        logging.info(f"Columns: {df.columns.tolist()}")

        if 'file' not in df.columns or 'text' not in df.columns:
            raise ValueError(
                f"Input must contain 'file' and 'text' columns. "
                f"Found: {df.columns.tolist()}"
            )

        self.df_input = df
        return df

    def _resolve_actor_type(self, row: pd.Series) -> str:
        """Determine actor type from explicit column or filename."""
        if 'actor' in row.index and pd.notna(row.get('actor')):
            actor = str(row['actor']).lower().strip()
            # Normalise common variations
            if actor in ('kommun', 'kommuner'):
                return 'kommun'
            if actor in ('länsstyrelse', 'lansstyrelse', 'länsstyrelsen'):
                return 'lansstyrelse'
            if actor in ('mcf', 'msb', 'myndighet'):
                return 'MCF'
            return actor
        return 'unknown'

    def process_document(
        self, row: pd.Series
    ) -> Tuple[List[Dict], DocumentQuality]:
        """Process a single document through the full pipeline."""
        doc_id = row['file']
        raw_text = row['text'] if pd.notna(row['text']) else ''
        actor_type = self._resolve_actor_type(row)

        # 1. Parse metadata from filename
        metadata = self.metadata_parser.parse_filename(doc_id)

        # 2. Clean text (mojibake, artifacts, whitespace)
        cleaned_text = self.cleaner.clean_document(raw_text)

        # 3. Remove intro chapters (kommun only)
        chapter_trimmed = False
        chapter_trim_failed = False
        if self.remove_intro_chapters and actor_type == 'kommun':
            text_before = cleaned_text
            cleaned_text = self.chapter_remover.remove_intro_chapters(
                cleaned_text, doc_id
            )
            chapter_trimmed = cleaned_text is not text_before
            chapter_trim_failed = (
                not chapter_trimmed
                and cleaned_text is text_before
            )

        # 4. Sentence segmentation
        sentences = self.segmenter.segment_document(cleaned_text)

        # 5. Filter artifact sentences
        filtered = self.quality_assessor.filter_sentences(sentences)

        # 6. Add metadata to each sentence
        for s in filtered:
            s['doc_id'] = doc_id
            s['municipality'] = metadata['municipality']
            s['year'] = metadata['year']
            s['maskad'] = metadata['maskad']
            s['actor_type'] = actor_type

        # 7. Quality assessment
        quality = self.quality_assessor.assess_document(
            doc_id=doc_id,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            sentences_before=sentences,
            sentences_after=filtered,
            chapter_trimmed=chapter_trimmed,
            chapter_trim_failed=chapter_trim_failed,
        )

        return filtered, quality

    def process_all(self) -> pd.DataFrame:
        """Process all documents."""
        logging.info("=" * 80)
        logging.info("PROCESSING DOCUMENTS")
        logging.info("=" * 80)
        logging.info(f"Processing {len(self.df_input)} documents...")
        logging.info("(Stanza tokenize-only; no lemmatization)\n")

        all_sentences = []
        start_time = time.time()

        for idx, row in self.df_input.iterrows():
            doc_num = idx + 1 if isinstance(idx, int) else idx
            total = len(self.df_input)

            if isinstance(doc_num, int) and (doc_num % 10 == 0 or doc_num == 1):
                elapsed = time.time() - start_time
                rate = doc_num / elapsed if elapsed > 0 else 0
                eta = (total - doc_num) / rate if rate > 0 else 0
                logging.info(
                    f"Progress: {doc_num}/{total} "
                    f"({rate:.1f} docs/sec, ETA: {eta / 60:.1f} min)"
                )

            try:
                sentences, quality = self.process_document(row)
                all_sentences.extend(sentences)
            except Exception as e:
                logging.error(f"Error processing {row['file']}: {e}")
                continue

        elapsed = time.time() - start_time
        logging.info(f"\nProcessing complete in {elapsed:.1f}s")

        # Apply quality score filtering if threshold > 0
        if self.min_quality_score > 0:
            quality_map = {
                q.doc_id: q.quality_score
                for q in self.quality_assessor.document_qualities
            }
            before_count = len(all_sentences)
            all_sentences = [
                s for s in all_sentences
                if quality_map.get(s['doc_id'], 0.0) >= self.min_quality_score
            ]
            after_count = len(all_sentences)
            removed_docs = sum(
                1 for q in self.quality_assessor.document_qualities
                if q.quality_score < self.min_quality_score
            )
            logging.info(
                f"Quality filter (>={self.min_quality_score}): "
                f"removed {removed_docs} documents, "
                f"{before_count - after_count} sentences"
            )

        # Add doc_quality score to each sentence
        quality_scores = {
            q.doc_id: q.quality_score
            for q in self.quality_assessor.document_qualities
        }
        for s in all_sentences:
            s['doc_quality'] = quality_scores.get(s['doc_id'], 0.0)

        self.df_output = pd.DataFrame(all_sentences)
        return self.df_output

    def print_statistics(self) -> None:
        """Print summary statistics about the processed corpus."""
        logging.info("\n" + "=" * 80)
        logging.info("STATISTICS")
        logging.info("=" * 80)

        if self.df_output is None or len(self.df_output) == 0:
            logging.info("No output data to report.")
            return

        df = self.df_output
        logging.info(f"Original documents: {len(self.df_input)}")
        logging.info(f"Total sentences: {len(df):,}")

        n_docs = df['doc_id'].nunique()
        logging.info(f"Documents with sentences: {n_docs}")
        logging.info(
            f"Average sentences per document: {len(df) / n_docs:.1f}"
        )
        logging.info(
            f"Average words per sentence: {df['word_count'].mean():.1f}"
        )
        logging.info(
            f"Median words per sentence: {df['word_count'].median():.0f}"
        )

        # Word count distribution
        quartiles = df['word_count'].quantile([0.25, 0.5, 0.75, 0.95])
        logging.info("\nWord count distribution:")
        logging.info(f"  25th percentile: {quartiles[0.25]:.0f} words")
        logging.info(f"  50th percentile: {quartiles[0.50]:.0f} words")
        logging.info(f"  75th percentile: {quartiles[0.75]:.0f} words")
        logging.info(f"  95th percentile: {quartiles[0.95]:.0f} words")

        # Actor distribution
        if 'actor_type' in df.columns:
            logging.info("\nSentences by actor type:")
            for actor, count in df.groupby('actor_type').size().items():
                logging.info(f"  {actor}: {count:,}")

        # Year distribution
        if 'year' in df.columns:
            logging.info("\nSentences by year:")
            for year, count in df.groupby('year').size().sort_index().items():
                logging.info(f"  {year}: {count:,}")

        # Quality summary
        qualities = self.quality_assessor.document_qualities
        if qualities:
            scores = [q.quality_score for q in qualities]
            flags = [q.quality_flag for q in qualities]
            logging.info(f"\nQuality scores: mean={sum(scores)/len(scores):.3f}")
            for flag_val in ["ok", "low_quality", "very_low_quality"]:
                count = flags.count(flag_val)
                if count > 0:
                    logging.info(f"  {flag_val}: {count} documents")

        # Chapter removal summary
        ch_stats = self.chapter_remover.get_stats()
        if ch_stats['documents_trimmed'] > 0 or ch_stats['detection_failed'] > 0:
            logging.info("\nChapter removal:")
            logging.info(
                f"  Documents trimmed: {ch_stats['documents_trimmed']}"
            )
            logging.info(
                f"  Detection failed: {ch_stats['detection_failed']}"
            )
            logging.info(
                f"  Safety guard triggered: "
                f"{ch_stats['safety_guard_triggered']}"
            )
            if ch_stats['documents_trimmed'] > 0:
                avg_removed = (
                    ch_stats['total_chars_removed']
                    / ch_stats['documents_trimmed']
                )
                logging.info(
                    f"  Average chars removed per document: {avg_removed:,.0f}"
                )

        # Cleaning summary
        cl_stats = self.cleaner.get_stats()
        logging.info("\nText cleaning:")
        logging.info(
            f"  Mojibake replacements: {cl_stats['mojibake_replacements']}"
        )
        logging.info(
            f"  Artifact lines removed: {cl_stats['artifact_lines_removed']}"
        )
        logging.info(
            f"  Encoding rescues: {cl_stats['encoding_rescues']}"
        )

    def print_examples(self, n_examples: int = 5) -> None:
        """Print example sentences."""
        if self.df_output is None or len(self.df_output) == 0:
            return

        logging.info("\n" + "=" * 80)
        logging.info("EXAMPLE SENTENCES")
        logging.info("=" * 80)

        sample = self.df_output.sample(
            n=min(n_examples, len(self.df_output)), random_state=42
        )

        for idx, (_, row) in enumerate(sample.iterrows(), 1):
            logging.info(f"\nExample {idx}:")
            logging.info(f"  Document: {row['doc_id']}")
            logging.info(
                f"  {row['municipality']} | {row['year']} | "
                f"actor={row['actor_type']} | quality={row['doc_quality']:.2f}"
            )
            text_preview = row['sentence_text'][:200]
            if len(row['sentence_text']) > 200:
                text_preview += "..."
            logging.info(f"  [{row['word_count']} words] {text_preview}")

    def save_output(self) -> None:
        """Save cleaned corpus to parquet and quality report to JSON."""
        logging.info("\n" + "=" * 80)
        logging.info("SAVING OUTPUT")
        logging.info("=" * 80)

        if self.df_output is None or len(self.df_output) == 0:
            logging.warning("No output data to save.")
            return

        # Ensure column order
        columns = [
            'doc_id', 'municipality', 'year', 'maskad', 'actor_type',
            'sentence_id', 'sentence_text', 'word_count', 'doc_quality',
        ]
        df_save = self.df_output[columns]

        # Save corpus
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_save.to_parquet(self.output_path, index=False)
        logging.info(f"Corpus saved to: {self.output_path}")
        logging.info(f"  Rows: {len(df_save):,}")
        logging.info(f"  Columns: {list(df_save.columns)}")
        logging.info(
            f"  File size: "
            f"{self.output_path.stat().st_size / 1024 / 1024:.1f} MB"
        )

        # Save quality report
        self.quality_assessor.write_quality_report(self.quality_report_path)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    input_path: Path,
    output_path: Path,
    remove_intro_chapters: bool = True,
    quality_report_path: Optional[Path] = None,
    min_quality_score: float = 0.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the complete BERT preprocessing pipeline.

    Parameters
    ----------
    input_path : Path
        Path to input parquet/CSV file.
    output_path : Path
        Path for output parquet file.
    remove_intro_chapters : bool
        Whether to remove intro chapters from kommun documents.
    quality_report_path : Path, optional
        Path for quality report JSON.
    min_quality_score : float
        Minimum quality score to include a document (0.0 = keep all).
    verbose : bool
        Enable verbose logging.

    Returns
    -------
    pd.DataFrame
        Processed sentence-level data.
    """
    setup_logging(verbose)

    logging.info("=" * 80)
    logging.info("BERT CORPUS PREPROCESSING PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Input:  {input_path}")
    logging.info(f"Output: {output_path}")
    logging.info(f"Remove intro chapters: {remove_intro_chapters}")
    logging.info(f"Min quality score: {min_quality_score}")
    logging.info("")

    processor = BertCorpusProcessor(
        input_path=input_path,
        output_path=output_path,
        remove_intro_chapters=remove_intro_chapters,
        quality_report_path=quality_report_path,
        min_quality_score=min_quality_score,
        verbose=verbose,
    )

    processor.load_input()
    df_output = processor.process_all()
    processor.print_statistics()
    processor.print_examples()
    processor.save_output()

    logging.info("\n" + "=" * 80)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 80)
    logging.info("\nNext steps:")
    logging.info("  1. Inspect the quality report for flagged documents")
    logging.info("  2. Run sampling script to create train/test split")
    logging.info("  3. Write codebook and hand-code training sample")
    logging.info("  4. Fine-tune Swedish BERT on the coded sample")

    return df_output


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            'Light preprocessing for BERT corpus: '
            'cleaning, chapter removal, quality assessment'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with merged input
    python preprocessing_bert.py \\
        --input data/merged/pdf_texts_all_actors.parquet \\
        --output data/processed/bert_corpus.parquet

    # Without intro chapter removal
    python preprocessing_bert.py \\
        --input data/merged/pdf_texts_all_actors.parquet \\
        --output data/processed/bert_corpus.parquet \\
        --no-remove-intro-chapters

    # With quality filtering (remove very low quality)
    python preprocessing_bert.py \\
        --input data/merged/pdf_texts_all_actors.parquet \\
        --output data/processed/bert_corpus.parquet \\
        --min-quality-score 0.3

    # Verbose output
    python preprocessing_bert.py \\
        --input data/merged/pdf_texts_all_actors.parquet \\
        --output data/processed/bert_corpus.parquet \\
        --verbose

Input Format:
    Parquet/CSV file with columns:
    - file: PDF filename (e.g., "RSA Ale 2015 Maskad.pdf")
    - text: Extracted text content
    - actor: (optional) Actor type (kommun/lansstyrelse/MCF)

Output Format:
    Parquet file with columns:
    - doc_id, municipality, year, maskad, actor_type
    - sentence_id, sentence_text, word_count
    - doc_quality (per-document quality score 0.0-1.0)

    JSON quality report with per-document metrics.
        """,
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input parquet/CSV file with "file" and "text" columns',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output parquet file',
    )
    parser.add_argument(
        '--quality-report',
        type=Path,
        default=None,
        help='Path for quality report JSON (default: <output>.quality.json)',
    )
    parser.add_argument(
        '--no-remove-intro-chapters',
        action='store_true',
        help='Skip introductory chapter removal for kommun documents',
    )
    parser.add_argument(
        '--min-quality-score',
        type=float,
        default=0.0,
        help='Minimum quality score to include document (default: 0.0 = keep all)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )

    return parser


def main() -> int:
    """Main entry point for command-line execution."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            remove_intro_chapters=not args.no_remove_intro_chapters,
            quality_report_path=args.quality_report,
            min_quality_score=args.min_quality_score,
            verbose=args.verbose,
        )
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "\nMake sure Stanza Swedish model is installed:",
            file=sys.stderr,
        )
        print(
            '  python -c "import stanza; stanza.download(\'sv\')"',
            file=sys.stderr,
        )
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
