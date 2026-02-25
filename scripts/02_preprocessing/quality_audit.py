#!/usr/bin/env python3
"""
Quality Audit Script

Identifies semantically garbage sentences that passed basic parsing filters.
Detects OCR failures, table fragments, and nonsense text using heuristics
and dictionary-based checks.

This is an inspection tool - it flags suspicious sentences for manual review
without automatically filtering them.

Heuristics:
    1. Dictionary coverage - % of tokens that are real Swedish words
    2. Repetition patterns - repeated short tokens, letter spam (AAA RAA)
    3. Mixed case - uppercase letters in middle of words (BEgreppSförklariNG)
    4. Character repetition - same character 3+ times in a row (mmm, AAA)
    5. Short token ratio - too many 1-3 char non-word tokens
    6. Digit ratio - excessive numbers mixed with text

Input:
    Sentence-level parquet or CSV with 'sentence_text' column.

Output:
    - flagged_sentences.csv: Sentences flagged by any heuristic
    - quality_audit_report.json: Summary statistics
    - (optional) all_sentences_scored.csv: Full corpus with scores

Usage:
    # Audit the filtered corpus
    python quality_audit.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/quality_audit/

    # Audit the hand-coding sample
    python quality_audit.py \\
        --input results/sampling/sample_full.csv \\
        --output results/quality_audit/ \\
        --export-all

    # Adjust thresholds
    python quality_audit.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/quality_audit/ \\
        --min-dictionary-coverage 0.4 \\
        --verbose

Requirements:
    pip install pandas pyarrow

Author: Swedish Risk Analysis Text-as-Data Project
Version: 1.0
Date: 2025-02-24
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)

# Swedish common words - high frequency words that are definitely valid
# This is a minimal set; the full dictionary is loaded separately
SWEDISH_STOPWORDS = {
    'och', 'i', 'att', 'det', 'som', 'en', 'på', 'är', 'av', 'för',
    'med', 'till', 'den', 'har', 'de', 'inte', 'om', 'ett', 'kan', 'från',
    'var', 'vi', 'vid', 'eller', 'men', 'så', 'finns', 'vara', 'detta',
    'ska', 'inom', 'samt', 'vid', 'genom', 'utan', 'alla', 'dessa', 'efter',
    'andra', 'hur', 'man', 'där', 'också', 'då', 'under', 'sig', 'nu',
    'än', 'här', 'vad', 'mellan', 'när', 'mot', 'över', 'upp', 'ut',
    'även', 'bara', 'får', 'flera', 'kommer', 'mycket', 'skulle', 'varit',
    'blir', 'dag', 'del', 'deras', 'dock', 'enligt', 'finns', 'få', 'går',
    'göra', 'hade', 'hela', 'honom', 'hon', 'hos', 'ingen', 'redan', 'se',
    'sedan', 'sin', 'sina', 'sitt', 'stor', 'stora', 'ta', 'tidigare',
    'två', 'vilka', 'vilket', 'väl', 'år', 'många', 'måste', 'något',
    'några', 'nya', 'olika', 'endast', 'kunna', 'bli', 'blev', 'bör',
}

# Swedish acronyms common in government/administrative documents
# These should NOT trigger the letter_spam detector
SWEDISH_ACRONYMS = {
    # Government agencies
    'MSB', 'SCB', 'SKR', 'SKL', 'FOI', 'FMV', 'SOS', 'SSM', 'SVA', 'SMHI',
    'FHM', 'IVO', 'SGU', 'SGI', 'SLU', 'KTH', 'LTH', 'VTI', 'TRV', 'STA',
    'PTS', 'SEK', 'SIS', 'SBU', 'TLV', 'RAÄ', 'MTM', 'SSB',
    # Emergency/crisis management
    'RSA', 'ROL', 'RCB', 'TIB', 'TIS', 'WIS', 'FRG', 'PDV', 'MRF', 'MCF',
    'CBRN', 'CBRNE', 'NBC', 'RVR', 'KKP', 'VMA', 'IVA',
    # Infrastructure/utilities
    'IT', 'VA', 'AB', 'KB', 'HB', 'EL', 'FM', 'TV', 'PC', 'GPS', 'GSM',
    'LTE', 'DNS', 'VPN', 'ICT', 'ISP', 'API',
    # International
    'EU', 'FN', 'UN', 'WHO', 'NATO', 'OECD', 'WEF', 'IMF', 'NGO', 'IPCC',
    # Military/defense
    'FM', 'HKV', 'MHS', 'FHS', 'FOA', 'FRA', 'KBV', 'PMF', 'ÖB',
    # Regional/municipal
    'KS', 'KF', 'KL', 'LS', 'LF', 'RF', 'BN', 'MN', 'SN', 'TN', 'ON',
    # Document types
    'RSA', 'ROS', 'LEH', 'LBE', 'RIB', 'SBA', 'LSO', 'OSL', 'PBL', 'MB',
    # Other common
    'VD', 'HR', 'PR', 'CV', 'ID', 'NR', 'TEL', 'FAX', 'WWW', 'URL', 'PDF',
    'BNP', 'KPI', 'SEK', 'EUR', 'USD', 'MKR', 'MDR', 'TWH', 'GWH', 'MWH',
}

# Common Swedish word patterns (endings that indicate real words)
SWEDISH_ENDINGS = [
    'tion', 'ning', 'het', 'ande', 'else', 'skap', 'lig', 'isk',
    'erna', 'arna', 'orna', 'erna', 'ens', 'ets', 'ans',
]

# Default thresholds
DEFAULT_MIN_DICT_COVERAGE = 0.5  # At least 50% real words
DEFAULT_MAX_REPEAT_RATIO = 0.3   # No more than 30% repeated tokens
DEFAULT_MAX_SHORT_TOKEN_RATIO = 0.5  # No more than 50% very short tokens
DEFAULT_MAX_DIGIT_RATIO = 0.3    # No more than 30% digits


# =============================================================================
# SWEDISH DICTIONARY
# =============================================================================

class SwedishDictionary:
    """Simple Swedish word validator using multiple strategies."""

    def __init__(self, custom_words: Optional[Set[str]] = None):
        """Initialize dictionary with stopwords and optional custom words."""
        self.words = set(SWEDISH_STOPWORDS)
        if custom_words:
            self.words.update(custom_words)

        # Build from corpus frequency if available
        self.corpus_words: Set[str] = set()

    def build_from_corpus(self, sentences: List[str], min_freq: int = 5) -> None:
        """Build word set from corpus frequency."""
        word_counts: Counter = Counter()

        for sent in sentences:
            tokens = self._tokenize(sent)
            word_counts.update(tokens)

        # Words appearing frequently are likely real
        self.corpus_words = {
            word for word, count in word_counts.items()
            if count >= min_freq and len(word) >= 3
        }

        logger.info(f"Built corpus dictionary: {len(self.corpus_words)} words (freq >= {min_freq})")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'[a-zåäö]+', text)
        return tokens

    def is_swedish_word(self, word: str) -> bool:
        """Check if word is likely Swedish."""
        word = word.lower()

        # Known word
        if word in self.words or word in self.corpus_words:
            return True

        # Has Swedish ending
        for ending in SWEDISH_ENDINGS:
            if word.endswith(ending) and len(word) > len(ending) + 2:
                return True

        return False

    def coverage(self, text: str) -> float:
        """Calculate dictionary coverage for text."""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0

        recognized = sum(1 for t in tokens if self.is_swedish_word(t))
        return recognized / len(tokens)


# =============================================================================
# HEURISTIC CHECKS
# =============================================================================

class QualityChecker:
    """Applies multiple heuristics to detect garbage sentences."""

    def __init__(
        self,
        dictionary: SwedishDictionary,
        min_dict_coverage: float = DEFAULT_MIN_DICT_COVERAGE,
        max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
        max_short_token_ratio: float = DEFAULT_MAX_SHORT_TOKEN_RATIO,
        max_digit_ratio: float = DEFAULT_MAX_DIGIT_RATIO,
    ):
        self.dictionary = dictionary
        self.min_dict_coverage = min_dict_coverage
        self.max_repeat_ratio = max_repeat_ratio
        self.max_short_token_ratio = max_short_token_ratio
        self.max_digit_ratio = max_digit_ratio

    def check_dictionary_coverage(self, text: str) -> Tuple[bool, float]:
        """Check if enough tokens are real Swedish words."""
        coverage = self.dictionary.coverage(text)
        flagged = coverage < self.min_dict_coverage
        return flagged, coverage

    def check_repetition(self, text: str) -> Tuple[bool, float, List[str]]:
        """Check for repeated short tokens (OCR reading patterns).

        Excludes common Swedish stopwords which naturally repeat in prose.
        Only flags repetition of unusual short tokens that suggest OCR garbage.
        """
        tokens = text.lower().split()
        if len(tokens) < 3:
            return False, 0.0, []

        # Common Swedish words that naturally repeat - don't count these
        common_words = {
            'och', 'i', 'att', 'det', 'som', 'en', 'på', 'är', 'av', 'för',
            'med', 'till', 'den', 'har', 'de', 'inte', 'om', 'ett', 'kan',
            'från', 'var', 'vi', 'vid', 'eller', 'men', 'så', 'ska', 'då',
            'nu', 'sig', 'sin', 'ut', 'upp', 'ta', 'få', 'se', 'år', 'nya',
            'del', 'kan', 'hur', 'där', 'här', 'vad', 'när', 'mot', 'över',
            'samt', 'inom', 'utan', 'alla', 'även', 'bara', 'blir', 'dock',
            'inte', 'vara', 'varit', 'bli', 'blev', 'bör', 'får', 'går',
            'ger', 'gör', 'har', 'hör', 'kan', 'kom', 'ska', 'stor', 'tas',
            'vid', 'vår', 'din', 'dom', 'era', 'han', 'hon', 'man', 'dem',
            'den', 'det', 'ett', 'ett', 'två', 'tre', 'per', 'vid', 'hos',
        }

        # Count token frequencies
        counts = Counter(tokens)

        # Find repeated short tokens that are NOT common words
        repeated = [
            tok for tok, count in counts.items()
            if count >= 3  # Must appear 3+ times (stricter)
            and 1 <= len(tok) <= 4
            and tok not in common_words
            and not tok.isdigit()  # Exclude numbers
        ]

        if not repeated:
            return False, 0.0, []

        repeat_count = sum(counts[tok] for tok in repeated)
        repeat_ratio = repeat_count / len(tokens)

        # Only flag if substantial repetition of unusual tokens
        flagged = repeat_ratio > self.max_repeat_ratio or len(repeated) >= 4
        return flagged, repeat_ratio, repeated

    def check_letter_spam(self, text: str) -> Tuple[bool, List[str]]:
        """Check for letter spam patterns like AAA, RAA, mmm.

        Excludes known Swedish acronyms (MSB, SCB, etc.) which are common
        in government documents.
        """
        # Find ALL CAPS tokens (2-5 letters)
        all_caps_tokens = re.findall(r'\b[A-Z]{2,5}\b', text)

        # Filter out known acronyms
        spam_tokens = [
            tok for tok in all_caps_tokens
            if tok not in SWEDISH_ACRONYMS
        ]

        # Also check for repeated letter patterns (AAA, AAAA)
        repeated_letters = re.findall(r'\b([A-Z])\1{2,}\b', text)
        spam_tokens.extend(repeated_letters)

        # Only flag if many spam tokens AND they look like garbage
        # (same token repeated, or pattern of repeated letters)
        token_counts = Counter(spam_tokens)
        repeated_spam = sum(1 for tok, count in token_counts.items() if count >= 2)

        # Flag if: many unknown caps tokens OR repeated spam patterns
        flagged = len(spam_tokens) >= 5 or repeated_spam >= 2
        return flagged, spam_tokens

    def check_mixed_case(self, text: str) -> Tuple[bool, List[str]]:
        """Check for weird capitalization within words."""
        # Pattern: lowercase followed by uppercase mid-word
        mixed_pattern = re.compile(r'\b\w*[a-zåäö][A-ZÅÄÖ][a-zåäö]\w*\b')
        matches = mixed_pattern.findall(text)

        flagged = len(matches) >= 1
        return flagged, matches

    def check_char_repetition(self, text: str) -> Tuple[bool, List[str]]:
        """Check for same character repeated 3+ times.

        Excludes common patterns:
        - Ellipsis (...)
        - Separator lines (---, ___)
        - Numbers (000, 111) - Swedish number formatting uses "30 000"
        - URL patterns (www, http)
        """
        # Find all repeated character sequences
        significant = []
        for m in re.finditer(r'(.)\1{2,}', text):
            seq = m.group(0)
            char = m.group(1)

            # Skip common legitimate patterns
            if seq in ['...', '---', '___', '***', '===']:
                continue
            # Skip repeated digits (Swedish number formatting: "30 000")
            if char.isdigit():
                continue
            # Skip if it's just whitespace
            if char.isspace():
                continue
            # Skip URL patterns (www, http://, https://)
            if seq.lower() == 'www':
                continue
            # Skip common repeated letters in URLs (e.g., "http" has no repeats but check anyway)
            # Check if this is part of a URL context
            start = max(0, m.start() - 10)
            end = min(len(text), m.end() + 5)
            context = text[start:end].lower()
            if 'www' in context or 'http' in context or '.se' in context or '.com' in context:
                continue

            significant.append(seq)

        # Only flag if multiple suspicious repetitions
        flagged = len(significant) >= 2
        return flagged, significant

    def check_short_tokens(self, text: str) -> Tuple[bool, float]:
        """Check ratio of very short non-word tokens."""
        tokens = text.split()
        if not tokens:
            return False, 0.0

        short_nonwords = [
            t for t in tokens
            if len(t) <= 3 and not self.dictionary.is_swedish_word(t.lower())
            and t.lower() not in {'och', 'att', 'det', 'som', 'en', 'på', 'är',
                                   'av', 'för', 'med', 'den', 'har', 'de', 'om',
                                   'ett', 'kan', 'var', 'vid', 'men', 'så', 'nu',
                                   'än', 'sig', 'ut', 'få', 'se', 'ta', 'två'}
        ]

        ratio = len(short_nonwords) / len(tokens)
        flagged = ratio > self.max_short_token_ratio
        return flagged, ratio

    def check_digit_ratio(self, text: str) -> Tuple[bool, float]:
        """Check if too many characters are digits."""
        if not text:
            return False, 0.0

        digits = sum(1 for c in text if c.isdigit())
        ratio = digits / len(text)

        flagged = ratio > self.max_digit_ratio
        return flagged, ratio

    def check_sentence(self, text: str) -> Dict:
        """Run all checks on a sentence and return results."""
        results = {
            'text': text,
            'flagged': False,
            'flags': [],
            'scores': {},
        }

        # Dictionary coverage
        dict_flagged, dict_coverage = self.check_dictionary_coverage(text)
        results['scores']['dict_coverage'] = round(dict_coverage, 3)
        if dict_flagged:
            results['flags'].append('low_dict_coverage')

        # Repetition
        rep_flagged, rep_ratio, rep_tokens = self.check_repetition(text)
        results['scores']['repeat_ratio'] = round(rep_ratio, 3)
        if rep_flagged:
            results['flags'].append('repetition')
            results['repeated_tokens'] = rep_tokens

        # Letter spam
        spam_flagged, spam_tokens = self.check_letter_spam(text)
        if spam_flagged:
            results['flags'].append('letter_spam')
            results['spam_tokens'] = spam_tokens

        # Mixed case
        mixed_flagged, mixed_words = self.check_mixed_case(text)
        if mixed_flagged:
            results['flags'].append('mixed_case')
            results['mixed_case_words'] = mixed_words

        # Character repetition
        char_flagged, char_seqs = self.check_char_repetition(text)
        if char_flagged:
            results['flags'].append('char_repetition')
            results['repeated_chars'] = char_seqs

        # Short tokens
        short_flagged, short_ratio = self.check_short_tokens(text)
        results['scores']['short_token_ratio'] = round(short_ratio, 3)
        if short_flagged:
            results['flags'].append('short_tokens')

        # Digit ratio
        digit_flagged, digit_ratio = self.check_digit_ratio(text)
        results['scores']['digit_ratio'] = round(digit_ratio, 3)
        if digit_flagged:
            results['flags'].append('high_digits')

        # Overall flag
        results['flagged'] = len(results['flags']) > 0
        results['flag_count'] = len(results['flags'])

        return results


# =============================================================================
# MAIN AUDITOR
# =============================================================================

class QualityAuditor:
    """Main orchestrator for quality audit."""

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        min_dict_coverage: float = DEFAULT_MIN_DICT_COVERAGE,
        max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
        export_all: bool = False,
        verbose: bool = False,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.min_dict_coverage = min_dict_coverage
        self.max_repeat_ratio = max_repeat_ratio
        self.export_all = export_all
        self.verbose = verbose

        self.df = None
        self.results = []
        self.dictionary = SwedishDictionary()
        self.checker = None

    def load_data(self) -> pd.DataFrame:
        """Load input file."""
        logger.info("=" * 60)
        logger.info("LOADING DATA")
        logger.info("=" * 60)

        suffix = self.input_path.suffix.lower()
        if suffix == '.parquet':
            self.df = pd.read_parquet(self.input_path)
        elif suffix == '.csv':
            self.df = pd.read_csv(self.input_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        # Find text column
        text_col = None
        for col in ['sentence_text', 'paragraph_text', 'text']:
            if col in self.df.columns:
                text_col = col
                break

        if text_col is None:
            raise ValueError(f"No text column found. Columns: {list(self.df.columns)}")

        self.text_col = text_col
        logger.info(f"Loaded {len(self.df):,} rows from {self.input_path}")
        logger.info(f"Text column: {text_col}")

        return self.df

    def build_dictionary(self) -> None:
        """Build dictionary from corpus."""
        logger.info("\nBuilding dictionary from corpus...")
        texts = self.df[self.text_col].dropna().tolist()
        self.dictionary.build_from_corpus(texts, min_freq=5)

    def run_audit(self) -> List[Dict]:
        """Run quality checks on all sentences."""
        logger.info("=" * 60)
        logger.info("RUNNING QUALITY AUDIT")
        logger.info("=" * 60)

        self.checker = QualityChecker(
            dictionary=self.dictionary,
            min_dict_coverage=self.min_dict_coverage,
            max_repeat_ratio=self.max_repeat_ratio,
        )

        texts = self.df[self.text_col].fillna('').tolist()
        total = len(texts)

        logger.info(f"Checking {total:,} sentences...")
        logger.info(f"Thresholds: dict_coverage >= {self.min_dict_coverage}, "
                   f"repeat_ratio <= {self.max_repeat_ratio}")

        self.results = []
        for i, text in enumerate(texts):
            if i % 10000 == 0 and i > 0:
                logger.info(f"  Progress: {i:,}/{total:,}")

            result = self.checker.check_sentence(text)
            result['row_idx'] = i
            self.results.append(result)

        flagged_count = sum(1 for r in self.results if r['flagged'])
        logger.info(f"\nFlagged: {flagged_count:,} / {total:,} ({flagged_count/total*100:.1f}%)")

        return self.results

    def generate_report(self) -> Dict:
        """Generate summary report."""
        total = len(self.results)
        flagged = [r for r in self.results if r['flagged']]

        # Count by flag type
        flag_counts = Counter()
        for r in flagged:
            for flag in r['flags']:
                flag_counts[flag] += 1

        # Score distributions for flagged sentences
        flagged_scores = {
            'dict_coverage': [r['scores']['dict_coverage'] for r in flagged],
            'repeat_ratio': [r['scores']['repeat_ratio'] for r in flagged],
            'short_token_ratio': [r['scores']['short_token_ratio'] for r in flagged],
        }

        report = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'input_file': str(self.input_path),
                'total_sentences': total,
                'thresholds': {
                    'min_dict_coverage': self.min_dict_coverage,
                    'max_repeat_ratio': self.max_repeat_ratio,
                }
            },
            'summary': {
                'flagged_sentences': len(flagged),
                'flagged_pct': round(len(flagged) / total * 100, 2),
                'clean_sentences': total - len(flagged),
            },
            'flags_by_type': dict(flag_counts.most_common()),
            'flagged_score_means': {
                k: round(sum(v) / len(v), 3) if v else 0
                for k, v in flagged_scores.items()
            },
            'examples': {
                'worst_dict_coverage': sorted(
                    flagged, key=lambda x: x['scores']['dict_coverage']
                )[:5],
                'highest_repetition': sorted(
                    flagged, key=lambda x: -x['scores']['repeat_ratio']
                )[:5],
            }
        }

        return report

    def save_outputs(self) -> None:
        """Save flagged sentences and report."""
        logger.info("=" * 60)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 60)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Flagged sentences CSV
        flagged = [r for r in self.results if r['flagged']]

        flagged_rows = []
        for r in flagged:
            row = {
                'row_idx': r['row_idx'],
                'text': r['text'][:500],  # Truncate for readability
                'flags': ', '.join(r['flags']),
                'flag_count': r['flag_count'],
                'dict_coverage': r['scores']['dict_coverage'],
                'repeat_ratio': r['scores']['repeat_ratio'],
                'short_token_ratio': r['scores']['short_token_ratio'],
                'digit_ratio': r['scores']['digit_ratio'],
            }

            # Add original metadata if available
            if self.df is not None:
                orig_row = self.df.iloc[r['row_idx']]
                for col in ['doc_id', 'sentence_id', 'paragraph_id', 'actor_type', 'year']:
                    if col in orig_row.index:
                        row[col] = orig_row[col]

            flagged_rows.append(row)

        flagged_df = pd.DataFrame(flagged_rows)
        flagged_path = self.output_dir / 'flagged_sentences.csv'
        flagged_df.to_csv(flagged_path, index=False)
        logger.info(f"Saved: {flagged_path} ({len(flagged_df):,} rows)")

        # Full corpus with scores (optional)
        if self.export_all:
            all_rows = []
            for r in self.results:
                row = {
                    'row_idx': r['row_idx'],
                    'flagged': r['flagged'],
                    'flags': ', '.join(r['flags']) if r['flags'] else '',
                    'dict_coverage': r['scores']['dict_coverage'],
                    'repeat_ratio': r['scores']['repeat_ratio'],
                    'short_token_ratio': r['scores']['short_token_ratio'],
                }
                all_rows.append(row)

            all_df = pd.DataFrame(all_rows)
            all_path = self.output_dir / 'all_sentences_scored.csv'
            all_df.to_csv(all_path, index=False)
            logger.info(f"Saved: {all_path} ({len(all_df):,} rows)")

        # Report JSON
        report = self.generate_report()
        report_path = self.output_dir / 'quality_audit_report.json'

        # Clean up non-serializable parts of examples
        for key in ['worst_dict_coverage', 'highest_repetition']:
            report['examples'][key] = [
                {'text': r['text'][:200], 'flags': r['flags'], 'scores': r['scores']}
                for r in report['examples'][key]
            ]

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {report_path}")

    def print_summary(self) -> None:
        """Print summary to console."""
        report = self.generate_report()

        logger.info("\n" + "=" * 60)
        logger.info("AUDIT SUMMARY")
        logger.info("=" * 60)

        logger.info(f"\nTotal sentences: {report['metadata']['total_sentences']:,}")
        logger.info(f"Flagged: {report['summary']['flagged_sentences']:,} "
                   f"({report['summary']['flagged_pct']:.1f}%)")
        logger.info(f"Clean: {report['summary']['clean_sentences']:,}")

        logger.info("\nFlags by type:")
        for flag, count in report['flags_by_type'].items():
            logger.info(f"  {flag}: {count:,}")

        logger.info("\nExample garbage sentences:")
        for ex in report['examples']['worst_dict_coverage'][:3]:
            text_preview = ex['text'][:80] + '...' if len(ex['text']) > 80 else ex['text']
            logger.info(f"  [{ex['scores']['dict_coverage']:.2f}] {text_preview}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Audit sentence quality and flag garbage text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Audit filtered corpus
    python quality_audit.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/quality_audit/

    # Audit hand-coding sample with all scores exported
    python quality_audit.py \\
        --input results/sampling/sample_full.csv \\
        --output results/quality_audit/ \\
        --export-all

    # Stricter thresholds
    python quality_audit.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/quality_audit/ \\
        --min-dictionary-coverage 0.5
"""
    )

    parser.add_argument(
        '--input', type=Path, required=True,
        help='Input parquet or CSV file'
    )
    parser.add_argument(
        '--output', type=Path, default=Path('results/quality_audit'),
        help='Output directory'
    )
    parser.add_argument(
        '--min-dictionary-coverage', type=float, default=DEFAULT_MIN_DICT_COVERAGE,
        help=f'Minimum Swedish word coverage (default: {DEFAULT_MIN_DICT_COVERAGE})'
    )
    parser.add_argument(
        '--max-repeat-ratio', type=float, default=DEFAULT_MAX_REPEAT_RATIO,
        help=f'Maximum repeated token ratio (default: {DEFAULT_MAX_REPEAT_RATIO})'
    )
    parser.add_argument(
        '--export-all', action='store_true',
        help='Export scores for all sentences, not just flagged'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logger.info("=" * 60)
    logger.info("QUALITY AUDIT")
    logger.info("=" * 60)

    try:
        auditor = QualityAuditor(
            input_path=args.input,
            output_dir=args.output,
            min_dict_coverage=args.min_dictionary_coverage,
            max_repeat_ratio=args.max_repeat_ratio,
            export_all=args.export_all,
            verbose=args.verbose,
        )

        auditor.load_data()
        auditor.build_dictionary()
        auditor.run_audit()
        auditor.save_outputs()
        auditor.print_summary()

        logger.info("\n" + "=" * 60)
        logger.info("AUDIT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nOutput: {args.output}")
        logger.info("Review flagged_sentences.csv to inspect garbage text.")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
