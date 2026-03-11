#!/usr/bin/env python3
"""
Risk Term Paragraph Filter

Filters sentence corpus to only include paragraphs containing at least one risk term.
This improves sample quality by excluding methodology, boilerplate, and irrelevant
sections while preserving full paragraph context around risk mentions.

The filter works at paragraph level (not sentence level) to ensure that:
- Context around risk terms is preserved
- Theoretical mechanisms appearing near risk terms are included
- Only paragraphs without any risk content are excluded

Input:
    Sentence-level parquet from preprocessing_bert.py with paragraph_id column.
    Must have columns: doc_id, paragraph_id, sentence_text

Output:
    Filtered parquet with same schema, containing only sentences from paragraphs
    that contain at least one risk term.

Usage:
    # Filter to paragraphs with any risk term
    python risk_term_filter.py \
        --input data/processed/bert_corpus.parquet \
        --output data/processed/bert_corpus_filtered.parquet

    # Filter to specific risk categories only
    python risk_term_filter.py \
        --input data/processed/bert_corpus.parquet \
        --output data/processed/bert_corpus_filtered.parquet \
        --categories naturhot biologiska_hot antagonistiska_hot

    # Require multiple risk terms per paragraph
    python risk_term_filter.py \
        --input data/processed/bert_corpus.parquet \
        --output data/processed/bert_corpus_filtered.parquet \
        --min-terms 2

Requirements:
    pip install pandas pyarrow
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# =============================================================================
# RISK DICTIONARY
# =============================================================================

# Risk dictionary copied from risk_context_analysis.py to avoid import dependencies.
# This is the original (non-lemmatized) version for pattern matching.
# Does not include "risk" as a word to avoid catching "risk- och sårbarhetsanalys".
# Does include some related words to risk however, such as säkerhet, following Boholm (2016)
RISK_DICTIONARY = {
    'naturhot': [
        'naturhändelser', 'naturhot', 'väderrelaterade händelser',
        'klimatförändring', 'klimatförändringarna', 'klimatförändringar',
        'översvämning', 'översvämningar', 'skyfall', 'höga flöden', 'högvatten',
        'värme', 'värmebölja', 'värmeböljor', 'torka', 'torkor',
        'ras', 'skred', 'jordskred', 'slamskred', 'erosion',
        'storm', 'stormar', 'stormfällning',
        'skogsbrand', 'skogsbränder', 'gräsbrand',
        'blixt', 'blixtnedslag', 'hagel', 'halka', 'köldknäpp',
        'stora snömängder', 'snöoväder',
        'extrem värme', 'extremvärme', 'extrem kyla',
        'låga flöden', 'lågvatten',
        # Added: seismic, avalanche, tsunami, water scarcity
        'jordbävning', 'jordskalv', 'seismisk',
        'lavin', 'laviner', 'snölavin',
        'tsunami', 'tsunamier',
        'vattenbrist', 'grundvattennivå', 'grundvattenbrist',
    ],
    'biologiska_hot': [
        'epidemi', 'epidemier', 'pandemi', 'pandemier',
        'epozooti', 'epizootier', 'covid', 'coronaviruset',
        'smittsam sjukdom', 'smittsamma sjukdomar',
        'smitta', 'smittspridning', 'sjukdomsutbrott',
        'influensa', 'influensapandemi',
        'djursjukdom', 'djursjukdomar', 'zoonos', 'zoonoser',
        'antibiotikaresistens', 'resistenta bakterier',
        'hälsa', 'folkhälsa',
    ],
    'olyckor': [
        'olycka vid farlig verksamhet', 'farlig verksamhet',
        'industriolycka', 'kemikalieolycka',
        'olycka med transport av farligt gods', 'olycka med farligt gods',
        'farligt gods', 'transport av farligt gods',
        'vägolycka', 'vägolyckor', 'trafikolycka', 'trafikolyckor',
        'tågolycka', 'tågolyckor', 'järnvägsolycka', 'järnvägsolyckor',
        'bussolycka', 'bussolyckor', 'spårbundna olyckor',
        'dammbrott',
        'fartygsolycka', 'fartygsolyckor', 'båtolycka', 'båtolyckor',
        'flygolycka', 'flygolyckor', 'flyghaveri',
        'olyckor med nukleära ämnen', 'olyckor med radioaktiva ämnen',
        'kärnkraftsolycka', 'strålning', 'radioaktivitet', 'kärnavfall',
        'brokollaps', 'tunnelolycka',
        'byggnadsras', 'byggnadskollaps',
        'försvunnen person', 'försvunna personer', 'försvunnen brukare',
        'försvinnande', 'saknad person', 
    ],
    'antagonistiska_hot': [
        'statliga antagonister', 'statlig antagonist',
        'icke-statliga antagonister', 'icke-statlig antagonist',
        'terror', 'terrorism', 'terrorhot', 'terrorattentat',
        'hot och våld', 'våld', 'våldsbrott',
        'pågående dödligt våld', 'våldsbejakande extremism',
        'sabotage', 'spionage',
        'brott', 'kriminalitet', 'organiserad brottslighet',
        'vandalism', 'skadegörelse', 'inbrott',
        'desinformation', 'påverkanskampanj', 'påverkanskampanjer',
        'hybrid hot', 'hybridhot',
        'säkerhetshot', 'säkerhet', 'hot om attentat',
        'kidnapping', 'väpnad konflikt', 'påverkansoperationer',
        'upplopp', 'beväpnad konflikt',
        # Added: military/war
        'krig', 'väpnat angrepp', 'invasion', 'mobilisering',
        'totalförsvar', 'totalförsvaret', 'krigstillstånd',
        'militärt hot', 'militärt angrepp',
    ],
    'cyber_hot': [
        'dataintrång', 'cyberattack', 'cyberattacker', 'cybersäkerhet',
        'nätattack', 'nätattacker', 'hackerattack', 'hackerattacker',
        'DDoS-attack', 'ddos-attack', 'ransomware', 'datavirus', 'virus',
        'IT-sabotage',
    ],
    'sociala_risker': [
        'samhällsvärden', 'värdesystem',
        'social oro', 'sociala oroligheter', 'civila oroligheter', 'upplopp',
        # Added: mass gatherings
        'folksamling', 'folksamlingar', 'stora evenemang', 'publikevenemang',
        'massevenemang', 'stor tillställning',
        # Added: migration/demographic
        'flyktingkris', 'migration', 'flyktingström', 'flyktingströmmar',
        'massflykt',
    ],
    'samhällsfunktioner': [
        # Healthcare capacity
        'vårdkapacitet', 'vårdkris', 'sjukvårdsbrist', 'sjukvårdskris',
        'vårdplatsbrist', 'intensivvårdsbrist',
        # Personnel/competence
        'kompetensbrist', 'personalbrist', 'bemanningsproblem',
        'kompetensförsörjning', 'personalförsörjning',
    ],
    'teknisk_infrastruktur': [
        'strömavbrott', 'elavbrott', 'kraftförsörjning', 'elförsörjning', 'effektbrist',
        'fjärrvärmebrott', 'fjärrvärme', 'värmeförsörjning',
        'vattenläcka', 'vattenläckor', 'vattenförsörjning', 'dricksvatten',
        'avloppsbrott', 'avloppssystem', 'dricksvattensförsörjning',
        'IT-bortfall', 'it-bortfall', 'IT-avbrott', 'it-avbrott',
        'dataförlust', 'systemfel', 'nätverksavbrott',
        'kommunikationsavbrott', 'teleavbrott', 'telebrott',
        'distributionsstörning', 'logistikavbrott', 'transportavbrott',
        'drivsmedelsbrist', 'bränslebrist', 'försörjningsbrist',
        'livsmedelsförsörjning', 'livsmedelsbrist', 'matförsörjning',
        # Added: supply chain
        'leveranskedja', 'leveranskedjor', 'leveransstörning',
        'försörjningskedja', 'försörjningskedjor',
    ],
    'brand': [
        'brand', 'bränder', 'skogsbrand', 'skogsbränder',
        'gräsbrand', 'gräsbränder', 'byggnadsbrand', 'fordonsbrand',
        'explosion', 'explosioner', 'gasexplosion', 'brandfarligt gods',
    ],
    'miljö_klimat': [
        'miljöförorening', 'kemikalieutsläpp', 'oljeutsläpp',
        'markförorening', 'luftföroreningar', 'vattenförorening',
        'miljöhot', 'miljöskada', 'utsläpp', 'föroreningar', 'klimatförändring',
        'klimatpåverkan', 'klimatrelaterade', 'klimatförändringen', 'försurning',
    ],
    'ekonomi': [
        'ekonomisk kris', 'finanskris', 'recession',
        'arbetslöshet', 'inflation', 'ekonomisk nedgång',
    ],
    'riskfamilj': [
        'säkerhet', 'riskabel', 'riskerar', 'fara', 'farlig',
        'säker', 'exponering', 'känslig', 'utsatthet', 'beredskap', 'försvar',
        'säkerställa', 'granskning', 'bevakning', 'tillsyn', 'drabba', 'sårbarhet',
        'haveri', 'katastrof', 
    ],
    'legitimitetsrisker': [
       'samhällsvärden', 'värdesystem',
        'social oro', 'sociala oroligheter',
        'tillit', 'misstro', 'förtroende',
    ]
}


# =============================================================================
# RISK TERM DETECTION
# =============================================================================

def build_risk_pattern(
    categories: Optional[List[str]] = None,
) -> re.Pattern:
    """
    Build compiled regex pattern for risk term detection.

    Parameters
    ----------
    categories : list of str, optional
        Categories to include. If None, uses all categories.

    Returns
    -------
    re.Pattern
        Compiled regex matching any risk term (case-insensitive).
    """
    if categories is None:
        categories = list(RISK_DICTIONARY.keys())

    # Collect all terms from selected categories
    all_terms = []
    for cat in categories:
        if cat in RISK_DICTIONARY:
            all_terms.extend(RISK_DICTIONARY[cat])
        else:
            logger.warning(f"Unknown category '{cat}', skipping.")

    if not all_terms:
        raise ValueError(f"No risk terms found for categories: {categories}")

    # Sort by length (longest first) to avoid partial matches
    all_terms = sorted(set(all_terms), key=len, reverse=True)

    # Build alternation pattern with word boundaries
    term_patterns = [r'\b' + re.escape(term) + r'\b' for term in all_terms]
    combined = '|'.join(term_patterns)

    return re.compile(combined, re.IGNORECASE)


def count_risk_terms(text: str, pattern: re.Pattern) -> int:
    """
    Count number of risk term matches in text.

    Parameters
    ----------
    text : str
        Text to search.
    pattern : re.Pattern
        Compiled risk term pattern.

    Returns
    -------
    int
        Number of matches.
    """
    if not text:
        return 0
    return len(pattern.findall(text))


# =============================================================================
# PARAGRAPH FILTERING
# =============================================================================

def filter_by_risk_paragraphs(
    df: pd.DataFrame,
    risk_pattern: re.Pattern,
    min_terms: int = 1,
) -> pd.DataFrame:
    """
    Filter corpus to sentences from paragraphs containing risk terms.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level dataframe with columns: doc_id, paragraph_id, sentence_text.
    risk_pattern : re.Pattern
        Compiled regex for risk term detection.
    min_terms : int
        Minimum risk terms required per paragraph.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only sentences from qualifying paragraphs.
    """
    logger.info("Computing risk term counts per paragraph...")

    # Concatenate all sentences within each (doc_id, paragraph_id) group
    paragraph_texts = df.groupby(['doc_id', 'paragraph_id'])['sentence_text'].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()
    paragraph_texts.columns = ['doc_id', 'paragraph_id', 'full_paragraph_text']

    # Count risk terms in each paragraph
    paragraph_texts['risk_term_count'] = paragraph_texts['full_paragraph_text'].apply(
        lambda text: count_risk_terms(text, risk_pattern)
    )

    # Identify paragraphs meeting the threshold
    qualifying = paragraph_texts[paragraph_texts['risk_term_count'] >= min_terms]
    qualifying_keys = set(
        zip(qualifying['doc_id'], qualifying['paragraph_id'])
    )

    logger.info(
        f"  Paragraphs with >= {min_terms} risk term(s): "
        f"{len(qualifying_keys)} / {len(paragraph_texts)}"
    )

    # Filter original dataframe to sentences from qualifying paragraphs
    df_filtered = df[
        df.apply(
            lambda row: (row['doc_id'], row['paragraph_id']) in qualifying_keys,
            axis=1
        )
    ].copy()

    return df_filtered


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Filter corpus to paragraphs containing risk terms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Filter to paragraphs with any risk term
    python risk_term_filter.py \\
        --input data/processed/bert_corpus.parquet \\
        --output data/processed/bert_corpus_filtered.parquet

    # Filter to specific risk categories only
    python risk_term_filter.py \\
        --input data/processed/bert_corpus.parquet \\
        --output data/processed/bert_corpus_filtered.parquet \\
        --categories naturhot biologiska_hot antagonistiska_hot

    # Require at least 2 risk terms per paragraph
    python risk_term_filter.py \\
        --input data/processed/bert_corpus.parquet \\
        --output data/processed/bert_corpus_filtered.parquet \\
        --min-terms 2

Available categories:
    naturhot, biologiska_hot, olyckor, antagonistiska_hot, cyber_hot,
    sociala_risker, teknisk_infrastruktur, brand, miljö_klimat, ekonomi,
    samhällsfunktioner, riskfamilj, legitimitetsrisker
        """,
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input parquet file (sentence-level with paragraph_id column)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output parquet file (filtered corpus)',
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=None,
        help='Risk categories to filter by (default: all categories)',
    )
    parser.add_argument(
        '--min-terms',
        type=int,
        default=1,
        help='Minimum risk terms required per paragraph (default: 1)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # -------------------------------------------------------------------------
    # Load input
    # -------------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("RISK TERM PARAGRAPH FILTER")
    logger.info("=" * 70)

    logger.info(f"\nInput:  {args.input}")
    logger.info(f"Output: {args.output}")
    if args.categories:
        logger.info(f"Categories: {args.categories}")
    else:
        logger.info("Categories: all")
    logger.info(f"Min terms per paragraph: {args.min_terms}")

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    logger.info("\nLoading corpus...")
    df = pd.read_parquet(args.input)
    logger.info(f"  Loaded {len(df):,} sentences from {df['doc_id'].nunique()} documents")

    # Validate required columns
    required_cols = ['doc_id', 'paragraph_id', 'sentence_text']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(
            f"Missing required columns: {missing}. "
            "Make sure to run preprocessing_bert.py with paragraph_id support."
        )
        return 1

    # -------------------------------------------------------------------------
    # Build risk pattern
    # -------------------------------------------------------------------------
    logger.info("\nBuilding risk term pattern...")
    try:
        risk_pattern = build_risk_pattern(args.categories)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Count total terms in pattern for logging
    if args.categories:
        n_terms = sum(len(RISK_DICTIONARY.get(c, [])) for c in args.categories)
    else:
        n_terms = sum(len(terms) for terms in RISK_DICTIONARY.values())
    logger.info(f"  Pattern includes {n_terms} risk terms")

    # -------------------------------------------------------------------------
    # Filter
    # -------------------------------------------------------------------------
    logger.info("\nFiltering by paragraph risk content...")

    # Statistics before
    n_paragraphs_before = df.groupby(['doc_id', 'paragraph_id']).ngroups
    n_sentences_before = len(df)
    n_docs_before = df['doc_id'].nunique()

    df_filtered = filter_by_risk_paragraphs(df, risk_pattern, args.min_terms)

    # Statistics after
    n_paragraphs_after = df_filtered.groupby(['doc_id', 'paragraph_id']).ngroups
    n_sentences_after = len(df_filtered)
    n_docs_after = df_filtered['doc_id'].nunique()

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    logger.info("\nSaving filtered corpus...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(args.output, index=False)
    logger.info(f"  Saved to: {args.output}")

    # -------------------------------------------------------------------------
    # Save filter report
    # -------------------------------------------------------------------------
    report = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'input_file': str(args.input),
            'output_file': str(args.output),
            'categories': args.categories or list(RISK_DICTIONARY.keys()),
            'min_terms_per_paragraph': args.min_terms,
            'n_risk_terms_in_pattern': n_terms,
        },
        'statistics': {
            'before': {
                'documents': n_docs_before,
                'paragraphs': n_paragraphs_before,
                'sentences': n_sentences_before,
            },
            'after': {
                'documents': n_docs_after,
                'paragraphs': n_paragraphs_after,
                'sentences': n_sentences_after,
            },
            'retention': {
                'documents_pct': round(n_docs_after / n_docs_before * 100, 1)
                    if n_docs_before > 0 else 0,
                'paragraphs_pct': round(n_paragraphs_after / n_paragraphs_before * 100, 1)
                    if n_paragraphs_before > 0 else 0,
                'sentences_pct': round(n_sentences_after / n_sentences_before * 100, 1)
                    if n_sentences_before > 0 else 0,
            },
        },
    }

    report_path = args.output.with_suffix('.filter_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved filter report to: {report_path}")

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 70)

    logger.info("\nStatistics:")
    logger.info(f"                   Before     After     Retained")
    logger.info(f"  Documents:    {n_docs_before:>8,}  {n_docs_after:>8,}  "
                f"({report['statistics']['retention']['documents_pct']}%)")
    logger.info(f"  Paragraphs:   {n_paragraphs_before:>8,}  {n_paragraphs_after:>8,}  "
                f"({report['statistics']['retention']['paragraphs_pct']}%)")
    logger.info(f"  Sentences:    {n_sentences_before:>8,}  {n_sentences_after:>8,}  "
                f"({report['statistics']['retention']['sentences_pct']}%)")

    logger.info("\nNext steps:")
    logger.info(f"  1. Run stratified_sample.py on the filtered corpus:")
    logger.info(f"     python scripts/04_sampling/stratified_sample.py \\")
    logger.info(f"         --input {args.output} \\")
    logger.info(f"         --output results/sampling/ \\")
    logger.info(f"         --n-sentences 500 --seed 42")

    return 0


if __name__ == '__main__':
    sys.exit(main())
