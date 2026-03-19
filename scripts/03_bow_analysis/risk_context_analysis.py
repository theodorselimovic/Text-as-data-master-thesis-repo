#!/usr/bin/env python3
"""
Risk Context Analysis Script (Token-Based)

Analyzes RSA documents for:
1. Risk terms from dictionary (by category)
2. Risk qualifications (sannolikhet, konsekvens, risk) with 5-level scale
3. Context for unknown qualifications

Key features:
- Uses pre-stemmed corpus for fast token matching (no regex)
- Sentence-level qualifier matching
- Groups qualifications into 5 levels: very_low, low, medium, high, very_high

Output: CSV with per-document counts + comprehensive report

Usage:
    python risk_context_analysis.py \
        --corpus data/processed/bow_corpus_stemmed.parquet \
        --output results/01_bow_analysis/context/
"""

import random
import pandas as pd
import json
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import logging
import sys

from nltk.stem.snowball import SnowballStemmer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Target words for qualification analysis (will be stemmed)
TARGET_WORDS_RAW = {
    'sannolikhet': ['sannolikhet', 'sannolikheten', 'sannolikhets', 'trolig', 'troligt', 'troliga'],
    'konsekvens': ['konsekvens', 'konsekvensen', 'konsekvenser'],
    'risk': ['risk', 'risken', 'risker', 'riskens']
}

# Boilerplate tokens to exclude (stemmed forms)
# These will be checked as n-grams in tokens
RSA_BOILERPLATE_STEMS = None  # Will be initialized after stemmer

# =============================================================================
# 5-LEVEL QUALIFICATION MAPPING (raw forms, will be stemmed)
# =============================================================================

QUALIFICATION_MAPPING_RAW = {
    'sannolikhet': {
        'very_low': ['mycket låg', 'mycket liten', 'osannolik', 'sällsynt'],
        'low': ['låg', 'liten', 'små'],
        'medium': ['medelhög', 'mellan', 'möjlig'],
        'high': ['hög', 'stor'],
        'very_high': ['mycket hög', 'mycket stora', 'stora'],
        'change': ['minska', 'minskar', 'minskad', 'minskande', 'öka', 'ökar', 'ökad', 'ökande',
                   'förändras', 'förändrad', 'stiger', 'stigande', 'sjunker', 'sjunkande'],
        'uncertainty': ['svår', 'svårt', 'svåra', 'lätt', 'lätta', 'lättare',
                        'osäker', 'osäkert', 'osäkra', 'osäkerhet',
                        'säker', 'säkert', 'säkra', 'säkerhet'],
        'acceptability': ['acceptabel', 'oacceptabel']
    },
    'konsekvens': {
        'very_low': ['mycket begränsade', 'mycket liten', 'försumbara'],
        'low': ['begränsade', 'lindriga', 'liten', 'små'],
        'medium': ['kännbara', 'måttliga', 'direkta'],
        'high': ['allvarlig', 'allvarliga', 'betydande', 'stor', 'stora', 'omfattande', 'svåra'],
        'very_high': ['mycket allvarlig', 'mycket allvarliga', 'mycket stora', 'mycket omfattande', 'katastrofal', 'katastrofala'],
        'change': ['minska', 'minskar', 'minskad', 'minskande', 'öka', 'ökar', 'ökad', 'ökande',
                   'förändras', 'förändrad', 'stiger', 'stigande', 'sjunker', 'sjunkande'],
        'uncertainty': ['osäker', 'osäkert', 'osäkra', 'osäkerhet',
                        'säker', 'säkert', 'säkra', 'säkerhet'],
        'acceptability': ['acceptabel', 'oacceptabel']
    },
    'risk': {
        'very_low': ['mycket låg', 'mycket liten'],
        'low': ['låg', 'liten', 'små'],
        'medium': ['medelhög', 'mellan'],
        'high': ['hög', 'stor', 'stora', 'omfattande'],
        'very_high': ['mycket hög', 'mycket stora', 'mycket omfattande'],
        'change': ['minska', 'minskar', 'minskad', 'minskande', 'öka', 'ökar', 'ökad', 'ökande',
                   'förändras', 'förändrad', 'stiger', 'stigande', 'sjunker', 'sjunkande'],
        'uncertainty': ['osäker', 'osäkert', 'osäkra', 'osäkerhet',
                        'säker', 'säkert', 'säkra', 'säkerhet'],
        'acceptability': ['acceptabel', 'oacceptabel', 'tolerabel', 'intolerabel']
    }
}

# Special uncertainty patterns: if BOTH stems appear in sentence, it's uncertainty
# e.g., "svår" + "bedöm" = "svårt att bedöma"
SPECIAL_UNCERTAINTY_STEMS = {
    'konsekvens': [('svår', 'bedöm')],
    'risk': [('svår', 'bedöm')],
}

# =============================================================================
# STEMMING INITIALIZATION
# =============================================================================

def initialize_stemmed_dictionaries():
    """
    Stem all dictionaries once at startup.

    Returns:
        tuple: (stemmed_targets, stemmed_qualifications, stem_to_level, stem_to_raw)
    """
    stemmer = SnowballStemmer('swedish')

    # Stem target words
    stemmed_targets = {}
    for concept, words in TARGET_WORDS_RAW.items():
        stemmed_targets[concept] = set(stemmer.stem(w) for w in words)

    # Stem qualification mapping and build reverse lookups
    stemmed_qualifications = {}  # concept -> level -> set of stems
    stem_to_level = {}  # concept -> stem -> level
    stem_to_raw = {}  # concept -> stem -> original term (for reporting)

    for concept, level_map in QUALIFICATION_MAPPING_RAW.items():
        stemmed_qualifications[concept] = {}
        stem_to_level[concept] = {}
        stem_to_raw[concept] = {}

        for level, terms in level_map.items():
            stemmed_terms = set()
            for term in terms:
                # For multi-word terms, stem each word and join
                words = term.lower().split()
                stemmed = '_'.join(stemmer.stem(w) for w in words)
                stemmed_terms.add(stemmed)
                stem_to_level[concept][stemmed] = level
                stem_to_raw[concept][stemmed] = term

                # Also add individual stems for single-word matching
                if len(words) == 1:
                    stem = stemmer.stem(words[0])
                    if stem not in stem_to_level[concept]:
                        stem_to_level[concept][stem] = level
                        stem_to_raw[concept][stem] = term

            stemmed_qualifications[concept][level] = stemmed_terms

    # Build flat set of all qualification stems per concept
    all_qual_stems = {}
    for concept in stemmed_qualifications:
        all_stems = set()
        for level, stems in stemmed_qualifications[concept].items():
            all_stems.update(stems)
        all_qual_stems[concept] = all_stems

    # Stem special uncertainty patterns
    stemmed_special = {}
    for concept, patterns in SPECIAL_UNCERTAINTY_STEMS.items():
        stemmed_special[concept] = [
            (stemmer.stem(p1), stemmer.stem(p2)) for p1, p2 in patterns
        ]

    # Boilerplate stems to exclude
    boilerplate_phrases = ['risk och sårbarhetsanalys', 'risk- och sårbarhetsanalys', 'rsa']
    boilerplate_stems = set()
    for phrase in boilerplate_phrases:
        words = phrase.replace('-', ' ').split()
        for w in words:
            boilerplate_stems.add(stemmer.stem(w))

    return {
        'targets': stemmed_targets,
        'qualifications': stemmed_qualifications,
        'all_qual_stems': all_qual_stems,
        'stem_to_level': stem_to_level,
        'stem_to_raw': stem_to_raw,
        'special_uncertainty': stemmed_special,
        'boilerplate': boilerplate_stems,
        'stemmer': stemmer,
    }


# =============================================================================
# TOKEN-BASED ANALYSIS
# =============================================================================

def check_special_uncertainty(tokens_set: set, concept: str, special_patterns: dict) -> bool:
    """Check if special uncertainty pattern (e.g., svår + bedöm) is present."""
    if concept not in special_patterns:
        return False

    for stem1, stem2 in special_patterns[concept]:
        # Check if both stems appear in any token
        has_stem1 = any(stem1 in t for t in tokens_set)
        has_stem2 = any(stem2 in t for t in tokens_set)
        if has_stem1 and has_stem2:
            return True

    return False


def find_qualification(tokens: list, concept: str, dicts: dict) -> tuple:
    """
    Find qualification level in token list.

    Matches multi-word qualifications first (e.g., "mycket_hög"),
    then single-word qualifications.

    Returns:
        (stem, level, raw_term) or (None, None, None)
    """
    tokens_set = set(tokens)

    # Check special uncertainty patterns first
    if check_special_uncertainty(tokens_set, concept, dicts['special_uncertainty']):
        return 'svår_bedöm', 'uncertainty', 'svårt att bedöma'

    stem_to_level = dicts['stem_to_level'][concept]
    stem_to_raw = dicts['stem_to_raw'][concept]

    # Sort stems by length (longest first) to match "mycket_hög" before "hög"
    sorted_stems = sorted(stem_to_level.keys(), key=len, reverse=True)

    # Check for n-gram matches first (underscored compounds)
    for stem in sorted_stems:
        if '_' in stem:
            # This is a multi-word qualifier - check if it exists as n-gram
            if stem in tokens_set:
                return stem, stem_to_level[stem], stem_to_raw[stem]

    # Check for single-word matches
    for stem in sorted_stems:
        if '_' not in stem:
            if stem in tokens_set:
                return stem, stem_to_level[stem], stem_to_raw[stem]

    return None, None, None


def analyze_sentence_tokens(tokens: list, dicts: dict) -> dict:
    """
    Analyze a single sentence (token list) for qualifications.

    Returns:
        dict with results for each concept (sannolikhet, konsekvens, risk)
    """
    tokens_set = set(tokens)
    results = {}

    for concept, target_stems in dicts['targets'].items():
        # Check if any target word stem is in the sentence
        has_target = bool(target_stems & tokens_set)

        if not has_target:
            results[concept] = None
            continue

        # Find qualification
        stem, level, raw_term = find_qualification(tokens, concept, dicts)

        results[concept] = {
            'found': True,
            'level': level if level else 'UNKNOWN',
            'stem': stem,
            'raw_term': raw_term,
        }

    return results


def is_boilerplate_sentence(tokens: list, boilerplate_stems: set) -> bool:
    """Check if sentence is likely boilerplate (RSA title, etc.)."""
    tokens_set = set(tokens)
    # If sentence contains multiple boilerplate stems, skip it
    overlap = tokens_set & boilerplate_stems
    return len(overlap) >= 2


# =============================================================================
# CORPUS ANALYSIS
# =============================================================================

def analyze_corpus(
    df: pd.DataFrame,
    dicts: dict,
    metadata_columns: list = None,
) -> tuple:
    """
    Analyze entire corpus using token matching.

    Parameters
    ----------
    df : pd.DataFrame
        Stemmed corpus with 'tokens' column (sentence-level).
    dicts : dict
        Stemmed dictionaries from initialize_stemmed_dictionaries().
    metadata_columns : list
        Metadata columns to include in output.

    Returns
    -------
    tuple: (results_df, aggregated_stats)
    """
    doc_results = []

    # Aggregate statistics
    total_level_counts = defaultdict(Counter)
    total_raw_counts = defaultdict(Counter)
    unknown_examples = defaultdict(list)

    # Actor-level statistics
    actor_level_counts = defaultdict(lambda: defaultdict(Counter))
    actor_doc_counts = Counter()

    total_rows = len(df)
    report_interval = max(1000, total_rows // 20)

    # Group by document for document-level results
    doc_groups = df.groupby('doc_id')
    total_docs = len(doc_groups)

    logger.info(f"Analyzing {total_docs} documents ({total_rows} sentences)...")

    for doc_idx, (doc_id, doc_df) in enumerate(doc_groups):
        if doc_idx % max(1, total_docs // 20) == 0:
            logger.info(f"  Processing document {doc_idx}/{total_docs} ({doc_idx/total_docs*100:.1f}%)...")

        # Get metadata from first row
        first_row = doc_df.iloc[0]
        actor = first_row.get('actor_type', first_row.get('actor', 'unknown'))
        actor_doc_counts[actor] += 1

        # Initialize document counts
        doc_level_counts = defaultdict(Counter)

        for _, row in doc_df.iterrows():
            tokens = row.get('tokens', [])

            # Handle various token formats
            if tokens is None:
                continue
            if isinstance(tokens, str):
                tokens = tokens.split()
            if not isinstance(tokens, list):
                tokens = list(tokens)

            # Skip boilerplate
            if is_boilerplate_sentence(tokens, dicts['boilerplate']):
                continue

            # Analyze sentence
            sentence_results = analyze_sentence_tokens(tokens, dicts)

            for concept, result in sentence_results.items():
                if result is None:
                    continue

                level = result['level']
                doc_level_counts[concept][level] += 1
                total_level_counts[concept][level] += 1
                actor_level_counts[actor][concept][level] += 1

                if result['raw_term']:
                    total_raw_counts[concept][result['raw_term']] += 1

                # Collect unknown examples
                if level == 'UNKNOWN' and len(unknown_examples[concept]) < 100:
                    unknown_examples[concept].append({
                        'doc_id': doc_id,
                        'tokens': ' '.join(tokens[:50]),  # First 50 tokens
                    })

        # Build result row
        result_row = {'doc_id': doc_id}

        # Add metadata
        if metadata_columns:
            for col in metadata_columns:
                if col in first_row:
                    result_row[col] = first_row[col]

        # Add qualification counts
        for concept in ['sannolikhet', 'konsekvens', 'risk']:
            counts = doc_level_counts[concept]
            result_row[f'{concept}_total'] = sum(counts.values())

            for level in ['very_low', 'low', 'medium', 'high', 'very_high',
                         'change', 'uncertainty', 'acceptability', 'UNKNOWN']:
                result_row[f'{concept}_{level}'] = counts.get(level, 0)

        doc_results.append(result_row)

    results_df = pd.DataFrame(doc_results)

    # Build aggregated stats
    aggregated = {
        'total_documents': total_docs,
        'total_sentences': total_rows,
        'qualifications': {
            concept: {
                'total': sum(level_counts.values()),
                'by_level': dict(level_counts),
                'raw_distribution': dict(total_raw_counts[concept]),
                'unknown_examples': unknown_examples[concept]
            }
            for concept, level_counts in total_level_counts.items()
        },
        'level_mapping': QUALIFICATION_MAPPING_RAW,
        'by_actor': {
            actor: {
                'doc_count': actor_doc_counts[actor],
                'qualifications': {
                    concept: dict(actor_level_counts[actor][concept])
                    for concept in ['sannolikhet', 'konsekvens', 'risk']
                    if actor_level_counts[actor][concept]
                }
            }
            for actor in actor_doc_counts.keys()
        }
    }

    return results_df, aggregated


# =============================================================================
# VISUALIZATIONS
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ACTOR_LABELS = {
    'kommun': 'Municipality',
    'lansstyrelse': 'Prefecture',
    'länsstyrelse': 'Prefecture',
    'MCF': 'MSB',
}

# Actor colors (consistent across all visualizations)
ACTOR_COLORS = {
    'kommun': '#e41a1c',        # Red
    'lansstyrelse': '#377eb8',  # Blue
    'MCF': '#4daf4a',           # Green
}

# Fixed order: Municipality, Prefecture, MSB (left to right)
ACTOR_ORDER = ['kommun', 'lansstyrelse', 'MCF']

LEVEL_ORDER = ['very_low', 'low', 'medium', 'high', 'very_high']
LEVEL_LABELS = {
    'very_low': 'Very Low',
    'low': 'Low',
    'medium': 'Medium',
    'high': 'High',
    'very_high': 'Very High',
}
LEVEL_COLORS = {
    'very_low': '#2ecc71',
    'low': '#a8e6cf',
    'medium': '#ffd93d',
    'high': '#ff6b6b',
    'very_high': '#c0392b',
}


def create_qualifications_over_time(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Line chart showing qualification levels over time for each concept."""
    if 'year' not in results_df.columns:
        return

    df = results_df.copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    # Only show meaningful levels (exclude UNKNOWN, acceptability)
    severity_levels = ['very_low', 'low', 'medium', 'high', 'very_high']
    concepts = ['sannolikhet', 'konsekvens', 'risk']
    concept_labels = {
        'sannolikhet': 'Probability',
        'konsekvens': 'Consequence',
        'risk': 'Risk',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, concept in zip(axes, concepts):
        # Aggregate by year: average mentions per document
        yearly_data = []
        for year in sorted(df['year'].unique()):
            year_df = df[df['year'] == year]
            n_docs = len(year_df)
            row = {'year': year, 'n_docs': n_docs}
            for level in severity_levels:
                col = f'{concept}_{level}'
                if col in year_df.columns:
                    row[level] = year_df[col].sum() / n_docs  # Average per doc
            yearly_data.append(row)

        yearly_df = pd.DataFrame(yearly_data)
        if yearly_df.empty:
            continue

        # Plot each level
        for level in severity_levels:
            if level in yearly_df.columns:
                ax.plot(yearly_df['year'], yearly_df[level],
                        marker='o', markersize=4, linewidth=1.5,
                        label=LEVEL_LABELS[level], color=LEVEL_COLORS[level])

        ax.set_title(concept_labels[concept], fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Avg. mentions per document')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_xlim(yearly_df['year'].min() - 0.5, yearly_df['year'].max() + 0.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.suptitle('Qualification Levels Over Time (Per Document Average)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'qualification_over_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'qualification_over_time.pdf', bbox_inches='tight')
    plt.close()
    logger.info("  Saved: qualification_over_time.png/pdf")


def create_high_severity_over_time(results_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Line chart showing high+very_high severity share over time by actor.
    Normalized to share of all qualified mentions (excluding UNKNOWN).
    """
    if 'year' not in results_df.columns:
        return

    df = results_df.copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    actor_col = 'actor_type' if 'actor_type' in df.columns else 'actor'
    if actor_col not in df.columns:
        return

    concepts = ['sannolikhet', 'konsekvens', 'risk']
    concept_labels = {
        'sannolikhet': 'Probability',
        'konsekvens': 'Consequence',
        'risk': 'Risk',
    }

    actor_colors = {
        'kommun': '#e41a1c',        # Red
        'lansstyrelse': '#377eb8',  # Blue
        'länsstyrelse': '#377eb8',  # Blue
        'MCF': '#4daf4a',           # Green
    }
    actor_labels = {
        'kommun': 'Municipality',
        'lansstyrelse': 'Prefecture',
        'länsstyrelse': 'Prefecture',
        'MCF': 'MCF',
    }

    severity_levels = ['very_low', 'low', 'medium', 'high', 'very_high']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, concept in zip(axes, concepts):
        for actor in df[actor_col].unique():
            if actor == 'unknown':
                continue

            actor_df = df[df[actor_col] == actor]
            yearly_data = []

            for year in sorted(actor_df['year'].unique()):
                year_df = actor_df[actor_df['year'] == year]

                # Sum high + very_high
                high_col = f'{concept}_high'
                vhigh_col = f'{concept}_very_high'
                high_sum = year_df[high_col].sum() if high_col in year_df.columns else 0
                vhigh_sum = year_df[vhigh_col].sum() if vhigh_col in year_df.columns else 0
                high_total = high_sum + vhigh_sum

                # Sum all severity levels (excluding UNKNOWN)
                total_qualified = sum(
                    year_df[f'{concept}_{level}'].sum()
                    for level in severity_levels
                    if f'{concept}_{level}' in year_df.columns
                )

                if total_qualified > 0:
                    yearly_data.append({
                        'year': year,
                        'high_share': high_total / total_qualified * 100
                    })

            if yearly_data:
                yearly_df = pd.DataFrame(yearly_data)
                ax.plot(yearly_df['year'], yearly_df['high_share'],
                        marker='o', markersize=4, linewidth=1.5,
                        label=actor_labels.get(actor, actor),
                        color=actor_colors.get(actor, '#999999'))

        ax.set_title(concept_labels[concept], fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('High+Very High share (%)')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.suptitle('High Severity Share Over Time by Actor',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'high_severity_over_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'high_severity_over_time.pdf', bbox_inches='tight')
    plt.close()
    logger.info("  Saved: high_severity_over_time.png/pdf")


def create_visualizations(aggregated: dict, output_dir: Path, results_df: pd.DataFrame = None) -> None:
    """Create all visualizations for risk context analysis."""
    output_dir = Path(output_dir)
    plt.style.use('seaborn-v0_8-whitegrid')

    create_qualification_distribution(aggregated, output_dir)

    if 'by_actor' in aggregated and len(aggregated['by_actor']) > 1:
        # Filter out 'unknown' actor if others exist
        actors = [a for a in aggregated['by_actor'].keys() if a != 'unknown']
        if len(actors) > 1:
            create_actor_comparison(aggregated, output_dir)

    # Time-based visualizations
    if results_df is not None and 'year' in results_df.columns:
        create_qualifications_over_time(results_df, output_dir)
        create_high_severity_over_time(results_df, output_dir)


def create_qualification_distribution(aggregated: dict, output_dir: Path) -> None:
    """Bar chart showing qualification counts by level for each concept."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    concepts = ['sannolikhet', 'konsekvens', 'risk']
    concept_labels = {
        'sannolikhet': 'Probability (Sannolikhet)',
        'konsekvens': 'Consequence (Konsekvens)',
        'risk': 'Risk',
    }

    for ax, concept in zip(axes, concepts):
        if concept not in aggregated['qualifications']:
            continue

        data = aggregated['qualifications'][concept]['by_level']
        counts = [data.get(level, 0) for level in LEVEL_ORDER]
        colors = [LEVEL_COLORS[level] for level in LEVEL_ORDER]
        labels = [LEVEL_LABELS[level] for level in LEVEL_ORDER]

        bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=0.5)

        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=9)

        ax.set_title(concept_labels[concept], fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xlabel('Qualification Level')

        total = sum(data.values())
        unknown = data.get('UNKNOWN', 0)
        ax.annotate(f'Total: {total}\nUnknown: {unknown} ({unknown/total*100:.1f}%)' if total > 0 else '',
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Qualification Distribution by Concept (5-Level Scale)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'qualification_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'qualification_distribution.pdf', bbox_inches='tight')
    plt.close()
    logger.info("  Saved: qualification_distribution.png/pdf")


def create_actor_comparison(aggregated: dict, output_dir: Path) -> None:
    """Grouped bar chart comparing qualification levels across actors."""
    available_actors = [a for a in aggregated['by_actor'].keys() if a != 'unknown']
    # Use fixed order: Municipality, Prefecture, MSB
    actors = [a for a in ACTOR_ORDER if a in available_actors]
    if len(actors) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    concepts = ['sannolikhet', 'konsekvens', 'risk']
    concept_labels = {
        'sannolikhet': 'Probability',
        'konsekvens': 'Consequence',
        'risk': 'Risk',
    }

    for ax, concept in zip(axes, concepts):
        x = np.arange(len(LEVEL_ORDER))
        width = 0.25
        offsets = np.linspace(-width, width, len(actors))

        for i, actor in enumerate(actors):
            actor_data = aggregated['by_actor'][actor].get('qualifications', {}).get(concept, {})
            total = sum(actor_data.values()) if actor_data else 0

            if total == 0:
                continue

            pcts = [(actor_data.get(level, 0) / total * 100) for level in LEVEL_ORDER]
            actor_label = ACTOR_LABELS.get(actor, actor)
            color = ACTOR_COLORS.get(actor, '#999999')
            ax.bar(x + offsets[i], pcts, width * 0.9, label=actor_label, color=color, alpha=0.8)

        ax.set_title(concept_labels[concept], fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage')
        ax.set_xlabel('Qualification Level')
        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_LABELS[l] for l in LEVEL_ORDER], rotation=45, ha='right')
        ax.legend(fontsize=8)

    plt.suptitle('Qualification Distribution by Actor Type (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'qualification_by_actor.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'qualification_by_actor.pdf', bbox_inches='tight')
    plt.close()
    logger.info("  Saved: qualification_by_actor.png/pdf")


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(results_df: pd.DataFrame, aggregated: dict, output_dir: Path):
    """Save all results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save document-level results
    csv_path = output_dir / 'risk_context_analysis_by_document.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Saved document results: {csv_path}")

    # Save aggregated results (JSON)
    json_path = output_dir / 'risk_context_analysis_aggregated.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved aggregated results: {json_path}")

    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(aggregated, output_dir, results_df)

    # Generate report
    generate_report(aggregated, output_dir)


def generate_report(aggregated: dict, output_dir: Path):
    """Generate comprehensive text report."""
    report = []
    report.append("=" * 80)
    report.append("RISK CONTEXT ANALYSIS - COMPREHENSIVE REPORT")
    report.append("=" * 80)

    report.append(f"\nTotal documents analyzed: {aggregated['total_documents']}")
    report.append(f"Total sentences analyzed: {aggregated['total_sentences']}")

    report.append("\n" + "-" * 40)
    report.append("METHODOLOGY NOTES:")
    report.append("-" * 40)
    report.append("- Token-based matching on pre-stemmed corpus (fast)")
    report.append("- Sentence-level qualifier matching")
    report.append("- Qualifications grouped into 5 severity levels: very_low -> very_high")
    report.append("- Additional categories: 'change', 'uncertainty', 'acceptability'")

    # Qualifications
    report.append("\n" + "=" * 80)
    report.append("QUALIFICATION ANALYSIS (5-level scale)")
    report.append("=" * 80)

    for concept in ['sannolikhet', 'konsekvens', 'risk']:
        if concept not in aggregated['qualifications']:
            continue
        data = aggregated['qualifications'][concept]

        report.append(f"\n{concept.upper()}:")
        report.append(f"  Total mentions: {data['total']}")

        report.append(f"\n  Distribution by level:")
        level_order = ['very_low', 'low', 'medium', 'high', 'very_high',
                       'change', 'uncertainty', 'acceptability', 'UNKNOWN']
        for level in level_order:
            count = data['by_level'].get(level, 0)
            if count > 0:
                pct = (count / data['total'] * 100) if data['total'] > 0 else 0
                report.append(f"    {level:15s}: {count:5d} ({pct:5.1f}%)")

        # Raw term breakdown
        if data['raw_distribution']:
            report.append(f"\n  Top raw terms:")
            sorted_terms = sorted(data['raw_distribution'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]
            for term, count in sorted_terms:
                report.append(f"    {term:20s}: {count:5d}")

        # Unknown examples
        unknown_count = data['by_level'].get('UNKNOWN', 0)
        if unknown_count > 0 and data['unknown_examples']:
            report.append(f"\n  Sample unknown contexts ({unknown_count} total):")
            samples = random.sample(data['unknown_examples'],
                                    min(5, len(data['unknown_examples'])))
            for ex in samples:
                report.append(f"    Doc {ex['doc_id']}: {ex['tokens'][:80]}...")

    # Actor comparison
    if 'by_actor' in aggregated and len(aggregated['by_actor']) > 1:
        report.append("\n" + "=" * 80)
        report.append("ACTOR COMPARISON")
        report.append("=" * 80)

        actors = [a for a in aggregated['by_actor'].keys() if a != 'unknown']
        report.append(f"\nActors: {', '.join(actors)}")

        for actor in actors:
            actor_data = aggregated['by_actor'][actor]
            report.append(f"\n{ACTOR_LABELS.get(actor, actor)} ({actor_data['doc_count']} documents):")

            for concept in ['sannolikhet', 'konsekvens', 'risk']:
                qual_data = actor_data.get('qualifications', {}).get(concept, {})
                if qual_data:
                    total = sum(qual_data.values())
                    high_pct = ((qual_data.get('high', 0) + qual_data.get('very_high', 0)) / total * 100) if total > 0 else 0
                    low_pct = ((qual_data.get('low', 0) + qual_data.get('very_low', 0)) / total * 100) if total > 0 else 0
                    report.append(f"  {concept}: {total} mentions (high+very_high: {high_pct:.1f}%, low+very_low: {low_pct:.1f}%)")

    # Save report
    report_text = '\n'.join(report)
    report_path = output_dir / 'risk_context_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"Saved report: {report_path}")

    print("\n" + report_text)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Risk context analysis using token matching on stemmed corpus'
    )

    parser.add_argument(
        '--corpus', '-c',
        type=Path,
        required=True,
        help='Path to stemmed corpus parquet (e.g., bow_corpus_stemmed.parquet)'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./results/01_bow_analysis/context'),
        help='Output directory'
    )

    parser.add_argument(
        '--metadata',
        nargs='+',
        default=['doc_id', 'actor_type', 'year', 'wave'],
        help='Metadata columns to include in output'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RISK CONTEXT ANALYSIS (Token-Based)")
    print("=" * 80)

    # Initialize stemmed dictionaries
    logger.info("Initializing stemmed dictionaries...")
    dicts = initialize_stemmed_dictionaries()
    logger.info(f"  Target stems: {sum(len(v) for v in dicts['targets'].values())} total")
    logger.info(f"  Qualification stems: {sum(len(v) for v in dicts['all_qual_stems'].values())} total")

    # Load corpus
    logger.info(f"\nLoading corpus: {args.corpus}")
    df = pd.read_parquet(args.corpus)
    logger.info(f"  Loaded {len(df)} sentences from {df['doc_id'].nunique()} documents")

    if 'tokens' not in df.columns:
        logger.error("Corpus must have 'tokens' column (use bow_corpus_stemmed.parquet)")
        return 1

    # Show actor distribution
    actor_col = 'actor_type' if 'actor_type' in df.columns else 'actor'
    if actor_col in df.columns:
        actor_dist = df.groupby(actor_col)['doc_id'].nunique()
        logger.info(f"  Actors: {actor_dist.to_dict()}")

    # Analyze
    logger.info("\nAnalyzing corpus...")
    results_df, aggregated = analyze_corpus(df, dicts, args.metadata)

    # Save
    logger.info("\nSaving results...")
    save_results(results_df, aggregated, args.output)

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files in {args.output}:")
    print("  - risk_context_analysis_by_document.csv")
    print("  - risk_context_analysis_aggregated.json")
    print("  - risk_context_analysis_report.txt")
    print("  - qualification_distribution.png/pdf")
    print("  - qualification_by_actor.png/pdf")
    print("  - qualification_over_time.png/pdf")
    print("  - high_severity_over_time.png/pdf")

    return 0


if __name__ == '__main__':
    sys.exit(main())
