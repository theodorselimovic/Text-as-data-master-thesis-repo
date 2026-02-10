#!/usr/bin/env python3
"""
Risk Context Analysis Script

Analyzes RSA documents for:
1. Risk terms from dictionary (by category)
2. Risk qualifications (sannolikhet, konsekvens, risk) with 5-level scale
3. Context for unknown qualifications

Key features:
- Filters out "risk- och sårbarhetsanalys" boilerplate
- Only counts "risk" when paired with a qualifier (adjective)
- Groups qualifications into 5 levels: very_low, low, medium, high, very_high

Output: CSV with per-document counts + comprehensive report

Usage:
    python risk_context_analysis.py \
        --texts your_texts.parquet \
        --text-column text \
        --metadata doc_id municipality year \
        --output ./results/
"""

import re
import random
import pandas as pd
import json
from pathlib import Path
from collections import Counter, defaultdict
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

# Phrases to exclude when counting "risk" (boilerplate)
RSA_EXCLUSION_PATTERNS = [
    r'risk-\s*och\s*sårbarhetsanalys\w*',
    r'rsa\b',
    r'risk-\s*och\s*sårbarhets\w*',
    r'risk\s*och\s*sårbarhetsanalys\w*',
    r'risk-\s*och\s*sårbarhetsbedömning\w*',
]

# Target words for qualification analysis
TARGET_WORDS = {
    'sannolikhet': ['sannolikhet', 'sannolikheten', 'sannolikhets', 'trolig'],
    'konsekvens': ['konsekvens', 'konsekvensen', 'konsekvenser'],
    'risk': ['risk', 'risken', 'risker', 'riskens']
}

# =============================================================================
# 5-LEVEL QUALIFICATION MAPPING
# =============================================================================

# Map raw qualifications to 5-level scale
# Each category maps terms to: very_low, low, medium, high, very_high
# Additional categories: change (increase/decrease), uncertainty, other (acceptability)

QUALIFICATION_MAPPING = {
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
        'acceptability': ['acceptabel', 'oacceptabel']  # acceptability, not level
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
        # Note: 'svårt att bedöma' handled specially - see SPECIAL_UNCERTAINTY_PATTERNS
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
        # Note: 'svårt att bedöma' handled specially - see SPECIAL_UNCERTAINTY_PATTERNS
        'acceptability': ['acceptabel', 'oacceptabel', 'tolerabel', 'intolerabel']  # acceptability
    }
}

# Special patterns for uncertainty that require multiple words
# For konsekvens and risk: "svårt/svår" + "bedöma" together implies uncertainty
SPECIAL_UNCERTAINTY_PATTERNS = {
    'konsekvens': [
        (r'svår\w*', r'bedöm\w*'),  # svårt att bedöma, svåra att bedöma, etc.
    ],
    'risk': [
        (r'svår\w*', r'bedöm\w*'),  # svårt att bedöma, svåra att bedöma, etc.
    ]
}

# Build flat list of all known qualifications per concept (for detection)
def _build_known_qualifications():
    """Build flat list of known qualifications from mapping."""
    known = {}
    for concept, level_map in QUALIFICATION_MAPPING.items():
        all_quals = []
        for level, terms in level_map.items():
            all_quals.extend(terms)
        known[concept] = all_quals
    return known

KNOWN_QUALIFICATIONS = _build_known_qualifications()

# Build reverse lookup: term -> level
def _build_term_to_level():
    """Build reverse lookup from term to level."""
    reverse = {}
    for concept, level_map in QUALIFICATION_MAPPING.items():
        reverse[concept] = {}
        for level, terms in level_map.items():
            for term in terms:
                reverse[concept][term.lower()] = level
    return reverse

TERM_TO_LEVEL = _build_term_to_level()

# Risk dictionary (unchanged)
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
    ],
    'biologiska_hot': [
        'epidemi', 'epidemier', 'pandemi', 'pandemier',
        'epozooti', 'epizootier',
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
        'säkerhetshot',
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
    ],
    'teknisk_infrastruktur': [
        'strömavbrott', 'elavbrott', 'kraftförsörjning', 'elförsörjning', 'effektbrist',
        'fjärrvärmebrott', 'fjärrvärme', 'värmeförsörjning',
        'vattenläcka', 'vattenläckor', 'vattenförsörjning', 'dricksvatten',
        'avloppsbrott', 'avloppssystem',
        'IT-bortfall', 'it-bortfall', 'IT-avbrott', 'it-avbrott',
        'dataförlust', 'systemfel', 'nätverksavbrott',
        'kommunikationsavbrott', 'teleavbrott', 'telebrott',
        'distributionsstörning', 'logistikavbrott', 'transportavbrott',
        'drivsmedelsbrist', 'bränslebrist', 'försörjningsbrist',
        'livsmedelsförsörjning', 'livsmedelsbrist', 'matförsörjning',
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
}

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def remove_rsa_boilerplate(text: str) -> str:
    """
    Remove 'risk- och sårbarhetsanalys' and related boilerplate from text.
    Returns cleaned text for risk qualification analysis.
    """
    cleaned = text
    for pattern in RSA_EXCLUSION_PATTERNS:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
    return cleaned

# =============================================================================
# RISK TERM COUNTING
# =============================================================================

def count_risk_terms(text: str, risk_dictionary: dict) -> dict:
    """Count occurrences of risk terms."""
    text_lower = text.lower()

    results = {'by_category': {}, 'total': 0}

    for category, terms in risk_dictionary.items():
        category_count = 0

        for term in terms:
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            count = len(re.findall(pattern, text_lower))
            category_count += count

        results['by_category'][category] = category_count
        results['total'] += category_count

    return results

# =============================================================================
# QUALIFICATION ANALYSIS (WITH REQUIRED QUALIFIER FOR RISK)
# =============================================================================

def extract_qualified_contexts(text, target_word, variations, known_quals,
                               max_intervening=5, context_window=4,
                               require_qualifier=False):
    """
    Extract contexts around target words.

    If require_qualifier=True, only return contexts where a known qualifier
    is found within the search window (used for 'risk' to filter boilerplate).
    """
    contexts = []

    # Build pattern
    target_pattern = '|'.join(re.escape(v) for v in variations)

    # Pattern: [words before] TARGET [words after]
    pattern = (
        r'(?:(\S+)\s+){0,' + str(context_window) + r'}' +
        r'\b(' + target_pattern + r')\b' +
        r'(?:\s+(\S+)){0,' + str(max_intervening + context_window) + r'}'
    )

    # Sort qualifiers by length (longest first) for matching
    sorted_quals = sorted(known_quals, key=lambda x: len(x), reverse=True)

    for match in re.finditer(pattern, text, re.IGNORECASE):
        full_match = match.group(0)
        target_found = match.group(2)

        # Split into sections
        target_start = full_match.lower().find(target_found.lower())
        before_target = full_match[:target_start]
        after_target = full_match[target_start + len(target_found):]

        # Get word lists
        before_words = before_target.strip().split()
        after_words = after_target.strip().split()

        # Middle is the intervening words
        middle_words = after_words[:max_intervening] if after_words else []

        ctx = {
            'target': target_found,
            'before': ' '.join(before_words[-context_window:]) if before_words else '',
            'middle': ' '.join(middle_words),
            'after': ' '.join(after_words[max_intervening:max_intervening+context_window]) if len(after_words) > max_intervening else '',
            'full_match': full_match.strip(),
            'position': match.start()
        }

        # If requiring qualifier, check if one exists
        if require_qualifier:
            has_qualifier = False
            combined_text = f"{ctx['before']} {ctx['middle']} {ctx['after']}".lower()
            for qual in sorted_quals:
                if qual.lower() in combined_text:
                    has_qualifier = True
                    break

            if not has_qualifier:
                continue  # Skip this match

        contexts.append(ctx)

    return contexts


def check_special_uncertainty(context_dict, concept):
    """
    Check for special uncertainty patterns that require multiple words.

    For konsekvens: "svårt/svår" + "bedöma" together = uncertainty

    Returns: True if special uncertainty pattern found
    """
    if concept not in SPECIAL_UNCERTAINTY_PATTERNS:
        return False

    combined_text = f"{context_dict.get('before', '')} {context_dict.get('middle', '')} {context_dict.get('after', '')}".lower()

    for pattern_pair in SPECIAL_UNCERTAINTY_PATTERNS[concept]:
        word1_pattern, word2_pattern = pattern_pair
        if re.search(word1_pattern, combined_text) and re.search(word2_pattern, combined_text):
            return True

    return False


def extract_qualification(context_dict, known_qualifications, term_to_level, concept=None):
    """
    Extract qualification from context and map to 5-level scale.

    Returns: (raw_qual, level, section)
    """
    # First check for special uncertainty patterns (e.g., "svårt att bedöma" for konsekvens)
    if concept and check_special_uncertainty(context_dict, concept):
        return 'svårt att bedöma', 'uncertainty', 'special'

    # Sort by length (longest first) to match "mycket hög" before "hög"
    sorted_quals = sorted(known_qualifications, key=lambda x: len(x), reverse=True)

    # Check in order: before, middle, after
    for section in ['before', 'middle', 'after']:
        section_text = context_dict.get(section, '').lower()
        for qual in sorted_quals:
            if qual.lower() in section_text:
                level = term_to_level.get(qual.lower(), 'acceptability')
                return qual, level, section

    return None, None, None


def analyze_qualifications(text, target_words, known_qualifications, term_to_level):
    """
    Analyze qualifications for a text.

    RSA boilerplate is removed before analysis. All concepts are analyzed
    without requiring a qualifier (the qualifier requirement was removed).
    """
    results = {}
    unknown_examples = {}

    for concept, variations in target_words.items():
        # No longer require qualifier for risk - RSA removal handles boilerplate
        contexts = extract_qualified_contexts(
            text, concept, variations,
            known_qualifications[concept],
            require_qualifier=False
        )

        # Count by 5-level categories
        level_counts = Counter()  # very_low, low, medium, high, very_high, other
        raw_counts = Counter()    # Keep raw counts too for reference
        unknown_contexts = []

        for ctx in contexts:
            raw_qual, level, location = extract_qualification(
                ctx, known_qualifications[concept], term_to_level[concept], concept=concept
            )

            if raw_qual and level:
                level_counts[level] += 1
                raw_counts[raw_qual] += 1
            else:
                level_counts['UNKNOWN'] += 1
                raw_counts['UNKNOWN'] += 1
                # Store example (collect more per doc for better random sampling later)
                if len(unknown_contexts) < 10:
                    unknown_contexts.append({
                        'context': ctx['full_match'],
                        'before': ctx['before'],
                        'middle': ctx['middle'],
                        'after': ctx['after']
                    })

        results[concept] = {
            'total': len(contexts),
            'level_counts': dict(level_counts),
            'raw_counts': dict(raw_counts),
            'unknown_examples': unknown_contexts
        }

    return results

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_document(
    text: str,
    risk_dictionary: dict,
    target_words: dict,
    known_qualifications: dict,
    term_to_level: dict
) -> dict:
    """Analyze a single document for both risks and qualifications."""

    # Clean text of RSA boilerplate for qualification analysis
    cleaned_text = remove_rsa_boilerplate(text)

    # Count risk terms (using original text for dictionary)
    risk_counts = count_risk_terms(text, risk_dictionary)

    # Analyze qualifications (using cleaned text)
    qual_results = analyze_qualifications(
        cleaned_text, target_words, known_qualifications, term_to_level
    )

    return {
        'risk_counts': risk_counts,
        'qualifications': qual_results
    }


def analyze_corpus(
    texts_df: pd.DataFrame,
    text_column: str = 'text',
    metadata_columns: list = None
) -> tuple:
    """Analyze entire corpus."""

    doc_results = []

    # Aggregate statistics (overall)
    total_risk_counts = Counter()
    total_level_counts = defaultdict(Counter)
    total_raw_counts = defaultdict(Counter)
    unknown_qual_examples = defaultdict(list)

    # Actor-level statistics
    actor_risk_counts = defaultdict(Counter)  # actor -> category -> count
    actor_level_counts = defaultdict(lambda: defaultdict(Counter))  # actor -> concept -> level -> count
    actor_doc_counts = Counter()  # actor -> num docs

    for idx, row in texts_df.iterrows():
        if idx % 50 == 0:
            print(f"Processing document {idx}/{len(texts_df)}...")

        text = str(row.get(text_column, ''))
        if not text:
            continue

        # Analyze
        analysis = analyze_document(
            text, RISK_DICTIONARY, TARGET_WORDS,
            KNOWN_QUALIFICATIONS, TERM_TO_LEVEL
        )

        # Build result row
        result_row = {}

        # Add metadata
        if metadata_columns:
            for col in metadata_columns:
                if col in row:
                    result_row[col] = row[col]

        result_row['doc_index'] = idx

        # Add risk counts
        result_row['total_risk_mentions'] = analysis['risk_counts']['total']
        for category, count in analysis['risk_counts']['by_category'].items():
            result_row[f'risk_{category}'] = count

        # Add qualification counts (5-level scale)
        for concept, data in analysis['qualifications'].items():
            result_row[f'{concept}_total'] = data['total']

            # Add level counts (5-level scale + change, uncertainty, acceptability)
            for level in ['very_low', 'low', 'medium', 'high', 'very_high', 'change', 'uncertainty', 'acceptability', 'UNKNOWN']:
                count = data['level_counts'].get(level, 0)
                result_row[f'{concept}_{level}'] = count

        doc_results.append(result_row)

        # Get actor for this document
        actor = row.get('actor', 'unknown')
        actor_doc_counts[actor] += 1

        # Aggregate (overall)
        for category, count in analysis['risk_counts']['by_category'].items():
            total_risk_counts[category] += count
            actor_risk_counts[actor][category] += count

        for concept, data in analysis['qualifications'].items():
            for level, count in data['level_counts'].items():
                total_level_counts[concept][level] += count
                actor_level_counts[actor][concept][level] += count
            for raw, count in data['raw_counts'].items():
                total_raw_counts[concept][raw] += count

            # Collect unknown examples (collect up to 100 for random sampling later)
            if data['unknown_examples'] and len(unknown_qual_examples[concept]) < 100:
                for ex in data['unknown_examples']:
                    if len(unknown_qual_examples[concept]) < 100:
                        ex['doc_id'] = row.get('doc_id', f'doc_{idx}')
                        unknown_qual_examples[concept].append(ex)

    results_df = pd.DataFrame(doc_results)

    aggregated = {
        'total_documents': len(texts_df),
        'risk_counts': {
            'total': sum(total_risk_counts.values()),
            'by_category': dict(total_risk_counts)
        },
        'qualifications': {
            concept: {
                'total': sum(level_counts.values()),
                'by_level': dict(level_counts),
                'raw_distribution': dict(total_raw_counts[concept]),
                'unknown_examples': unknown_qual_examples[concept]
            }
            for concept, level_counts in total_level_counts.items()
        },
        'level_mapping': QUALIFICATION_MAPPING,
        # Actor-level statistics
        'by_actor': {
            actor: {
                'doc_count': actor_doc_counts[actor],
                'risk_counts': {
                    'total': sum(actor_risk_counts[actor].values()),
                    'by_category': dict(actor_risk_counts[actor])
                },
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
# OUTPUT
# =============================================================================

def save_results(results_df: pd.DataFrame, aggregated: dict, output_dir: Path):
    """Save all results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save document-level results
    csv_path = output_dir / 'risk_context_analysis_by_document.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nSaved document results: {csv_path}")

    # Save aggregated results (JSON)
    json_path = output_dir / 'risk_context_analysis_aggregated.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"Saved aggregated results: {json_path}")

    # Generate comprehensive report
    report = []
    report.append("="*80)
    report.append("RISK CONTEXT ANALYSIS - COMPREHENSIVE REPORT")
    report.append("="*80)

    report.append(f"\nTotal documents analyzed: {aggregated['total_documents']}")

    report.append("\n" + "-"*40)
    report.append("METHODOLOGY NOTES:")
    report.append("-"*40)
    report.append("- 'risk- och sårbarhetsanalys' phrases excluded from analysis")
    report.append("- Qualifications grouped into 5 severity levels: very_low, low, medium, high, very_high")
    report.append("- Additional categories: 'change' (minska/öka), 'uncertainty', 'acceptability'")
    report.append("- For konsekvens: 'svårt att bedöma' pattern detected for uncertainty")
    report.append("- For sannolikhet: 'svår/lätt' alone indicates uncertainty")

    # RISK COUNTS
    report.append("\n" + "="*80)
    report.append("RISK TERM COUNTS (from dictionary)")
    report.append("="*80)

    report.append(f"\nTotal risk term mentions: {aggregated['risk_counts']['total']}")
    report.append("\nBy category:")
    for category, count in sorted(
        aggregated['risk_counts']['by_category'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        pct = (count / aggregated['risk_counts']['total'] * 100) if aggregated['risk_counts']['total'] > 0 else 0
        report.append(f"  {category:30s}: {count:6d} ({pct:5.1f}%)")

    # QUALIFICATIONS
    report.append("\n" + "="*80)
    report.append("QUALIFICATION ANALYSIS (5-level scale)")
    report.append("="*80)

    for concept in ['sannolikhet', 'konsekvens', 'risk']:
        if concept in aggregated['qualifications']:
            data = aggregated['qualifications'][concept]

            report.append(f"\n{concept.upper()}:")
            report.append(f"  Total qualified mentions: {data['total']}")

            report.append(f"\n  Distribution by level:")
            level_order = ['very_low', 'low', 'medium', 'high', 'very_high', 'change', 'uncertainty', 'acceptability', 'UNKNOWN']
            for level in level_order:
                count = data['by_level'].get(level, 0)
                if count > 0 or level != 'acceptability':
                    pct = (count / data['total'] * 100) if data['total'] > 0 else 0
                    report.append(f"    {level:15s}: {count:5d} ({pct:5.1f}%)")

            # Show raw term counts
            report.append(f"\n  Raw term breakdown:")
            for term, count in sorted(
                data['raw_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if term != 'UNKNOWN':
                    level = TERM_TO_LEVEL[concept].get(term.lower(), 'other')
                    report.append(f"    {term:20s} -> {level:10s}: {count:5d}")

            # Show unknown examples
            unknown_count = data['by_level'].get('UNKNOWN', 0)
            if unknown_count > 0 and data['unknown_examples']:
                unknown_pct = (unknown_count / data['total'] * 100) if data['total'] > 0 else 0

                report.append(f"\n  UNKNOWN QUALIFICATIONS ({unknown_count} total, {unknown_pct:.1f}%):")

                # Randomly sample up to 20 examples
                all_examples = data['unknown_examples']
                if len(all_examples) > 20:
                    sampled_examples = random.sample(all_examples, 20)
                else:
                    sampled_examples = all_examples

                report.append(f"  {len(sampled_examples)} randomly sampled examples to help identify missing terms:")

                for i, ex in enumerate(sampled_examples, 1):
                    report.append(f"\n    Example {i}:")
                    report.append(f"      Doc: {ex.get('doc_id', 'unknown')}")
                    report.append(f"      Context: {ex['context']}")
                    if ex['before']:
                        report.append(f"      Words before: {ex['before']}")
                    if ex['middle']:
                        report.append(f"      Words in middle: {ex['middle']}")

    # Actor comparison section
    if 'by_actor' in aggregated and len(aggregated['by_actor']) > 1:
        report.append("\n" + "="*80)
        report.append("ACTOR COMPARISON")
        report.append("="*80)

        actors = list(aggregated['by_actor'].keys())
        report.append(f"\nActors found: {', '.join(actors)}")

        # Document counts
        report.append("\nDocument counts:")
        for actor in actors:
            count = aggregated['by_actor'][actor]['doc_count']
            report.append(f"  {actor}: {count}")

        # Risk category comparison
        report.append("\nRisk mentions by category (per document average):")
        all_categories = set()
        for actor in actors:
            all_categories.update(aggregated['by_actor'][actor]['risk_counts']['by_category'].keys())

        for category in sorted(all_categories):
            report.append(f"\n  {category}:")
            for actor in actors:
                count = aggregated['by_actor'][actor]['risk_counts']['by_category'].get(category, 0)
                doc_count = aggregated['by_actor'][actor]['doc_count']
                avg = count / doc_count if doc_count > 0 else 0
                report.append(f"    {actor}: {count} total ({avg:.2f} per doc)")

        # Qualification comparison
        report.append("\nQualification distribution by actor:")
        for concept in ['sannolikhet', 'konsekvens', 'risk']:
            report.append(f"\n  {concept.upper()}:")
            level_order = ['very_low', 'low', 'medium', 'high', 'very_high', 'change', 'uncertainty']

            for level in level_order:
                counts_str = []
                for actor in actors:
                    qual_data = aggregated['by_actor'][actor].get('qualifications', {}).get(concept, {})
                    count = qual_data.get(level, 0)
                    total = sum(qual_data.values()) if qual_data else 0
                    pct = (count / total * 100) if total > 0 else 0
                    counts_str.append(f"{actor}: {count} ({pct:.1f}%)")
                report.append(f"    {level:12s}: {' | '.join(counts_str)}")

    # Show level mapping for reference
    report.append("\n" + "="*80)
    report.append("LEVEL MAPPING REFERENCE")
    report.append("="*80)

    for concept, mapping in QUALIFICATION_MAPPING.items():
        report.append(f"\n{concept.upper()}:")
        for level, terms in mapping.items():
            report.append(f"  {level:10s}: {', '.join(terms)}")

    # Save report
    report_text = "\n".join(report)
    report_path = output_dir / 'risk_context_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Saved comprehensive report: {report_path}")

    # Print summary to console
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description='Risk context analysis: risk counts + 5-level qualifications'
    )

    parser.add_argument(
        '--texts',
        type=Path,
        required=True,
        help='Path to parquet file with texts'
    )

    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Column name containing text (default: text)'
    )

    parser.add_argument(
        '--metadata',
        nargs='+',
        default=['doc_id', 'municipality', 'year', 'actor'],
        help='Metadata columns to include'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./risk_context_analysis'),
        help='Output directory'
    )

    args = parser.parse_args()

    print("="*80)
    print("RISK CONTEXT ANALYSIS")
    print("="*80)

    print(f"\nLoading texts from: {args.texts}")
    texts_df = pd.read_parquet(args.texts)
    print(f"Loaded {len(texts_df)} documents")

    print("\nAnalyzing documents...")
    print("  - Removing 'risk- och sårbarhetsanalys' boilerplate")
    print("  - Counting risk terms from dictionary")
    print("  - Extracting qualified mentions (sannolikhet, konsekvens, risk)")
    print("  - Mapping to 5-level scale: very_low -> very_high")

    results_df, aggregated = analyze_corpus(
        texts_df,
        text_column=args.text_column,
        metadata_columns=args.metadata
    )

    print("\nSaving results...")
    save_results(results_df, aggregated, args.output)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"""
Output files in {args.output}:
  - risk_context_analysis_by_document.csv  (per-document counts)
  - risk_context_analysis_aggregated.json  (corpus-level statistics)
  - risk_context_analysis_report.txt       (comprehensive report)

The CSV contains:
  - Metadata columns (doc_id, municipality, year, etc.)
  - Risk counts (total_risk_mentions, risk_naturhot, etc.)
  - 5-level qualification counts (sannolikhet_very_low, sannolikhet_high, etc.)

Key changes from previous version:
  - 'risk- och sårbarhetsanalys' phrases excluded
  - 'risk- och sårbarhetsanalys' phrases excluded
  - Qualifications grouped into 5 levels + change, uncertainty, acceptability
    """)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
