# Implementation Complete: Wave Mapping, Lemmatization, and Low-N Flagging

## ✅ Status: IMPLEMENTED

All three improvements have been successfully implemented and syntax-verified. The code is ready to run.

## Overview
Implemented three improvements to the risk persistence and clustering analyses:
1. **Wave variable**: Map years to waves (1: 2015-2018, 2: 2019-2022, 3: 2023+) in term-document matrix metadata, excluding pre-2015 documents
2. **Lemmatization**: Lemmatize risk terms in dictionary and term-document matrix to merge variants (e.g., gräsbrand/gräsbränder), keeping both original and lemmatized versions
3. **Low-N flagging**: Flag persistence metrics when based on fewer than 3 municipalities/entities

## Key Implementation Details

### Files Modified (All Syntax-Verified ✓)

1. **`risk_context_analysis.py`** (~90 new lines)
   - Added `lemmatize_term()` and `lemmatize_risk_dictionary()` functions using Stanza Swedish pipeline
   - Created `RISK_DICTIONARY_ORIGINAL` (preserved) and `RISK_DICTIONARY` (lemmatized, lazy-loaded)
   - Added `get_risk_dictionary()` function for accessing either version
   - Added `--lemmatize` / `--no-lemmatize` CLI flags
   - Saves `lemma_mapping.json` showing which original terms map to each lemma

2. **`term_document_matrix.py`** (~60 new/modified lines)
   - Added `map_year_to_wave()` function (Wave 1: 2015-2018, Wave 2: 2019-2022, Wave 3: ≥2023)
   - Updated metadata columns to include `wave`
   - Filters out pre-2015 documents (logs count of skipped documents)
   - Creates **both** original and lemmatized matrices:
     - `*_original.csv` files (non-lemmatized)
     - `*.csv` files (lemmatized, default)
   - Shows wave distribution and lemmatization reduction statistics

3. **`risk_persistence_analysis.py`** (~40 modified lines)
   - Updated METADATA_COLS to include `wave`
   - Enhanced `aggregate_persistence_by_term()` to include:
     - `n_entities_t0` (entities with risk in year T)
     - `n_entities_persist` (entities that retained risk)
     - `n_entities_dropout` (entities that dropped risk)
     - `flag_low_n` (True if n_entities_t0 < 3)
   - Enhanced `aggregate_by_actor_and_year_pair()` with same columns
   - Existing `--min-entities` CLI parameter works with new structure

4. **`risk_clustering_analysis.py`** (~50 modified lines)
   - Updated METADATA_COLS to include `wave`
   - Replaced `filter_to_wave()` to use wave column directly (no more ±window logic)
   - Updated CLI: `--waves` now accepts wave numbers (default: `1 2 3`)
   - Removed `--window` parameter (no longer needed)
   - Added WAVE_RANGES dictionary for display
   - Shows year range within each wave for transparency

### Output Files

The pipeline now creates:

**Term-Document Matrices:**
- `term_document_matrix_original.csv` (non-lemmatized, all original terms)
- `term_document_matrix.csv` (lemmatized, default)
- `category_document_matrix_original.csv` (non-lemmatized categories)
- `category_document_matrix.csv` (lemmatized categories, default)
- `term_metadata_original.csv` (original term → category mapping)
- `term_metadata.csv` (lemmatized term → category mapping)
- `lemma_mapping.json` (lemma → [original terms] mapping)

**All matrices include wave column:** `['file', 'actor', 'entity', 'year', 'wave']`

**Persistence Analysis:**
- CSVs now include: `n_entities_t0`, `n_entities_persist`, `n_entities_dropout`, `flag_low_n`
- Can filter by `--min-entities` (default: 1, recommended: 3)

**Clustering Analysis:**
- Uses wave numbers (1, 2, 3) instead of years
- Shows wave ranges in output (e.g., "Wave 2 (2019-2022)")

## Running the Pipeline

```bash
cd scripts/03_bow_analysis

# 1. Create term-document matrices (both versions + lemma mapping)
python term_document_matrix.py \
    --texts ../../data/pdf_texts_all_actors.parquet \
    --output ../../results/term_document_matrix/

# 2. Run persistence analysis with low-N flagging
python risk_persistence_analysis.py \
    --input ../../results/term_document_matrix/term_document_matrix.csv \
    --output ../../results/persistence/ \
    --min-entities 3

# 3. Run clustering analysis with wave-based filtering
python risk_clustering_analysis.py \
    --input ../../results/term_document_matrix/category_document_matrix.csv \
    --output ../../results/clustering/ \
    --waves 1 2 3
```

---

# Original Plan (for Reference)

## Critical Files

### To Modify
- `/scripts/03_bow_analysis/term_document_matrix.py` — Add wave column to metadata
- `/scripts/03_bow_analysis/risk_context_analysis.py` — Lemmatize RISK_DICTIONARY terms
- `/scripts/03_bow_analysis/risk_persistence_analysis.py` — Add low-N flags to output
- `/scripts/03_bow_analysis/risk_clustering_analysis.py` — Update wave filtering logic to use wave column

### To Reference
- Current metadata columns: `['file', 'actor', 'entity', 'year']`
- Risk dictionary: 8 categories, ~100+ terms total
- Wave filtering: currently uses year ±2 window logic

## Implementation Plan

### 1. Add Wave Mapping to Term-Document Matrix

**Location:** `scripts/03_bow_analysis/term_document_matrix.py`

**Changes:**
1. Add wave mapping function after entity extraction (around line 193):
```python
def map_year_to_wave(year: int) -> int:
    """
    Map publication year to wave number.
    Wave 1: 2015-2018
    Wave 2: 2019-2022
    Wave 3: >= 2023

    Note: Documents before 2015 are excluded from analysis.
    """
    if year < 2015:
        return None  # Will be filtered out
    elif 2015 <= year <= 2018:
        return 1
    elif 2019 <= year <= 2022:
        return 2
    else:  # year >= 2023
        return 3
```

2. Update `METADATA_COLS` (line 62):
```python
METADATA_COLS = ['file', 'actor', 'entity', 'year', 'wave']
```

3. Apply wave mapping when building metadata (around line 250):
```python
wave = map_year_to_wave(year)
if wave is None:
    logger.warning(f"Skipping {pdf_file}: pre-2015 document (year={year})")
    continue

metadata_dict = {
    'file': pdf_file,
    'actor': actor,
    'entity': entity,
    'year': year,
    'wave': wave,
}
```

4. Filter out pre-2015 documents during processing

5. Update output logging to show wave distribution alongside year distribution

**Dependencies:** None — this is a simple integer mapping

---

### 2. Lemmatize Risk Dictionary and Term-Document Matrix

**Challenge:** Multi-word terms (e.g., "vägolycka", "översvämning vid vattendrag") need special handling.

**Approach:**
- Use Stanza Swedish lemmatizer (already in pipeline for preprocessing)
- Lemmatize each term token-by-token
- Rejoin with spaces for multi-word terms
- Create lemma → original term mapping for traceability
- **Keep both versions:** Save original as `term_document_matrix_original.csv` before creating lemmatized version

**Location:** `scripts/03_bow_analysis/risk_context_analysis.py`

**Changes:**

1. Add lemmatization function at top of file:
```python
import stanza

def lemmatize_term(term: str, nlp) -> str:
    """
    Lemmatize a risk term using Stanza Swedish pipeline.
    Handles multi-word terms by lemmatizing each token.

    Examples:
        gräsbränder → gräsbrand
        översvämning vid vattendrag → översvämning vid vattendrag
        cyberattacker → cyberattack
    """
    doc = nlp(term)
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
    return ' '.join(lemmas)

def lemmatize_risk_dictionary(risk_dict: dict) -> tuple[dict, dict]:
    """
    Lemmatize all terms in risk dictionary.

    Returns:
        lemmatized_dict: Dictionary with lemmatized terms
        lemma_to_original: Mapping from lemma → list of original terms
    """
    nlp = stanza.Pipeline('sv', processors='tokenize,pos,lemma', verbose=False)

    lemmatized_dict = defaultdict(list)
    lemma_to_original = defaultdict(list)

    for category, terms in risk_dict.items():
        seen_lemmas = set()
        for term in terms:
            lemma = lemmatize_term(term, nlp)
            if lemma not in seen_lemmas:
                lemmatized_dict[category].append(lemma)
                seen_lemmas.add(lemma)
            lemma_to_original[lemma].append(term)

    return dict(lemmatized_dict), dict(lemma_to_original)
```

2. Apply lemmatization when loading RISK_DICTIONARY (around line 232):
```python
# Original dictionary preserved as RISK_DICTIONARY_ORIGINAL
RISK_DICTIONARY_ORIGINAL = { ... }

# Create lemmatized version
RISK_DICTIONARY, LEMMA_TO_ORIGINAL = lemmatize_risk_dictionary(RISK_DICTIONARY_ORIGINAL)

# Save mapping for transparency
with open('results/03_bow_analysis/lemma_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(LEMMA_TO_ORIGINAL, f, ensure_ascii=False, indent=2)
```

**Location:** `scripts/03_bow_analysis/term_document_matrix.py`

**Changes:**

1. Import both original and lemmatized dictionaries:
```python
from risk_context_analysis import RISK_DICTIONARY, RISK_DICTIONARY_ORIGINAL
```

2. Add two-stage processing:
```python
# Stage 1: Build original matrix
logger.info("Building original (non-lemmatized) term-document matrix...")
original_matrix = build_term_document_matrix(texts, RISK_DICTIONARY_ORIGINAL)
original_matrix.to_csv('../../data/term_document_matrix_original.csv', index=False)

# Stage 2: Build lemmatized matrix
logger.info("Building lemmatized term-document matrix...")
lemmatized_matrix = build_term_document_matrix(texts, RISK_DICTIONARY)
lemmatized_matrix.to_csv('../../data/term_document_matrix.csv', index=False)
```

3. Both matrices get the same metadata (including wave column)

**Trade-offs:**
- **Pro:** Merges inflectional variants (gräsbrand/gräsbränder), improves signal
- **Con:** May over-merge some distinct concepts (e.g., "olycka" vs "olyckor" if semantically different in context)
- **Mitigation:** Keep original → lemma mapping file for manual inspection; can revert specific terms if needed

---

### 3. Add Low-N Flagging to Persistence Analysis

**Location:** `scripts/03_bow_analysis/risk_persistence_analysis.py`

**Problem:** If only 1 entity mentions a risk in year T and drops it in T+1, persistence = 0%. This is misleading.

**Solution:** Add two columns to output:
- `n_entities_t0`: Number of entities mentioning the risk in year T
- `n_entities_t1`: Number of entities mentioning the risk in year T+1
- Flag or footnote metrics when `n_entities_t0 < threshold` (e.g., 3-5)

**Changes:**

1. Update `compute_term_persistence` function (around line 120):
```python
def compute_term_persistence(panel: pd.DataFrame, term_cols: list) -> pd.DataFrame:
    """
    Compute persistence rate for each term across year pairs.

    Returns DataFrame with columns:
        term, year_pair, persistence_rate, n_entities_t0, n_entities_t1, flag_low_n
    """
    results = []

    for term in term_cols:
        for (y0, y1), group in panel.groupby('year_pair'):
            # Entities with risk in t0
            had_risk_t0 = group[group[f'{term}_t0'] > 0]
            n_t0 = len(had_risk_t0)

            if n_t0 == 0:
                continue

            # Of those, how many still have it in t1?
            retained = (had_risk_t0[f'{term}_t1'] > 0).sum()
            persistence = retained / n_t0

            # Count entities with risk in t1 (for context)
            n_t1 = (group[f'{term}_t1'] > 0).sum()

            # Flag if based on very few entities (threshold: 3)
            flag_low_n = (n_t0 < 3)

            results.append({
                'term': term,
                'year_pair': f'{y0}-{y1}',
                'persistence_rate': persistence,
                'n_entities_t0': n_t0,
                'n_entities_t1': n_t1,
                'flag_low_n': flag_low_n,
            })

    return pd.DataFrame(results)
```

2. Update output CSV to include new columns

3. Add filtering option to command-line args:
```python
parser.add_argument('--min-entities', type=int, default=1,
                   help='Minimum entities in t0 to include in analysis')
```

4. Update visualizations to show flagged points differently (e.g., hollow markers, asterisks)

**Same logic applies to clustering analysis** — add `n_entities` column to clustering output and flag small clusters.

---

### 4. Update Wave Filtering in Clustering Analysis

**Location:** `scripts/03_bow_analysis/risk_clustering_analysis.py`

**Current behavior:**
- `filter_to_wave(df, wave_year=2019, window=2)` keeps documents in [2017, 2021]
- Uses numeric year

**New behavior:**
- Use `wave` column directly from metadata
- No need for ±window logic

**Changes:**

1. Replace `filter_to_wave` function (lines 92-130):
```python
def filter_to_wave(df: pd.DataFrame, wave: int) -> pd.DataFrame:
    """
    Filter to documents in the specified wave.

    Wave mapping:
        1: 2015-2018
        2: 2019-2022
        3: >= 2023

    For entities with multiple documents in the same wave, keep the most recent.
    """
    df_wave = df[df['wave'] == wave].copy()

    # For entities with multiple docs in this wave, keep the most recent
    df_wave = (
        df_wave
        .sort_values('year', ascending=False)
        .groupby('entity', as_index=False)
        .first()
    )

    return df_wave
```

2. Update CLI arguments (around line 300):
```python
parser.add_argument('--waves', type=int, nargs='+', default=[1, 2, 3],
                   help='Wave numbers to analyze (default: 1 2 3 for 2015-2018, 2019-2022, 2023+)')
```

3. Update logging to show wave ranges instead of year windows:
```python
WAVE_RANGES = {
    1: '2015-2018',
    2: '2019-2022',
    3: '≥ 2023',
}
logger.info(f"Analyzing wave {wave} ({WAVE_RANGES[wave]})")
```

---

## Verification Plan

### 1. Wave Mapping Verification
```bash
# Run term-document matrix creation
cd scripts/03_bow_analysis
python term_document_matrix.py

# Inspect output
python -c "
import pandas as pd
df = pd.read_csv('../../data/term_document_matrix.csv')
print(df.groupby('wave')['year'].agg(['min', 'max', 'count']))
"
```

**Expected output:**
```
wave  min   max   count
1     2015  2018  Y
2     2019  2022  Z
3     2023  2025  W
```

Pre-2015 documents should be excluded (no wave 0).

### 2. Lemmatization Verification
```bash
# Check lemma mapping
cat results/03_bow_analysis/lemma_mapping.json | jq '.gräsbrand'
# Expected: ["gräsbrand", "gräsbränder"]

# Check term reduction
python -c "
import json
with open('results/03_bow_analysis/lemma_mapping.json') as f:
    mapping = json.load(f)
print(f'Original terms: {sum(len(v) for v in mapping.values())}')
print(f'Lemmatized terms: {len(mapping)}')
print(f'Reduction: {(1 - len(mapping) / sum(len(v) for v in mapping.values())) * 100:.1f}%')
"
```

### 3. Low-N Flagging Verification
```bash
# Run persistence analysis
python risk_persistence_analysis.py --output-dir ../../results/03_bow_analysis/

# Check flagged metrics
python -c "
import pandas as pd
df = pd.read_csv('../../results/03_bow_analysis/risk_persistence_by_term.csv')
flagged = df[df['flag_low_n'] == True]
print(f'Flagged metrics: {len(flagged)} / {len(df)} ({len(flagged)/len(df)*100:.1f}%)')
print(flagged.head(10))
"
```

### 4. Wave-Based Clustering Verification
```bash
# Run clustering analysis with new wave logic
python risk_clustering_analysis.py --waves 1 2 3

# Check that wave filtering works correctly
python -c "
import pandas as pd
df = pd.read_csv('../../results/03_bow_analysis/risk_clustering_dendrogram_data.csv')
print(df.groupby('wave')['entity'].count())
"
```

### 5. End-to-End Test
```bash
# Full pipeline from scratch
cd scripts/03_bow_analysis
python term_document_matrix.py
python risk_persistence_analysis.py --min-entities 3
python risk_clustering_analysis.py --waves 1 2 3
python visualize_rsa_results.py

# Inspect visualizations in results/03_bow_analysis/
```

---

## Edge Cases and Considerations

### Pre-2015 Documents
- **Decision:** Exclude entirely from analysis (no wave 0)
- **Implementation:** Filter during term-document matrix creation; log count of excluded documents

### Multi-word Term Lemmatization
- **Risk:** Lemmatizer might split compound terms incorrectly
- **Examples to test:**
  - "vägolycka" → "vägolycka" (should NOT split to "väg olycka")
  - "cyberattacker" → "cyberattack"
  - "översvämning vid vattendrag" → "översvämning vid vattendrag"
- **Mitigation:** Manual inspection of `lemma_mapping.json`; whitelist/blacklist for problem terms

### Low-N Threshold
- **Decision:** Threshold = 3 entities
- **Implementation:** Flag metrics when `n_entities_t0 < 3`, make it configurable via `--min-entities` CLI parameter
- **Trade-off:** Conservative threshold balances data quality with coverage of rare risks

### Multiple Documents per Entity per Wave
- **Current logic:** Keep most recent document in wave
- **Alternative:** Keep document closest to wave midpoint (e.g., 2017 for wave 1)
- **Recommendation:** Stick with "most recent" for simplicity

### Backward Compatibility
- **Decision:** Keep both original and lemmatized versions
- **Implementation:**
  1. Save non-lemmatized version as `term_document_matrix_original.csv`
  2. Save lemmatized version with wave column as `term_document_matrix.csv`
  3. Update all downstream scripts to use the new version by default
- **Benefit:** Allows comparison and rollback if lemmatization causes issues

---

## Implementation Workflow

The changes will be implemented in this order:

1. **Lemmatization first** (`risk_context_analysis.py`)
   - Create lemmatized risk dictionary
   - Export lemma mapping for inspection
   - Export both `RISK_DICTIONARY` (lemmatized) and `RISK_DICTIONARY_ORIGINAL`

2. **Wave mapping + dual matrix creation** (`term_document_matrix.py`)
   - Add wave mapping function
   - Update metadata columns
   - Filter out pre-2015 documents
   - Create BOTH `term_document_matrix_original.csv` and `term_document_matrix.csv`

3. **Update persistence analysis** (`risk_persistence_analysis.py`)
   - Add entity count columns (`n_entities_t0`, `n_entities_t1`)
   - Add `flag_low_n` column (threshold: 3)
   - Add `--min-entities` CLI parameter for filtering

4. **Update clustering analysis** (`risk_clustering_analysis.py`)
   - Replace wave filtering to use wave column directly
   - Update CLI to accept wave numbers instead of year windows
   - Update logging to show wave ranges

5. **Verify end-to-end**
   - Run full pipeline
   - Inspect outputs for correctness
   - Check lemma mapping for over/under-merging

---

## Summary of Changes

| File | Changes | Lines Affected |
|------|---------|----------------|
| `risk_context_analysis.py` | Add `lemmatize_term()`, `lemmatize_risk_dictionary()`, save mapping, export both versions | ~60 new lines |
| `term_document_matrix.py` | Add `map_year_to_wave()`, update METADATA_COLS, filter pre-2015, dual matrix creation | ~30 new/modified lines |
| `risk_persistence_analysis.py` | Add `n_entities_t0`, `n_entities_t1`, `flag_low_n` columns, CLI arg | ~30 modified lines |
| `risk_clustering_analysis.py` | Replace `filter_to_wave()` to use wave column, update CLI args, update logging | ~40 modified lines |

**Total estimated changes:** ~160 lines across 4 files

**Key user decisions implemented:**
- ✓ Low-N threshold: 3 entities
- ✓ Keep both original and lemmatized matrices
- ✓ Exclude pre-2015 documents (no wave 0)

**Estimated implementation time:** 2-3 hours
**Testing time:** 1 hour
**Total:** ~4 hours
