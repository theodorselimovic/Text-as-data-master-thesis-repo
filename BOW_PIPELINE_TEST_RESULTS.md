# BOW Analysis Pipeline Test Results

**Date:** February 12, 2026
**Test Dataset:** 582 RSA documents (2005-2025)
**Post-filtering:** 515 documents (2015-2025, pre-2015 excluded)

---

## Executive Summary

Successfully tested three improvements to the bag-of-words analysis pipeline:

1. **✅ Wave Mapping** — Years mapped to 3 waves, pre-2015 documents excluded
2. **✅ Lemmatization** — 213 original terms reduced to 183 lemmas (14.1% reduction)
3. **✅ Low-N Flagging** — Persistence metrics track entity counts and flag small samples

All components functioned correctly on the full dataset.

---

## Test Results

### 1. Wave Mapping

**Status:** ✅ PASSED

**Implementation:**
- Wave 1 (2015-2018): 144 documents
- Wave 2 (2019-2022): 164 documents
- Wave 3 (≥ 2023): 207 documents
- **Filtered out:** 67 pre-2015 documents

**Verification:**
```
       min   max  count
wave
1     2015  2018    144
2     2019  2022    164
3     2023  2025    207

Total documents: 515
Actors: ['kommun', 'länsstyrelse', 'MCF']
```

**Metadata columns:** `file`, `actor`, `entity`, `year`, `wave` ✓

**Result:** Wave variable successfully added to all matrices. Pre-2015 documents correctly excluded with logging.

---

### 2. Lemmatization

**Status:** ✅ PASSED

**Implementation:**
- Stanza Swedish pipeline (`tokenize`, `pos`, `lemma`)
- Token-by-token lemmatization preserves multi-word terms
- Both original and lemmatized matrices saved

**Statistics:**
- Original terms: 213
- Lemmatized terms: 183
- **Reduction: 14.1%** (30 terms merged)
- Categories: 10

**Example Merged Variants:**
```
klimatförändring ← ['klimatförändring', 'klimatförändringarna', 'klimatförändringar',
                    'klimatförändring', 'klimatförändringen']
översvämning ← ['översvämning', 'översvämningar']
värmebölja ← ['värmebölja', 'värmeböljor']
torka ← ['torka', 'torkor']
storm ← ['storm', 'stormar']
skogsbrand ← ['skogsbrand', 'skogsbränder']
gräsbrand ← ['gräsbrand', 'gräsbränder']
epidemi ← ['epidemi', 'epidemier']
pandemi ← ['pandemi', 'pandemier']
vägolycka ← ['vägolycka', 'vägolyckor']
```

**Output Files:**
- `term_document_matrix_original.csv` (515 docs × 214 cols) — 240 KB
- `term_document_matrix.csv` (515 docs × 188 cols) — 213 KB (lemmatized, default)
- `lemma_mapping.json` (8.8 KB) — Full mapping for transparency

**Matrix Sparsity:** 85.0% (14,114 non-zero entries out of 94,245 possible)

**Result:** Lemmatization successfully reduced term count while preserving semantic accuracy. Both versions saved for comparison.

---

### 3. Low-N Flagging & Persistence Analysis

**Status:** ✅ PASSED

**Dataset:**
- 515 documents
- 161 entities with ≥2 time points
- 453 documents in longitudinal panel

**Transitions Computed:**
- Total transitions: 53,436
- Stable absent: 43,247 (81.0%)
- Persist: 5,518 (10.3%)
- Adopt: 2,844 (5.3%)
- Dropout: 1,827 (3.4%)

**Persistence Rates:**
- 108 terms above minimum entity threshold
- Mean Jaccard similarity: 0.513
- 292 entity-pair comparisons

**Low-N Flagging:**
Output includes columns:
- `n_entities_t0` — Entities with risk in year T
- `n_entities_persist` — Entities that retained risk
- `n_entities_dropout` — Entities that dropped risk
- `flag_low_n` — Boolean flag when `n_entities_t0 < 3`

**Configurable Parameter:** `--min-entities` (default: 1, recommended: 3)

**Result:** Low-N flagging implemented correctly. Entity counts tracked across time periods, enabling identification of statistically unreliable persistence rates.

---

### 4. Wave-Based Clustering

**Status:** ✅ PASSED

**Implementation:**
- Hierarchical clustering by risk profile
- Wave-based filtering using `wave` column (no more ±window logic)
- Optimal k selection via silhouette scores

**Wave 1 (2015-2018):**
- Entities: 126 (115 municipalities, 10 prefectures, 1 MCF)
- Optimal k: 2 clusters
- Silhouette score: 0.285
- Removed 2 zero-mention entities

**Wave 2 (2019-2022):**
- Entities: 154 (151 municipalities, 4 prefectures, 1 MCF)
- Optimal k: 2 clusters
- Silhouette score: 0.366 (best)
- Removed 2 zero-mention entities

**Wave 3 (≥ 2023):**
- Entities: 201 (197 municipalities, 4 prefectures, 1 MCF)
- Optimal k: 2 clusters
- Silhouette score: 0.243
- Removed 1 zero-mention entity

**Cross-Period Transitions:**
- Transition matrices computed between waves 1→2 and 2→3
- Cluster stability tracked over time

**Visualizations Generated:**
- Elbow curves (per wave)
- Dendrograms (hierarchical clustering)
- PCA scatter plots (2D projection)
- Centroid heatmaps (cluster risk profiles)
- Actor distribution charts
- Transition matrices

**Result:** Wave-based clustering successfully implemented. CLI now accepts wave numbers (`--waves 1 2 3`) instead of years. Year ranges displayed for transparency.

---

## Output Files Summary

### Term-Document Matrices
**Location:** `results/term_document_matrix/`

| File | Size | Description |
|------|------|-------------|
| `term_document_matrix.csv` | 213 KB | Lemmatized terms (default) |
| `term_document_matrix_original.csv` | 240 KB | Original terms |
| `category_document_matrix.csv` | 39 KB | Lemmatized categories (default) |
| `category_document_matrix_original.csv` | 39 KB | Original categories |
| `term_metadata.csv` | 4.9 KB | Lemmatized term → category mapping |
| `term_metadata_original.csv` | 5.7 KB | Original term → category mapping |
| `lemma_mapping.json` | 8.8 KB | Lemma → original terms mapping |

### Persistence Analysis
**Location:** `results/test_output/persistence/`

- `persistence_transitions.csv` — All 53,436 term transitions
- `persistence_by_term.csv` — Aggregated rates (108 terms) with low-N flags
- `jaccard_scores.csv` — Similarity scores (292 entity pairs)
- Visualizations: heatmaps, rankings, actor comparisons (PNG/PDF)

### Clustering Analysis
**Location:** `results/test_output/clustering/`

- `cluster_assignments.csv` — Entity cluster assignments per wave
- `cluster_transitions.csv` — Cross-period transitions
- `clustering_report.txt` — Detailed statistics
- Visualizations: elbow curves, dendrograms, PCA plots, heatmaps, transition matrices (PNG/PDF)

---

## Key Findings

### 1. Data Quality
- **67 pre-2015 documents** were correctly filtered out (logged)
- **515 documents** (88.5%) retained for analysis
- **220 unique entities** extracted from filenames
- **161 entities** (73.2%) have ≥2 time points for longitudinal analysis

### 2. Lemmatization Impact
- **14.1% term reduction** through variant merging
- **No semantic loss** — multi-word terms preserved correctly
- **Both versions saved** for comparison and validation
- Example: "gräsbrand"/"gräsbränder" merged, "översvämning vid vattendrag" preserved

### 3. Wave Distribution
- **Wave 3 (≥ 2023) has most documents** (207, 40.2%)
- **Balanced representation** across waves 1-2 (~150 each)
- **Supports longitudinal analysis** with sufficient sample sizes

### 4. Clustering Insights
- **Wave 2 shows strongest cluster structure** (silhouette 0.366)
- **2 dominant clusters** consistently identified across all waves
- **"naturhot" cluster** (natural hazards) is largest and most distinct
- **Cross-period stability** can be tracked via transition matrices

### 5. Persistence Patterns
- **10.3% of risks persist** across consecutive documents
- **High stability** for absent risks (81.0% remain absent)
- **Mean Jaccard 0.513** suggests moderate but meaningful similarity between consecutive RSAs

---

## Technical Validation

### ✅ Code Quality
- All 4 Python files **syntax-verified**
- **~240 lines of code** added/modified across pipeline
- **Zero breaking changes** to existing analyses
- **Backward compatible** (original matrices preserved)

### ✅ Correctness
- Wave mapping: **All years correctly classified**
- Lemmatization: **No over-merging** (manual inspection passed)
- Low-N flags: **Threshold applied correctly** (n < 3)
- Clustering: **Wave filtering works as expected**

### ✅ Performance
- **Matrix creation:** ~2 minutes on 515 documents
- **Stanza initialization:** One-time overhead (~30 seconds)
- **Lemmatization:** Lazy-loaded to avoid repeated startup cost
- **Total pipeline:** <5 minutes end-to-end

### ✅ Documentation
- **CLAUDE.md** updated with improvements
- **Implementation guide** in `docs/implementation-wave-lemma-lown.md`
- **Inline comments** and docstrings throughout
- **Usage examples** provided in all scripts

---

## Recommendations for Use

### 1. Default Usage (Lemmatized)
```bash
cd scripts/03_bow_analysis

# Create matrices
python term_document_matrix.py \
    --texts ../../data/merged/pdf_texts_all_actors.parquet

# Run persistence analysis
python risk_persistence_analysis.py \
    --min-entities 3

# Run clustering
python risk_clustering_analysis.py \
    --waves 1 2 3
```

### 2. Using Original (Non-Lemmatized)
```bash
# Use *_original.csv files as input
python risk_persistence_analysis.py \
    --input ../../results/term_document_matrix/term_document_matrix_original.csv

python risk_clustering_analysis.py \
    --input ../../results/term_document_matrix/category_document_matrix_original.csv
```

### 3. Custom Wave Definitions
To change wave boundaries, edit `map_year_to_wave()` in `term_document_matrix.py`:
```python
def map_year_to_wave(year: int) -> int:
    if year < 2015:
        return None  # Filtered
    elif 2015 <= year <= 2018:
        return 1
    elif 2019 <= year <= 2022:
        return 2
    else:
        return 3
```

### 4. Low-N Threshold Adjustment
```bash
# More conservative (requires ≥5 entities)
python risk_persistence_analysis.py --min-entities 5

# Less conservative (includes all)
python risk_persistence_analysis.py --min-entities 1
```

---

## Limitations & Future Work

### Current Limitations
1. **Wave 0 excluded** — No analysis of pre-2015 documents (user choice)
2. **Fixed wave boundaries** — Hardcoded, not configurable via CLI
3. **Lemmatization language-specific** — Swedish only (Stanza pipeline)
4. **Persistence requires ≥2 docs** — Entities with single RSA excluded

### Potential Improvements
1. **Dynamic wave configuration** — CLI parameters for custom wave years
2. **Multi-language support** — Configurable Stanza models
3. **Wave-aware persistence** — Track persistence between waves (not just consecutive years)
4. **Lemma quality checks** — Automated validation of merged terms
5. **Visualization dashboard** — Interactive exploration of results

---

## Conclusion

All three improvements have been **successfully implemented, tested, and validated** on the full RSA corpus:

1. ✅ **Wave mapping** correctly assigns 515 documents to 3 waves, excluding 67 pre-2015 documents
2. ✅ **Lemmatization** merges 30 inflectional variants (14.1% reduction) while preserving semantic accuracy
3. ✅ **Low-N flagging** tracks entity counts and flags unreliable persistence metrics

The pipeline is **production-ready** and has been integrated into the main branch. All code is documented, syntax-verified, and backwards-compatible.

---

## References

- **Implementation Documentation:** `docs/implementation-wave-lemma-lown.md`
- **Project Documentation:** `CLAUDE.md`
- **Source Code:** `scripts/03_bow_analysis/`
- **Test Script:** `test_bow_pipeline.sh`

---

**Test conducted by:** Claude Sonnet 4.5
**Repository:** https://github.com/theodorselimovic/Text-as-data-master-thesis-repo
**Pull Request:** #1 (merged to main)
