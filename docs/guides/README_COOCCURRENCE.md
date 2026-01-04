# Co-occurrence Analysis for RSA Documents

## Overview

This package provides chi-square statistical tests to analyze how political effects and institutional actors co-occur in Swedish municipal risk analysis documents.

## Files

1. **cooccurrence_analysis.py** - Main analysis script (command-line tool)
2. **cooccurrence_analysis_notebook.ipynb** - Interactive Jupyter notebook
3. **README_COOCCURRENCE.md** - This file

## Installation

```bash
pip install pandas numpy scipy matplotlib seaborn pyarrow
```

## Quick Start

### Option 1: Command Line (Recommended for Full Analysis)

```bash
python cooccurrence_analysis.py \
    --input sentence_vectors_with_metadata.parquet \
    --output-dir cooccurrence_results
```

This will:
- Load your sentence data
- Identify actors in each sentence
- Create binary indicators for all concepts
- Run chi-square tests on all concept pairs
- Perform temporal analysis
- Generate visualizations
- Save all results to `cooccurrence_results/`

### Option 2: Jupyter Notebook (Recommended for Exploration)

```bash
jupyter lab cooccurrence_analysis_notebook.ipynb
```

This provides:
- Interactive exploration of results
- Detailed examples with interpretation
- Step-by-step execution
- Easy modification for custom analyses

## Understanding the Analysis

### What Does Co-occurrence Mean?

Two concepts "co-occur" if they appear in the same sentence.

**Example:**
```
Sentence: "kommun ansvar resiliens hantering"
Co-occurrences: (kommun, resiliens), (ansvar, resiliens), etc.
```

### What Does Chi-Square Test?

Chi-square tests whether co-occurrence is **more than random chance**.

**Example:**
- 1000 sentences total
- 200 have "efficiency" (20%)
- 150 have "kommun" (15%)
- Random expectation: 200 × 0.15 = 30 sentences with both
- Observed: 60 sentences with both
- **Result**: 2× more than expected! (χ²=45.2, p<0.001)

### How to Interpret Results

Three key statistics:

1. **Chi-square (χ²)**: Test statistic
   - Higher = stronger deviation from independence
   - Compare to critical value (usually χ²>3.84 for p<0.05)

2. **P-value**: Statistical significance
   - p < 0.05: Significant (not due to chance)
   - p < 0.01: Highly significant
   - p < 0.001: Very highly significant

3. **Cramér's V**: Effect size (strength of association)
   - 0.00-0.10: Negligible
   - 0.10-0.30: Weak
   - 0.30-0.50: Moderate
   - 0.50+: Strong

**Example interpretation:**

> "Efficiency and kommun co-occur 2.3× more than expected by chance 
> (χ²=67.4, p<0.001, V=0.23). This weak-to-moderate association suggests 
> risk analysis systematically frames municipal responsibilities through 
> efficiency discourse."

## Output Files

### CSV Files (Results)

1. **effect_cooccurrence.csv**
   - Chi-square tests for all effect pairs
   - Columns: concept1, concept2, chi2, p_value, cramers_v, n_both, ratio
   - Use for: Testing H4 (effects interact)

2. **effect_actor_associations.csv**
   - Chi-square tests for effect-actor pairs
   - Columns: effect, actor, chi2, p_value, cramers_v, n_both, ratio
   - Use for: Testing H2 (actors with specific effects)

3. **actor_cooccurrence.csv**
   - Chi-square tests for actor pairs
   - Columns: actor1, actor2, chi2, p_value, cramers_v, n_both, ratio
   - Use for: Understanding institutional collaboration discourse

4. **temporal_frequencies.csv**
   - Concept frequencies by time period
   - Columns: period, concept, count, total, percentage
   - Use for: Testing H3 (complexity over time)

### Parquet Files (Data)

5. **sentence_binary_indicators.parquet**
   - Sentence-level data with binary indicators
   - Columns: doc_id, year, municipality, has_resilience, has_risk, has_kommun, etc.
   - Use for: Custom analyses, filtering, subgroup comparisons

### Visualizations

6. **effect_frequencies.png**
   - Bar chart of effect frequencies
   - Use for: Answering H1 (which effects dominate)

7. **effect_actor_heatmap.png**
   - Heatmap of Cramér's V for effect-actor pairs
   - Use for: Visual overview of associations

## Research Hypotheses

### H1: Which effects are most prominent?

**Analysis**: Check `effect_frequencies.png` and frequency counts

**Interpretation**: 
```python
# From notebook
for effect in EFFECT_CATEGORIES:
    count = df_binary[f'has_{effect}'].sum()
    pct = count / len(df_binary) * 100
```

### H2: Effects associate with specific actors

**Analysis**: Check `effect_actor_associations.csv`

**Look for**:
- Significant p-values (< 0.05)
- Cramér's V > 0.1
- Observed/expected ratio > 1.5 or < 0.67

**Example finding**:
> "Efficiency is significantly associated with kommun (χ²=67.4, p<0.001, V=0.23),
> appearing together 2.3× more than expected."

### H3: Complexity increases over time

**Analysis**: Check `temporal_frequencies.csv`

**Method**:
```python
df_complexity = df_temporal[df_temporal['concept'] == 'complexity']
first_pct = df_complexity.iloc[0]['percentage']
last_pct = df_complexity.iloc[-1]['percentage']
change = last_pct - first_pct
```

**Interpretation**:
- Change > +2 percentage points: Support for H3
- Change < -2 percentage points: Counter to H3
- |Change| < 2: Inconclusive

### H4: Effects interact (co-occur)

**Analysis**: Check `effect_cooccurrence.csv`

**Look for**:
- Significant associations between effect pairs
- Example: Efficiency and complexity co-occur (coordination in complex systems)

## Actor Identification

The "agency" category is automatically split into specific actors:

- **kommun**: kommun, kommunal, kommuner, kommunen
- **stat**: stat, staten, statlig, statliga
- **länsstyrelse**: länsstyrelse, länsstyrelsen, länsstyrelser
- **region**: region, regionen, regional, regionala
- **näringsliv**: näringsliv, näringslivet, företag, företagen
- **civilsamhälle**: civilsamhälle, civilsamhället, frivillig, frivilliga
- **förening**: förening, föreningen, föreningar

Actors are identified using **word-level matching** (not substring) to avoid false positives like "internationell" → "intern".

## Customization

### Test Specific Concept Pairs

```python
from cooccurrence_analysis import CooccurrenceAnalyzer

analyzer = CooccurrenceAnalyzer(df_binary)
result = analyzer.test_pair('efficiency', 'kommun')

print(f"Chi-square: {result['chi2']:.2f}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Cramér's V: {result['cramers_v']:.3f}")
```

### Filter by Time Period

```python
# Only analyze 2020-2024
recent = df_binary[df_binary['year'] >= 2020]
analyzer_recent = CooccurrenceAnalyzer(recent)
df_results = analyzer_recent.test_effect_actor_association()
```

### Filter by Municipality

```python
# Analyze specific municipality
arvidsjaur = df_binary[df_binary['municipality'] == 'Arvidsjaur']
analyzer_arv = CooccurrenceAnalyzer(arvidsjaur)
```

### Compare Two Periods

```python
from cooccurrence_analysis import TemporalAnalyzer

temporal = TemporalAnalyzer(df_binary)
comparison = temporal.compare_periods(
    'efficiency', 'kommun',
    period1=(2011, 2015),
    period2=(2020, 2024)
)

print(f"Early period V: {comparison['period1_v']:.3f}")
print(f"Recent period V: {comparison['period2_v']:.3f}")
print(f"Change: {comparison['period2_v'] - comparison['period1_v']:+.3f}")
```

## Common Issues

### Issue 1: "No module named 'cooccurrence_analysis'"

**Solution**: Make sure you're in the same directory as `cooccurrence_analysis.py`

```bash
cd /path/to/scripts
python cooccurrence_analysis.py
```

Or add to Python path:
```python
import sys
sys.path.append('/path/to/scripts')
from cooccurrence_analysis import *
```

### Issue 2: "File not found: sentence_vectors_with_metadata.parquet"

**Solution**: Specify the correct path

```bash
python cooccurrence_analysis.py \
    --input /full/path/to/sentence_vectors_with_metadata.parquet
```

### Issue 3: Low frequencies causing warnings

**Solution**: This is normal for rare combinations. Chi-square may be unreliable when expected counts < 5.

Filter to combinations with sufficient data:
```python
df_results = df_results[df_results['expected_both'] >= 5]
```

## Reporting Results in Thesis

### Good Example

> "We tested whether political effects co-occurred with specific institutional 
> actors using chi-square tests of independence. Efficiency terms appeared in 
> 18.7% of sentences (n=8,894), significantly more than complexity (14.2%, 
> n=6,751) or equality (2.3%, n=1,093). Efficiency co-occurred with kommun 
> 2.3 times more often than expected by random chance (χ²=67.4, df=1, p<0.001, 
> Cramér's V=0.23), indicating a weak-to-moderate association. This suggests 
> risk analysis systematically frames municipal responsibilities through 
> efficiency discourse."

### Bad Example

> "The efficiency and kommun vectors were similar (cosine=0.68), suggesting 
> they appear in similar contexts."

**Why bad?**
- Confuses vector similarity (vocabulary) with co-occurrence (discourse structure)
- No statistical test
- Vague interpretation

## Next Steps

After running co-occurrence analysis:

1. **Correspondence Analysis** - Map concepts in 2D space
2. **Network Analysis** - Visualize co-occurrence networks
3. **Log-linear Models** - Model three-way interactions
4. **Municipality Comparison** - Test geographic variation
5. **Qualitative Validation** - Manual coding of sample sentences

## References

- Agresti, A. (2002). *Categorical Data Analysis* (2nd ed.). Wiley.
- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.

## Contact

For questions about the analysis, refer to:
- **METHODOLOGY_SUMMARY.md** - Full methodological documentation
- **cooccurrence_analysis_notebook.ipynb** - Interactive examples
