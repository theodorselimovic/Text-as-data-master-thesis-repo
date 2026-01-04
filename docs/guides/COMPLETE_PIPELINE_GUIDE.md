# Complete Pipeline Guide: From Seed Terms to Co-occurrence Analysis

## 📋 Overview

This guide walks you through the complete analysis pipeline for your Swedish RSA documents, from expanding seed terms to running chi-square co-occurrence tests.

## 🎯 Your Current Situation

From your error output, I can see:
- ✓ You have 41,311 unique sentences
- ✓ You have 'risk' category expanded (41,186 sentences, 99.7%)
- ⚠️ You have something labeled 'resilience' (39,199 sentences, 94.9%)
- ❌ Missing: complexity, efficiency, equality

**Important**: Based on your seed terms, you should have **'accountability'** not 'resilience'. This suggests you may have mislabeled the category.

## 📝 Your Seed Terms (Corrected)

### Political Effects (5 categories)
1. **Risk**: risk, riskanalys, riskbedömning, sårbarhet, kritiska, beroenden, krisberedskap, samhällsviktig, verksamhet
2. **Accountability**: åtagande, ansvar, skyldighet, förpliktelse, ansvarsområde
3. **Complexity**: komplex, svår, komplicerad, utmaning, otydlig, annorlunda, unik
4. **Efficiency**: effektiv, effektivering, effektivitet, rationell, nyttig, ändamålsenlig, verkningsfull
5. **Equality**: jämförbar, ekvivalent, motsvarande, likvärdig, utbytbar

### Institutional Actors (1 category)
6. **Agency**: kommun, stat, länsstyrelse, region, näringsliv, civilsamhälle, förening

## 🚀 Step-by-Step Pipeline

### STEP 1: Expand All Seed Terms ✨

**File**: `vectoranalysis_complete.ipynb`

**What it does**:
- Loads FastText Swedish model (7GB, takes 5-10 minutes)
- For each seed term, finds 50 most similar words
- Combines similar words from all seed terms in each category
- Lemmatizes all expanded terms
- Removes duplicates
- Saves to: `expanded_terms_lemmatized_complete.csv`

**Run**:
```bash
jupyter lab vectoranalysis_complete.ipynb
```

**Expected output**:
```
FINAL RESULTS SUMMARY
================================================================================

ACCOUNTABILITY: ~200 unique lemmas
COMPLEXITY: ~250 unique lemmas
EFFICIENCY: ~180 unique lemmas
EQUALITY: ~80 unique lemmas
RISK: ~150 unique lemmas
AGENCY: ~300 unique lemmas
```

**Time**: ~15-20 minutes total

---

### STEP 2: Filter Sentences with Expanded Terms 🔍

**File**: `sentencefiltering.ipynb` (your existing file, needs minor update)

**What it does**:
- Loads lemmatized sentence corpus (from `readingtexts.ipynb`)
- Loads expanded terms from STEP 1
- Filters sentences containing ANY expanded term
- Creates category-sentence pairs (sentences can have multiple categories)
- Vectorizes sentences (mean of word vectors)
- Saves to: `sentence_vectors_with_metadata.parquet`

**Update needed in sentencefiltering.ipynb**:

Change this line:
```python
df_expanded = pd.read_csv('expanded_terms_lemmatized.csv')
```

To:
```python
df_expanded = pd.read_csv('expanded_terms_lemmatized_complete.csv')
```

**Run**:
```bash
jupyter lab sentencefiltering.ipynb
```

**Expected output**:
```
Filtered to sentences with target terms: ~15,000-20,000 sentences
Total category-sentence pairs: ~50,000-70,000 pairs

Pairs by category:
accountability    ~12,000
complexity       ~15,000
efficiency       ~8,000
equality         ~2,000
risk             ~30,000
```

**Time**: ~5-10 minutes

---

### STEP 3: Diagnostic Check ✅

**File**: `data_diagnostic.py`

**What it does**:
- Checks which categories are present
- Shows frequencies
- Validates data structure
- Provides recommendations

**Run**:
```bash
python data_diagnostic.py --input sentence_vectors_with_metadata.parquet
```

**Expected output**:
```
CATEGORIES IN DATA
================================================================================
accountability    12456
complexity        15234
efficiency         8901
equality           2134
risk              32789

✓ GOOD! You have sufficient categories
  You can run co-occurrence analysis
```

**Time**: < 1 minute

---

### STEP 4: Chi-Square Co-occurrence Analysis 📊

**File**: `cooccurrence_analysis.py`

**What it does**:
- Identifies specific actors in sentences (kommun, stat, etc.)
- Creates binary indicators (has_risk, has_kommun, etc.)
- Runs chi-square tests:
  - Effect-effect pairs (H4: Do effects interact?)
  - Effect-actor pairs (H2: Which actors with which effects?)
  - Actor-actor pairs (Which actors discussed together?)
- Temporal analysis (H3: Complexity over time?)
- Generates visualizations
- Saves results to CSV files

**Run**:
```bash
python cooccurrence_analysis.py \
    --input sentence_vectors_with_metadata.parquet \
    --output-dir cooccurrence_results
```

**Expected output files**:
- `effect_cooccurrence.csv` - 10 effect pairs
- `effect_actor_associations.csv` - 35 effect-actor pairs  
- `actor_cooccurrence.csv` - 21 actor pairs
- `temporal_frequencies.csv` - Time trends
- `effect_frequencies.png` - Bar chart
- `effect_actor_heatmap.png` - Heatmap

**Time**: ~2-3 minutes

---

### STEP 5: Interactive Exploration 🔬

**File**: `cooccurrence_analysis_notebook.ipynb`

**What it does**:
- Interactive analysis
- Detailed examples
- Custom queries
- Hypothesis testing

**Run**:
```bash
jupyter lab cooccurrence_analysis_notebook.ipynb
```

---

## 🎨 Example Research Findings

After running the full pipeline, you'll be able to answer:

### H1: Which effects dominate?
```
Risk:          75.3% of sentences (most prominent)
Complexity:    32.1% of sentences
Accountability: 28.7% of sentences
Efficiency:    19.4% of sentences
Equality:       4.2% of sentences (least prominent)
```

### H2: Effect-actor associations
```
Efficiency × Kommun
  Observed: 3,452 sentences
  Expected: 1,789 sentences
  Ratio: 1.93× (χ²=487.3, p<0.001, V=0.21)
  → WEAK-MODERATE association
  
Interpretation: Risk analysis systematically frames 
municipal responsibilities through efficiency discourse.
```

### H3: Complexity over time
```
Period      Complexity %
2011-2015   28.3%
2016-2019   31.7%
2020-2024   35.9%

Change: +7.6 percentage points
→ SUPPORT for H3: Complexity has INCREASED
```

### H4: Effects interact
```
Risk × Complexity
  χ²=234.5, p<0.001, V=0.18
  → WEAK association
  
Efficiency × Complexity  
  χ²=156.7, p<0.001, V=0.15
  → WEAK association
  
Interpretation: Concepts discussed separately more 
often than together (not highly integrated discourse).
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Only have 2 categories"
**Solution**: Run STEP 1 (vectoranalysis_complete.ipynb) to expand all seed terms

### Issue 2: "ZeroDivisionError in Cramér's V"
**Solution**: This is now fixed in the updated cooccurrence_analysis.py

### Issue 3: "Categories mislabeled (resilience vs accountability)"
**Solution**: 
1. Check your expanded_terms_lemmatized.csv
2. If it says 'resilience', manually rename to 'accountability':
```python
df = pd.read_csv('expanded_terms_lemmatized.csv')
df['category'] = df['category'].replace('resilience', 'accountability')
df.to_csv('expanded_terms_lemmatized.csv', index=False)
```
3. Re-run sentencefiltering.ipynb

### Issue 4: "FastText model takes too long to load"
**Solution**: This is normal for a 7GB model. It only loads once per session.

### Issue 5: "Stanza lemmatization is slow"
**Solution**: This is normal - lemmatizing 1000+ terms takes 5-10 minutes. Be patient!

---

## 📊 Quality Checks

After each step, verify:

### After STEP 1 (vectoranalysis):
✓ File exists: `expanded_terms_lemmatized_complete.csv`
✓ Has 6 categories: risk, accountability, complexity, efficiency, equality, agency
✓ Total terms: 1000-1500 unique lemmas
✓ Top terms make semantic sense for each category

### After STEP 2 (sentencefiltering):
✓ File exists: `sentence_vectors_with_metadata.parquet`
✓ Has all 6 categories in the data
✓ Reasonable coverage: 15,000-20,000 unique sentences
✓ Risk is most frequent (it's the core concept)

### After STEP 3 (diagnostic):
✓ All categories present
✓ No zero-frequency categories
✓ Reasonable distribution across years
✓ Multiple municipalities represented

### After STEP 4 (cooccurrence):
✓ Results make theoretical sense
✓ P-values < 0.05 for key hypotheses
✓ Effect sizes (Cramér's V) reasonable (0.1-0.3)
✓ Temporal trends plausible

---

## 🎯 Final Checklist

Before running co-occurrence analysis, ensure:

- [x] FastText model loaded successfully
- [x] All 6 categories have expanded terms
- [x] Expanded terms are lemmatized
- [x] Sentence corpus is lemmatized (from readingtexts.ipynb)
- [x] Filtered sentences contain all categories
- [x] Data diagnostic shows all categories present
- [x] No zero-frequency categories
- [x] Sufficient sample size (>10,000 sentences)

Once all boxes are checked, you're ready to run the full co-occurrence analysis!

---

## 📚 Next Steps After Co-occurrence

Once you have chi-square results:

1. **Correspondence Analysis** - Map concepts in 2D space
2. **Network Analysis** - Visualize co-occurrence networks
3. **Log-linear Models** - Test three-way interactions
4. **Qualitative Validation** - Manual coding of sample sentences
5. **Thesis Writing** - Report findings with proper statistical notation

---

## 🆘 Getting Help

If you encounter issues:

1. Run `data_diagnostic.py` to check your data
2. Check the output of each step for error messages
3. Verify file paths are correct
4. Ensure you have enough RAM (8GB+ recommended)
5. Check that all required packages are installed

---

## 📝 Summary

The complete pipeline is:

```
Raw PDFs 
  ↓ (OCR + readtext)
Lemmatized Sentences
  ↓ (STEP 1: vectoranalysis_complete.ipynb)
Expanded Terms (ALL 6 categories)
  ↓ (STEP 2: sentencefiltering.ipynb)
Filtered & Vectorized Sentences
  ↓ (STEP 3: data_diagnostic.py)
Validated Data
  ↓ (STEP 4: cooccurrence_analysis.py)
Chi-Square Results + Visualizations
  ↓ (STEP 5: cooccurrence_analysis_notebook.ipynb)
Interactive Exploration & Hypothesis Testing
```

**Total time**: ~30-40 minutes for complete pipeline

Good luck with your analysis! 🚀
