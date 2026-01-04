# Seed Terms Update Summary

## What Changed (Version 2.0)

### Updates Made: 2025-01-02

---

## 📊 Changes by Category

### 1. **Risk** - EXPANDED ✨

**Before (4 terms):**
- risk, riskanalys, riskbedömning, sårbarhet

**After (9 terms):**
- risk, riskanalys, riskbedömning, sårbarhet, **kritiska, beroenden, krisberedskap, samhällsviktig, verksamhet**

**Added (5 new terms):**
- `kritiska` - critical
- `beroenden` - dependencies
- `krisberedskap` - crisis preparedness
- `samhällsviktig` - societally important
- `verksamhet` - activities

**Rationale:**
These terms capture broader risk discourse including critical infrastructure, dependencies, and crisis management - all central to RSA documents.

**Impact:**
- Likely to increase risk category coverage from ~40% to ~70-80% of sentences
- Better captures "critical dependencies" concept from Swedish risk governance
- "krisberedskap" specifically captures preparedness discourse

---

### 2. **Accountability** - UNCHANGED ✓

**Terms (5):**
- åtagande, ansvar, skyldighet, förpliktelse, ansvarsområde

**Status:** No changes - these terms comprehensively capture accountability discourse.

---

### 3. **Complexity** - REVISED 🔄

**Before (7 terms):**
- **beroende, ömsesidighet,** komplex, svår, komplicerad, utmaning, otydlig

**After (7 terms):**
- komplex, svår, komplicerad, utmaning, otydlig, **annorlunda, unik**

**Removed (2 terms):**
- `beroende` - dependency (moved to Risk as "beroenden")
- `ömsesidighet` - mutuality

**Added (2 new terms):**
- `annorlunda` - different
- `unik` - unique

**Rationale:**
- **Removed**: "beroende" and "ömsesidighet" describe structural relationships, not linguistic markers of complexity
- **Added**: "annorlunda" and "unik" capture discourse about local specificity and non-standard situations
- **Focus shift**: From structural interdependence → linguistic markers of difficulty/uniqueness

**Impact:**
- Clearer semantic distinction between Risk (structural dependencies) and Complexity (difficulty/uniqueness)
- May reduce overlap between Risk and Complexity categories
- Better captures "local uniqueness" aspect of complexity theory

---

### 4. **Efficiency** - UNCHANGED ✓

**Terms (7):**
- effektiv, effektivering, effektivitet, rationell, nyttig, ändamålsenlig, verkningsfull

**Status:** No changes - comprehensive coverage of efficiency discourse.

---

### 5. **Equality** - EXPANDED ✨

**Before (4 terms):**
- jämförbar, ekvivalent, motsvarande, likvärdig

**After (5 terms):**
- jämförbar, ekvivalent, motsvarande, likvärdig, **utbytbar**

**Added (1 new term):**
- `utbytbar` - interchangeable

**Rationale:**
"Utbytbar" captures discourse about interchangeability - situations where different solutions/actors are treated as functionally equivalent.

**Impact:**
- Slightly increases equality category coverage
- Better captures "spaces of equivalence" concept

---

### 6. **Agency** - UNCHANGED ✓

**Terms (7):**
- kommun, stat, länsstyrelse, region, näringsliv, civilsamhälle, förening

**Status:** No changes - comprehensive coverage of institutional actors.

---

## 📈 Expected Impact on Analysis

### Coverage Changes

| Category | V1.0 Terms | V2.0 Terms | Expected Coverage Change |
|----------|------------|------------|--------------------------|
| Risk | 4 | **9** | +30-40 percentage points |
| Accountability | 5 | 5 | No change |
| Complexity | 7 | 7 | -5 to -10 pp (more focused) |
| Efficiency | 7 | 7 | No change |
| Equality | 4 | **5** | +1-2 percentage points |
| Agency | 7 | 7 | No change |

### Semantic Clarity

**Improved:**
- ✓ Clearer distinction between Risk (dependencies) and Complexity (difficulty)
- ✓ More comprehensive risk discourse capture
- ✓ Better alignment with theoretical constructs

**Trade-offs:**
- ⚠️ Complexity category may have lower raw frequency
- ⚠️ Risk category may overlap more with other categories (due to expansion)

---

## 🎯 Theoretical Justification for Changes

### Moving "beroenden" from Complexity to Risk

**Old conception:**
- Complexity = structural interdependence (beroende, ömsesidighet)

**New conception:**
- Risk = includes critical dependencies as risk factors
- Complexity = linguistic markers of difficulty/uniqueness

**Why better:**
1. **Swedish risk governance**: "kritiska beroenden" is a standard term in RSA documents
2. **Theoretical clarity**: Separates structural features (Risk) from experiential/linguistic features (Complexity)
3. **Empirical testability**: Can test whether dependency discourse is framed as risk vs. complexity

### Adding "annorlunda" and "unik" to Complexity

**Theoretical grounding:**
- Complexity theory emphasizes local specificity and non-standardization
- Risk standardization may face resistance from "uniqueness" claims

**Empirical prediction:**
- If municipalities claim "our situation is unique/different", this resists standardization
- Test: Do smaller municipalities use "unik" more? (Resistance to equivalence)

### Adding "krisberedskap" and "samhällsviktig verksamhet" to Risk

**Swedish context:**
- "Krisberedskap" (crisis preparedness) is a legal requirement for Swedish municipalities
- "Samhällsviktig verksamhet" (societally important activities) is a technical term in Swedish risk governance

**Why important:**
- These are institutional terms that structure how municipalities categorize activities
- Capture formalized risk discourse, not just general risk language

---

## 🔄 What You Need to Do

### If You Haven't Run vectoranalysis yet:
**Just run the updated notebook!**
```bash
jupyter lab vectoranalysis_complete.ipynb
```
The seed terms are already updated in the file.

### If You Already Ran vectoranalysis with old terms:
**You need to re-run with the updated terms:**

1. Open `vectoranalysis_complete.ipynb`
2. Verify seed terms match the V2.0 list above
3. Run all cells
4. This will create a new `expanded_terms_lemmatized_complete.csv`
5. Then re-run `sentencefiltering.ipynb`
6. Then run `cooccurrence_analysis.py`

**Why re-run?**
- Different seed terms = different expanded term lists
- Critical for ensuring "beroenden" appears in Risk, not Complexity
- Adds new risk-related terms that may appear in many sentences

---

## 📋 Verification Checklist

After running vectoranalysis with new terms, check:

### 1. Risk Category Expansion
```
Expected top expanded terms:
- risken, riskbedömningen, riskanalysen
- säkerhetsrisk, hälsorisk
- beroende, beroenden (← should be here now!)
- krisberedskap, krishantering
- kritisk, kritiska
```

### 2. Complexity Category Refinement
```
Expected top expanded terms:
- komplex, komplexitet, komplexa
- svår, svårt, svåra
- komplicerad, komplicerade
- utmaning, utmaningar
- unik, unika, annorlunda (← new!)

Should NOT include:
- beroende (moved to Risk)
- ömsesidighet (removed)
```

### 3. Term Counts
```python
df_expanded.groupby('category')['lemma'].nunique()

Expected:
risk             180-220 (increased from ~120-150)
accountability   120-150 (stable)
complexity       150-180 (decreased from ~180-220)
efficiency       150-180 (stable)
equality         80-120 (slight increase)
agency           250-350 (stable)
```

---

## 🎓 Methodological Note

### Why Seed Term Selection Matters

**Problem:**
Seed terms define what we measure. If "beroende" is in Complexity, we test:
- "Does complexity (as interdependence) increase over time?"

If "beroenden" is in Risk, we test:
- "Does risk discourse include dependency framing?"
- "Does complexity (as difficulty/uniqueness) increase over time?"

**The second framing is theoretically clearer** because:
1. Risk analysis explicitly identifies dependencies as risks
2. Complexity should capture difficulty, not just network structure
3. We can test both dependency discourse AND uniqueness claims

### Validity Consideration

**Construct validity:**
- Do our measures capture the theoretical constructs we intend?

**V1.0 issue:**
- "Complexity" mixed structural features (beroende) with experiential features (svår)

**V2.0 improvement:**
- Clearer construct boundaries
- Risk = what is at risk (including dependencies)
- Complexity = how difficult/unique situations are described

---

## 📚 Documentation Updated

The following files have been updated with V2.0 seed terms:

1. ✓ `vectoranalysis_complete.ipynb` - Contains new seed_terms dict
2. ✓ `COMPLETE_PIPELINE_GUIDE.md` - Lists updated terms
3. ✓ `SEED_TERMS_REFERENCE.md` - Full documentation with justifications
4. ✓ This file - Change summary

---

## ⏭️ Next Steps

1. **Run** `vectoranalysis_complete.ipynb` with V2.0 seed terms
2. **Verify** expanded terms look reasonable (check top 20 per category)
3. **Run** `sentencefiltering.ipynb` with new expanded terms
4. **Run** `data_diagnostic.py` to verify all categories present
5. **Run** `cooccurrence_analysis.py` for chi-square tests
6. **Interpret** results with updated theoretical framework

**Expected analysis time**: ~30 minutes total

---

## 📊 Comparison: V1.0 vs V2.0

### Total Seed Terms
- V1.0: 33 seed terms
- V2.0: 40 seed terms (+7)

### Category Changes
- Risk: 4 → 9 (+5 terms, +125%)
- Complexity: 7 → 7 (same count, but 2 removed, 2 added)
- Equality: 4 → 5 (+1 term, +25%)
- Others: No change

### Theoretical Clarity
- V1.0: Some overlap between Risk and Complexity
- V2.0: Clearer boundaries, better construct validity

---

**Version:** 2.0  
**Date:** 2025-01-02  
**Status:** Ready to use
