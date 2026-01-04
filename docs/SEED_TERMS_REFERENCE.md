# Seed Terms Reference Guide

## Overview

This document lists all seed terms used for expanding categories in the Swedish RSA text analysis project. Each category represents a theoretical construct from the research framework analyzing how risk analysis instruments structure politics.

## Complete Seed Terms by Category

### 1. Risk (Core Concept)

**Swedish Terms:**
- risk
- riskanalys
- riskbedömning
- sårbarhet
- kritiska
- beroenden
- krisberedskap
- samhällsviktig
- verksamhet

**English Translation:**
- risk
- risk analysis
- risk assessment
- vulnerability
- critical
- dependencies
- crisis preparedness
- societally important
- activities

**Justification:**
Core concepts for identifying and assessing threats and vulnerabilities in municipal RSA documents. These terms capture the fundamental risk discourse.

**Theoretical Connection:**
Risk is the central organizing concept around which RSA documents are structured. These terms identify where and how risks are being defined, assessed, and categorized.

---

### 2. Accountability (Political Effect)

**Swedish Terms:**
- åtagande
- ansvar
- skyldighet
- förpliktelse
- ansvarsområde

**English Translation:**
- commitment
- responsibility
- obligation
- duty
- area of responsibility

**Justification:**
Terms related to managing institutional risk and delimiting accountability. These capture how responsibilities are assigned and bounded.

**Theoretical Connection:**
Risk analysis creates accountability structures by defining who is responsible for what. This category tests H2 (effects associate with specific actors) by identifying which actors are discussed in terms of responsibility.

---

### 3. Complexity (Political Effect)

**Swedish Terms:**
- komplex
- svår
- komplicerad
- utmaning
- otydlig
- annorlunda
- unik

**English Translation:**
- complex
- difficult
- complicated
- challenge
- unclear
- different
- unique

**Justification:**
Terms related to complexity and local uniqueness. These capture discourse about difficulty, interdependence, and non-standard situations.

**Theoretical Connection:**
Risk analysis may increase complexity by revealing interdependencies. This category tests H3 (complexity increases over time) and H4 (effects interact) by tracking when systems are described as complex or difficult.

**Note:** 
Originally included "beroende" (dependency) and "ömsesidighet" (mutuality), but these were removed to focus on linguistic markers of complexity rather than structural interdependence itself. "Beroenden" was moved to the Risk category as "kritiska beroenden" (critical dependencies).

---

### 4. Efficiency (Political Effect)

**Swedish Terms:**
- effektiv
- effektivering
- effektivitet
- rationell
- nyttig
- ändamålsenlig
- verkningsfull

**English Translation:**
- effective
- effectivisation
- effectiveness
- rational
- useful
- goal appropriate
- efficacious

**Justification:**
Terms related to efficiency and smooth cooperation. These capture when systems, actors, or processes are discussed in terms of their effectiveness or optimization.

**Theoretical Connection:**
Risk analysis may frame governance through efficiency discourse. This category tests H1 (which effects dominate) and H2 (efficiency associates with specific actors, e.g., kommun).

---

### 5. Equality (Political Effect)

**Swedish Terms:**
- jämförbar
- ekvivalent
- motsvarande
- likvärdig
- utbytbar

**English Translation:**
- comparable
- equivalent
- corresponding
- tantamount
- interchangeable

**Justification:**
Terms related to creating "spaces of equivalence" - discourse that frames different situations, actors, or solutions as comparable or interchangeable.

**Theoretical Connection:**
Risk analysis may create equivalences by standardizing risk categories across contexts. This category tests whether discourse treats diverse situations as comparable, potentially obscuring local specificity.

---

### 6. Agency (Institutional Actors)

**Swedish Terms:**
- kommun
- stat
- länsstyrelse
- region
- näringsliv
- civilsamhälle
- förening

**English Translation:**
- municipality
- state
- county administrative board
- region
- business/industry
- civil society
- association

**Justification:**
Relevant institutional actors in Swedish risk governance. These are automatically separated in co-occurrence analysis to test actor-specific associations.

**Theoretical Connection:**
Different actors may be framed through different effects. This category enables testing H2 (which actors associate with which effects) by identifying when specific institutions are mentioned.

**Analysis Note:**
In co-occurrence analysis, this category is automatically split into 7 separate actor indicators (has_kommun, has_stat, etc.) to test fine-grained associations.

---

## Seed Term Selection Criteria

### Inclusion Criteria

Terms are included if they:
1. **Theoretically grounded**: Derived from political science theory about risk governance
2. **Lemmatizable**: Can be normalized to a base form by Stanza
3. **Semantically clear**: Have well-defined meanings in Swedish
4. **Corpus-relevant**: Likely to appear in bureaucratic RSA documents
5. **Distinguishable**: Represent distinct concepts (avoid synonyms)

### Exclusion Criteria

Terms are excluded if they:
1. **Too general**: Would capture unrelated discourse (e.g., "och", "är")
2. **Ambiguous**: Multiple unrelated meanings (e.g., "risk" as both "danger" and "scratch")
3. **Rare**: Unlikely to appear in corpus
4. **Redundant**: Already captured by other terms after lemmatization
5. **Non-diagnostic**: Don't distinguish theoretical categories

---

## Term Expansion Process

### Step 1: Seed Terms → Similar Words

For each seed term:
```python
model.get_nearest_neighbors(seed_word, k=50)
```

This finds 50 most similar words using FastText Swedish embeddings (cc.sv.300.bin).

**Example for "risk":**
- risken (0.8148) - "the risk"
- riskbedömningen (0.8459) - "the risk assessment"
- riskanalys (0.7234) - "risk analysis"
- säkerhetsrisk (0.7156) - "security risk"
- ...

### Step 2: Combine & Deduplicate

Within each category:
- Combine similar words from all seed terms
- Keep highest similarity score if word appears multiple times
- Example: "risken" might appear for both "risk" and "riskanalys"

### Step 3: Lemmatization

All expanded terms are lemmatized using Stanza:
```python
stanza.Pipeline('sv', processors='tokenize,pos,lemma')
```

**Example:**
- risken → risk
- riskerna → risk
- riskbedömningen → riskbedömning

### Step 4: Deduplication After Lemmatization

Remove duplicate lemmas within categories:
- Before: 300 expanded words
- After lemmatization: 150-200 unique lemmas

**Result:** Each category has 100-300 unique expanded term lemmas.

---

## Expected Expansion Results

Based on similar Swedish text analysis projects:

| Category | Seed Terms | Expected Expanded Lemmas | Coverage in Corpus |
|----------|------------|--------------------------|-------------------|
| Risk | 9 | 150-200 | 70-80% of sentences |
| Accountability | 5 | 120-150 | 25-35% of sentences |
| Complexity | 7 | 180-220 | 30-40% of sentences |
| Efficiency | 7 | 150-180 | 15-25% of sentences |
| Equality | 5 | 80-120 | 3-8% of sentences |
| Agency | 7 | 250-350 | 60-70% of sentences |

**Note:** Percentages are not mutually exclusive - sentences can contain multiple categories.

---

## Quality Control Checks

After expansion, verify:

### 1. Semantic Coherence
Top 20 expanded terms should be semantically related to seed terms.

**Good example (Risk):**
- Top terms: risken, riskbedömning, säkerhetsrisk, riskanalys, sårbarhet

**Bad example:**
- Top terms: restaurant, building, computer (→ model issue or corpus contamination)

### 2. No Excessive Overlap
Different categories should have mostly distinct terms.

**Check:**
```python
overlap = set(risk_lemmas) & set(efficiency_lemmas)
overlap_pct = len(overlap) / min(len(risk_lemmas), len(efficiency_lemmas))
# Should be < 10%
```

### 3. Reasonable Frequencies
Expanded terms should appear with reasonable frequency.

**Check in corpus:**
- Not too rare: >5 occurrences
- Not too common: <50% of sentences (except "risk")

### 4. Lemmatization Correctness
Spot-check lemmas are correct Swedish base forms.

**Check:**
- effektivering → effektivering ✓
- effektiverings → effektivering ✓
- effektiveringar → effektivering ✓

---

## Using Expanded Terms

### In Sentence Filtering (sentencefiltering.ipynb)

```python
# Load expanded terms
df_expanded = pd.read_csv('expanded_terms_lemmatized_complete.csv')

# Create category lookup
category_lemmas = {}
for category in df_expanded['category'].unique():
    lemmas = set(df_expanded[df_expanded['category'] == category]['lemma'])
    category_lemmas[category] = lemmas

# Filter sentences
def sentence_has_category(sentence_text, category_lemmas):
    words = sentence_text.split()
    return any(word in category_lemmas for word in words)
```

### In Co-occurrence Analysis (cooccurrence_analysis.py)

```python
# Create binary indicators
for category in ['risk', 'accountability', 'complexity', 'efficiency', 'equality']:
    col = f'has_{category}'
    sentences_with_cat = set(df[df['category'] == category]['sentence_id'])
    df_binary[col] = df_binary['sentence_id'].isin(sentences_with_cat).astype(int)
```

---

## Theoretical Justification by Category

### Risk → Core Organizing Concept
**Theory:** Risk analysis as a technology of government (Foucault, Beck)
**Test:** Frequency, ubiquity, co-occurrence with all other categories

### Accountability → Legitimacy Effects
**Theory:** Risk creates responsibility structures (Power, 2007)
**Test:** H2 - Which actors are held accountable? Association with "kommun"?

### Complexity → Structural Effects
**Theory:** Risk reveals interdependencies (Perrow, 1984)
**Test:** H3 - Does complexity discourse increase over time? H4 - Interacts with efficiency?

### Efficiency → Rationalization Effects
**Theory:** Risk enables optimization discourse (Miller & Rose, 1990)
**Test:** H1 - Is efficiency prominent? H2 - Associates with "kommun"?

### Equality → Standardization Effects
**Theory:** Risk creates commensurable categories (Espeland & Stevens, 2008)
**Test:** H1 - How prominent? Association with state-level actors?

---

## References

### Theoretical Framework
- Power, M. (2007). *Organized Uncertainty: Designing a World of Risk Management*
- Beck, U. (1992). *Risk Society: Towards a New Modernity*
- Perrow, C. (1984). *Normal Accidents: Living with High-Risk Technologies*
- Miller, P., & Rose, N. (1990). "Governing Economic Life"
- Espeland, W., & Stevens, M. (2008). "Commensuration as a Social Process"

### Methodological
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases"
- Grave, E., et al. (2018). "Learning Word Vectors for 157 Languages"
- Qi, P., et al. (2020). "Stanza: A Python Natural Language Processing Toolkit"

---

## Version History

**Version 2.0** (2025-01-02)
- Updated Risk: Added "kritiska, beroenden, krisberedskap, samhällsviktig, verksamhet"
- Updated Complexity: Removed "beroende, ömsesidighet", added "annorlunda, unik"
- Updated Equality: Added "utbytbar"
- Rationale: Refined based on theoretical clarity and semantic distinctiveness

**Version 1.0** (2024-12-31)
- Initial seed terms based on theoretical framework
- 5 political effect categories + 1 actor category
- Total: 33 seed terms

---

## Contact & Questions

For questions about seed term selection or expansion methodology, refer to:
- `METHODOLOGY_SUMMARY.md` - Full methodological documentation
- `COMPLETE_PIPELINE_GUIDE.md` - Step-by-step usage guide
- `vectoranalysis_complete.ipynb` - Implementation code
