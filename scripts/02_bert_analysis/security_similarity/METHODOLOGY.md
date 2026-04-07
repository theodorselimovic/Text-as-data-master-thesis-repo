# Methodology: Isomorphism Analysis of Security Risk Framing

This document describes the methodology for measuring institutional isomorphism in Swedish Risk and Vulnerability Analyses (RSA), designed for inclusion in the thesis.

## 1. Research Question

**Does municipal discourse on security risks show institutional isomorphism with central government (MSB) and regional authorities (prefectures)?**

The analysis tests whether municipalities copy MSB/prefecture framing on security risks versus adapting discourse to local circumstances. Security risks (cyber threats, disinformation, military threats) may show higher isomorphism than other risks due to:
- Less local expertise on novel security threats
- More standardized MSB guidance
- Higher perceived uncertainty about appropriate responses

## 2. Text Unit Definition

### 2.1 Paragraph Tagging by Risk Category

Each paragraph in the corpus is assigned to a dominant risk category based on term frequency matching against the RISK_DICTIONARY (a curated dictionary of Swedish risk terms across 13 categories).

**Process:**
1. For each paragraph, count occurrences of terms from each risk category
2. Assign paragraph to category with highest term count (if >= 1 term present)
3. Paragraphs with no risk terms remain unassigned

### 2.2 Security Risks Analyzed

Two risk categories are analyzed as "security risks":

| Category | Swedish Terms (examples) | English Translation |
|----------|-------------------------|---------------------|
| `cyber_hot` | cyberattack, ransomware, dataintrång, DDoS-attack | Cyber threats |
| `antagonistiska_hot` | terrorism, sabotage, desinformation, hybridhot, väpnat angrepp, krig | Antagonistic threats (including military) |

### 2.3 Text Unit Aggregation

For each document, the "text unit" for a risk category consists of all sentences from paragraphs assigned to that category. A minimum threshold of 5 sentences is required for inclusion in the analysis.

## 3. Embedding Model

### 3.1 Model Selection

**Model:** `KBLab/sentence-bert-swedish-cased`

This is a Swedish Sentence-BERT model trained by the Royal Library of Sweden (Kungliga biblioteket), optimized for semantic similarity tasks in Swedish text.

**Key properties:**
- Embedding dimension: 768
- Training: Siamese BERT architecture trained on Swedish paraphrase data
- Source: https://huggingface.co/KBLab/sentence-bert-swedish-cased

### 3.2 Why Sentence-BERT?

Unlike standard BERT (optimized for classification), Sentence-BERT produces embeddings that:
1. Capture semantic meaning at the sentence level
2. Are comparable via cosine similarity
3. Preserve semantic relationships (similar sentences cluster together)

This makes SBERT suitable for measuring textual similarity without requiring task-specific fine-tuning.

## 4. Similarity Measures

Two complementary measures are used to capture different aspects of textual similarity:

### 4.1 Max-Match Averaging

**Definition:** For each sentence in the source text (municipality), find the maximum cosine similarity to any sentence in the target text (reference). Average these maxima across all source sentences.

**Formula:**
```
MaxMatch(A, B) = (1/|A|) * Σ_{a ∈ A} max_{b ∈ B} cos(a, b)
```

Where:
- A = set of sentence embeddings from municipality text
- B = set of sentence embeddings from reference text
- cos(a, b) = cosine similarity between embeddings

**Interpretation:**
- Captures "borrowing" - how well can each municipality sentence be matched to reference language?
- High score indicates the municipality uses similar formulations to the reference
- **Range:** [0, 1] where 1 = perfect alignment

**Asymmetry:** The measure is directional (A → B). We compute municipality → reference direction, measuring how well municipality text can be "explained" by reference text.

### 4.2 Earth Mover's Distance (EMD)

**Definition:** Treats each text as an empirical distribution over sentence embeddings. EMD computes the minimum "work" required to transform one distribution into another, where work is the sum of (mass moved × distance moved).

**Formula:**
```
EMD(A, B) = min_T Σ_{i,j} T_{ij} * d(a_i, b_j)
```

Where:
- T = transport plan (how much mass to move from a_i to b_j)
- d(a_i, b_j) = cosine distance = 1 - cos(a_i, b_j)
- Subject to: T must be a valid transport plan (row sums = 1/|A|, column sums = 1/|B|)

**Implementation:** Uses the POT (Python Optimal Transport) library's `emd2` function for efficient computation.

**Interpretation:**
- Measures distributional difference between the two texts
- Low EMD indicates similar "coverage" of semantic space
- **Range:** [0, 2] where 0 = identical distributions (using cosine distance)

**Symmetry:** EMD is symmetric (EMD(A,B) = EMD(B,A)), unlike max-match.

### 4.3 Complementary Perspectives

| Measure | Captures | Sensitive to |
|---------|----------|--------------|
| Max-Match | Best-case alignment (borrowing) | Individual sentence matches |
| EMD | Overall distributional similarity | Full semantic coverage |

A high max-match with high EMD suggests selective borrowing (some sentences match well, but overall coverage differs). Low max-match with low EMD is theoretically unlikely. High max-match with low EMD indicates comprehensive isomorphism.

## 5. Comparison Structure

### 5.1 Primary Comparisons

For each municipality document containing security risk paragraphs:

| Comparison | Source | Target |
|------------|--------|--------|
| Municipality → MSB | Municipality security paragraphs | MSB security paragraphs (nearest year) |
| Municipality → Prefecture | Municipality security paragraphs | Prefecture security paragraphs (same län, nearest year) |

Prefecture matching uses the standard Swedish administrative mapping (each municipality belongs to one of 21 län/counties, each with its own länsstyrelse/prefecture).

### 5.2 Baseline Comparisons

Two baselines establish the null distribution against which target similarity is compared:

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Within-document | 1 random paragraph from same RSA, different risk category | Controls for document-level writing style, formatting, boilerplate |
| Cross-municipality | 1 paragraph from different municipality, same risk category | Measures horizontal similarity (peer-to-peer) independent of vertical diffusion |

### 5.3 Why These Baselines?

**Within-document baseline:** If municipalities simply have similar writing styles to MSB (formal administrative Swedish), all comparisons would show high similarity. Comparing to a different-topic paragraph within the same document controls for this.

**Cross-municipality baseline:** If all municipalities discuss security risks similarly (regardless of MSB influence), this would show comparable similarity to the target comparison. This baseline tests whether municipalities are more similar to MSB/prefecture than to each other.

## 6. Isomorphism Index

### 6.1 Definition

The isomorphism index normalizes the target similarity against the within-document baseline:

```
IsomorphismIndex = (TargetSimilarity - WithinDocBaseline) / (1 - WithinDocBaseline)
```

### 6.2 Interpretation

| Index Value | Interpretation |
|-------------|----------------|
| > 0 | Higher similarity to reference than to random within-document text |
| = 0 | Same similarity to reference as to random text (no isomorphism signal) |
| < 0 | Lower similarity to reference than to random text (unexpected) |
| → 1 | Near-perfect alignment with reference |

### 6.3 Advantages

1. **Comparability:** Allows comparison across documents with different baseline styles
2. **Bounded:** Results are interpretable on a consistent scale
3. **Controls:** Implicitly controls for document-level confounds

## 7. Computational Approach

### 7.1 Sampling Strategy

To reduce computational cost, only sentences that will actually be compared are embedded:

1. **Target paragraphs:** Municipality paragraphs tagged with security risk categories
2. **Reference paragraphs:** Matching MSB/prefecture paragraphs
3. **Baseline paragraphs:** One sampled paragraph per baseline type

This reduces embedding requirements from ~438K sentences to ~20K sentences (95% reduction).

### 7.2 Year Matching

When no exact year match exists for reference documents, the nearest available year is used (e.g., a 2017 municipality document matches a 2016 MSB document if 2017 is unavailable).

## 8. Limitations

1. **Paragraph tagging imprecision:** A paragraph may discuss multiple risks; dominant-category assignment may miss nuance
2. **Sentence boundaries:** OCR quality affects sentence segmentation accuracy
3. **Sampling variance:** Single-sample baselines introduce variance; results should be interpreted as point estimates
4. **Temporal ordering:** The analysis measures similarity, not causality; high similarity does not prove diffusion direction

## 9. References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP 2019*.
- Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport. *Foundations and Trends in Machine Learning*, 11(5-6), 355-607.
- KBLab Swedish BERT models: https://huggingface.co/KBLab
- POT (Python Optimal Transport): https://pythonot.github.io/
