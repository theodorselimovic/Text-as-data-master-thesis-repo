#!/bin/bash

# Test script for BOW analysis pipeline
# Tests wave mapping, lemmatization, and low-N flagging

echo "============================================================"
echo "BOW ANALYSIS PIPELINE TEST"
echo "============================================================"
echo ""
echo "This script tests the three improvements:"
echo "1. Wave mapping (years → waves 1, 2, 3)"
echo "2. Lemmatization (merges inflectional variants)"
echo "3. Low-N flagging (flags metrics with n<3 entities)"
echo ""
echo "============================================================"
echo ""

# Change to scripts directory
cd "$(dirname "$0")/scripts/03_bow_analysis" || exit 1

# Create results directory
mkdir -p ../../results/term_document_matrix
mkdir -p ../../results/test_output

echo "Step 1: Creating term-document matrices (original + lemmatized)"
echo "------------------------------------------------------------"
python term_document_matrix.py \
    --texts ../../data/merged/pdf_texts_all_actors.parquet \
    --output ../../results/term_document_matrix/ \
    2>&1 | tee ../../results/test_output/01_matrix_creation.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Matrix creation completed successfully"
    echo ""
else
    echo "✗ Matrix creation failed"
    exit 1
fi

# Check outputs
echo "Verifying output files..."
echo "------------------------------------------------------------"
ls -lh ../../results/term_document_matrix/*.csv 2>/dev/null | awk '{print $9, "(" $5 ")"}'
ls -lh ../../results/term_document_matrix/*.json 2>/dev/null | awk '{print $9, "(" $5 ")"}'
echo ""

# Analyze wave distribution
echo "Step 2: Analyzing wave distribution"
echo "------------------------------------------------------------"
python -c "
import pandas as pd
df = pd.read_csv('../../results/term_document_matrix/term_document_matrix.csv')
print('Wave distribution:')
print(df.groupby('wave')['year'].agg(['min', 'max', 'count']))
print('')
print(f'Total documents: {len(df)}')
print(f'Actors: {df[\"actor\"].unique().tolist()}')
" 2>&1 | tee -a ../../results/test_output/02_wave_analysis.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Wave analysis completed"
    echo ""
else
    echo "✗ Wave analysis failed"
fi

# Check lemmatization
echo "Step 3: Checking lemmatization results"
echo "------------------------------------------------------------"
if [ -f "../../results/term_document_matrix/lemma_mapping.json" ]; then
    python -c "
import json
with open('../../results/term_document_matrix/lemma_mapping.json') as f:
    mapping = json.load(f)

# Show statistics
print(f'Total unique lemmas: {len(mapping)}')
print(f'Total original terms: {sum(len(v) for v in mapping.values())}')
reduction = (1 - len(mapping) / sum(len(v) for v in mapping.values())) * 100
print(f'Term reduction: {reduction:.1f}%')
print('')

# Show examples
print('Example mappings (merged variants):')
examples = [(k, v) for k, v in mapping.items() if len(v) > 1][:5]
for lemma, originals in examples:
    print(f'  {lemma} ← {originals}')
" 2>&1 | tee -a ../../results/test_output/03_lemmatization_check.log
    echo "✓ Lemmatization check completed"
else
    echo "⚠ Lemma mapping file not found"
fi
echo ""

# Run persistence analysis (if enough data)
echo "Step 4: Testing persistence analysis with low-N flagging"
echo "------------------------------------------------------------"
python risk_persistence_analysis.py \
    --input ../../results/term_document_matrix/term_document_matrix.csv \
    --output ../../results/test_output/persistence/ \
    --min-entities 3 \
    2>&1 | head -50 | tee ../../results/test_output/04_persistence_test.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Persistence analysis completed"
    echo ""

    # Check low-N flags
    if [ -f "../../results/test_output/persistence/risk_persistence_by_term.csv" ]; then
        python -c "
import pandas as pd
df = pd.read_csv('../../results/test_output/persistence/risk_persistence_by_term.csv')
if 'flag_low_n' in df.columns:
    flagged = df['flag_low_n'].sum()
    total = len(df)
    print(f'Low-N flags: {flagged}/{total} metrics flagged ({flagged/total*100:.1f}%)')
    print('')
    print('Sample flagged metrics:')
    print(df[df['flag_low_n'] == True][['term', 'n_entities_t0', 'n_entities_persist', 'persistence_rate']].head(5).to_string(index=False))
else:
    print('flag_low_n column not found')
" 2>&1 | tee -a ../../results/test_output/04_persistence_test.log
    fi
else
    echo "⚠ Persistence analysis skipped or failed (may need more data)"
fi
echo ""

# Run clustering analysis
echo "Step 5: Testing clustering with wave-based filtering"
echo "------------------------------------------------------------"
python risk_clustering_analysis.py \
    --input ../../results/term_document_matrix/category_document_matrix.csv \
    --output ../../results/test_output/clustering/ \
    --waves 1 2 3 \
    2>&1 | head -100 | tee ../../results/test_output/05_clustering_test.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Clustering analysis completed"
else
    echo "⚠ Clustering analysis skipped or failed (may need more data)"
fi
echo ""

echo "============================================================"
echo "PIPELINE TEST COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: results/test_output/"
echo ""
echo "Summary of outputs:"
echo "  - Term matrices (original + lemmatized): results/term_document_matrix/"
echo "  - Lemma mapping: results/term_document_matrix/lemma_mapping.json"
echo "  - Test logs: results/test_output/*.log"
echo "  - Persistence results: results/test_output/persistence/"
echo "  - Clustering results: results/test_output/clustering/"
echo ""
