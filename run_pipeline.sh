#!/usr/bin/env bash
# Full RSA pipeline — extraction → preprocessing → analysis
#
# Usage (overnight run — prevents Mac sleep):
#   caffeinate -i bash run_pipeline.sh
#
# The script logs everything to logs/pipeline_TIMESTAMP.log.
# Critical steps abort on failure; analysis steps warn and continue.

REPO="/Users/theodorselimovic/Library/CloudStorage/OneDrive-Personal/Sciences Po/Master Thesis/Text analysis code/Text-as-data-master-thesis-repo"
PDF_BASE="/Users/theodorselimovic/Sciences Po/Material/Risk analyses"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$REPO/logs/pipeline_${TIMESTAMP}.log"
mkdir -p "$REPO/logs"

# Tee all output to log
exec > >(tee -a "$LOG") 2>&1

cd "$REPO"

# ── helpers ──────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo; echo "════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════"; }
die()  { log "FATAL: $*"; exit 1; }
warn() { log "WARNING: $* — continuing"; }

log "Pipeline started. Log: $LOG"

# ── 1. PDF EXTRACTION ────────────────────────────────────────────────────────
# Each extraction merges with the previous output so the final mcf parquet
# contains everything. The --merge-with flag deduplicates on filename,
# keeping the freshly extracted version (keep='last'). Old OCR data for
# files not re-extracted is preserved from data/merged/pdf_texts_all_actors.parquet.
mkdir -p data/merged

step "1 · PDF extraction — municipalities (merges with existing corpus)"
python scripts/01_pdf_extraction/pdf_reader_enhanced.py \
    --input-dir "$PDF_BASE/Kommunala RSA" \
    --output-dir data/raw/ \
    --actor kommun \
    --ocr \
    --merge-with data/merged/pdf_texts_all_actors.parquet \
    --verbose \
    || die "Municipality extraction failed"

# Merge municipality output into main corpus
cp data/raw/pdf_texts.parquet data/merged/pdf_texts_all_actors.parquet

step "2 · PDF extraction — prefectures (merges with existing corpus)"
python scripts/01_pdf_extraction/pdf_reader_enhanced.py \
    --input-dir "$PDF_BASE/Länsstyrelser RSA" \
    --output-dir data/raw/lansstyrelse_extraction/ \
    --actor länsstyrelse \
    --ocr \
    --merge-with data/merged/pdf_texts_all_actors.parquet \
    --verbose \
    || die "Prefecture extraction failed"

# Merge prefecture output into main corpus
cp data/raw/lansstyrelse_extraction/pdf_texts.parquet data/merged/pdf_texts_all_actors.parquet

step "3 · PDF extraction — MSB (merges with existing corpus)"
python scripts/01_pdf_extraction/pdf_reader_enhanced.py \
    --input-dir "$PDF_BASE/MSB NRSB" \
    --output-dir data/raw/mcf_extraction/ \
    --actor MCF \
    --ocr \
    --merge-with data/merged/pdf_texts_all_actors.parquet \
    --verbose \
    || die "MSB extraction failed"

# ── 2. PROMOTE FINAL MERGED OUTPUT ───────────────────────────────────────────
step "4 · Promote merged corpus → data/merged/pdf_texts_all_actors.parquet"
cp data/raw/mcf_extraction/pdf_texts.parquet data/merged/pdf_texts_all_actors.parquet \
    || die "Failed to promote merged corpus"
python3 -c "
import pandas as pd
df = pd.read_parquet('data/merged/pdf_texts_all_actors.parquet')
print(f'  Total: {len(df)} docs')
print(df['actor'].value_counts().to_string())
"

# ── 3. PREPROCESSING ─────────────────────────────────────────────────────────
step "5 · BERT preprocessing (sentence segmentation + quality filter)"
python scripts/02_preprocessing/preprocessing_bert.py \
    --input  data/merged/pdf_texts_all_actors.parquet \
    --output data/processed/bert_corpus.parquet \
    --verbose \
    || die "BERT preprocessing failed"

step "6 · BOW preprocessing (stemming + n-grams  ·  ~15 min)"
python scripts/02_preprocessing/preprocessing_bow.py \
    --input  data/processed/bert_corpus.parquet \
    --output data/processed/bow_corpus_stemmed.parquet \
    --verbose \
    || die "BOW preprocessing failed"

# ── 4. BOW ANALYSIS ──────────────────────────────────────────────────────────
TERM_MATRIX="results/01_bow_analysis/term_matrices/term_document_matrix.csv"
CAT_MATRIX="results/01_bow_analysis/term_matrices/category_document_matrix.csv"

step "8 · BOW analysis"

log "8a · Dictionary counter — term and category matrices"
python scripts/03_bow_analysis/risk_dictionary_counter.py \
    --input   data/processed/bow_corpus_stemmed.parquet \
    --output  results/01_bow_analysis/term_matrices/ \
    --verbose \
    || die "Dictionary counter failed"

log "8b · Persistence analysis"
python scripts/03_bow_analysis/risk_persistence_analysis.py \
    --input  "$TERM_MATRIX" \
    --output results/01_bow_analysis/persistence/ \
    --verbose \
    || warn "Persistence analysis failed"

log "8d · Prevalence analysis"
python scripts/03_bow_analysis/risk_prevalence_analysis.py \
    --corpus data/processed/bow_corpus_stemmed.parquet \
    --output results/01_bow_analysis/prevalence/ \
    || warn "Prevalence analysis failed"

log "8e · Diffusion analysis"
python scripts/03_bow_analysis/risk_diffusion_analysis.py \
    --input  "$TERM_MATRIX" \
    --output results/01_bow_analysis/diffusion/ \
    --verbose \
    || warn "Diffusion analysis failed"

log "8f · Convergence analysis"
python scripts/03_bow_analysis/risk_convergence_analysis.py \
    --input  "$TERM_MATRIX" \
    --output results/01_bow_analysis/convergence/ \
    --verbose \
    || warn "Convergence analysis failed"

log "8g · Distinctiveness analysis (by wave)"
python scripts/03_bow_analysis/risk_distinctiveness_analysis.py \
    --input   "$TERM_MATRIX" \
    --output  results/01_bow_analysis/distinctiveness/ \
    --by-wave \
    --verbose \
    || warn "Distinctiveness analysis failed"

log "8h · Actor similarity analysis (PERMANOVA)"
python scripts/03_bow_analysis/actor_similarity_analysis.py \
    --input  "$CAT_MATRIX" \
    --output results/01_bow_analysis/actor_similarity/ \
    --verbose \
    || warn "Actor similarity analysis failed"

log "8i · Clustering analysis"
python scripts/03_bow_analysis/risk_clustering_analysis.py \
    --input  "$CAT_MATRIX" \
    --output results/01_bow_analysis/clustering/ \
    --verbose \
    || warn "Clustering analysis failed"

log "8j · Allvarliga störningar analysis"
python scripts/03_bow_analysis/allvarliga_storningar_analysis.py \
    --corpus data/processed/bow_corpus_stemmed.parquet \
    --output results/01_bow_analysis/allvarliga_storningar/ \
    || warn "Allvarliga störningar analysis failed"

log "8k · Security riskification analysis (Q6)"
python scripts/03_bow_analysis/security_riskification_analysis.py \
    --corpus data/processed/bow_corpus_stemmed.parquet \
    --output results/01_bow_analysis/security_riskification/ \
    --verbose \
    || warn "Security riskification analysis failed"

log "8l · Security adoption timing analysis"
python scripts/03_bow_analysis/security_adoption_timing.py \
    || warn "Security adoption timing analysis failed"

log "8m · Risk context analysis (qualifications)"
python scripts/03_bow_analysis/risk_context_analysis.py \
    --corpus data/processed/bow_corpus_stemmed.parquet \
    --output results/01_bow_analysis/context/ \
    || warn "Risk context analysis failed"

# ── 5. ISOMORPHISM ───────────────────────────────────────────────────────────
step "9 · Isomorphism analysis  (slow — BERT embeddings)"
python scripts/07_bert_analysis/security_similarity/isomorphism_analysis.py \
    --input   data/processed/bert_corpus.parquet \
    --output  results/07_bert_analysis/security_similarity/ \
    --verbose \
    || warn "Isomorphism analysis failed"

# ── DONE ─────────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Log → $LOG"
echo "════════════════════════════════════════════════"
