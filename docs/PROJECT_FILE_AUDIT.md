# Project File Audit Report

**Date**: 2025-01-04  
**Repository**: Text-as-data-master-thesis-repo

## 📊 Summary

**Total files**: 23 files  
**Status**:
- ✅ **Keep & Current**: 6 files
- ⚠️ **Needs Update**: 3 files
- 🔄 **Redundant/Deprecated**: 4 files
- 📁 **Data Files**: 7 files (need organization)
- 📄 **Sample/Reference**: 3 files

---

## ✅ Files to KEEP (Already Current)

### 1. **vectoranalysis.py** ✓
- **Status**: CURRENT (V2.0 with updated seed terms)
- **Location**: Should move to `scripts/03_expansion/`
- **Size**: 727 lines, well-documented
- **Seed Terms**: V2.0 (risk: 9 terms, complexity: 7 terms, equality: 5 terms)
- **Action**: KEEP, just move to proper location

### 2. **cooccurrence_analysis.py** ✓
- **Status**: CURRENT
- **Location**: Should move to `scripts/05_analysis/`
- **Size**: 35KB
- **Features**: Chi-square tests, temporal analysis, visualizations
- **Action**: KEEP, move to proper location

### 3. **ocr_swedish_pdfs_improved.py** ✓
- **Status**: CURRENT
- **Location**: Should move to `scripts/01_ocr/`
- **Size**: ~1,400 lines
- **Purpose**: Main OCR processing script
- **Action**: KEEP, move to proper location

### 4. **run_ocr.py** ✓
- **Status**: CURRENT
- **Location**: Should move to `scripts/01_ocr/`
- **Purpose**: Simple runner script
- **Action**: KEEP, move to proper location

### 5. **METHODOLOGY_SUMMARY.md** ✓
- **Status**: CURRENT
- **Location**: Should move to `docs/`
- **Purpose**: Detailed methodology documentation
- **Action**: KEEP, move to proper location

### 6. **SEED_TERMS_REFERENCE.md** ✓
- **Status**: CURRENT (Version 2.0)
- **Location**: Should move to `docs/`
- **Purpose**: Complete seed term documentation with justifications
- **Action**: KEEP, move to proper location

---

## ⚠️ Files that NEED UPDATES

### 1. **expanded_terms_lemmatized.csv** ⚠️
- **Status**: OUTDATED (Only 2 categories!)
- **Current content**: 
  - resilience: 109 terms
  - risk: 114 terms
  - **Missing**: accountability, complexity, efficiency, equality
- **Problem**: Generated with old seed terms (V1.0)
- **Action**: 
  - ❌ DELETE this file
  - ✅ REGENERATE using `vectoranalysis.py` with V2.0 seed terms
  - ✅ Should produce ~1,000-1,500 terms across 6 categories
- **Command to regenerate**:
  ```bash
  python vectoranalysis.py --model-path /path/to/cc.sv.300.bin
  ```

### 2. **sentence_vectors_metadata.csv** ⚠️
- **Status**: NEEDS REGENERATION
- **Current content**: 47,570 sentences with only 2 categories (resilience, risk)
- **Problem**: Filtered using old expanded_terms_lemmatized.csv
- **Action**: 
  - Keep for now as reference
  - REGENERATE after updating expanded terms
  - Use `sentencefiltering.py` (needs conversion from .ipynb)
- **Expected result**: ~15,000-20,000 sentences with all 6 categories

### 3. **sentences_with_negation_flags.csv** ⚠️
- **Status**: NEEDS REGENERATION
- **Same issue**: Based on old 2-category data
- **Action**: REGENERATE after fixing upstream data

---

## 🔄 Files that are REDUNDANT/DEPRECATED

### 1. **vectoranalysis.ipynb** 🔄
- **Status**: DEPRECATED (replaced by vectoranalysis.py)
- **Problem**: Jupyter notebook, harder to version control
- **Action**: 
  - Move to `archive/notebooks/` OR
  - DELETE (you have the .py version)
- **Recommendation**: ARCHIVE, don't delete yet (might want to reference)

### 2. **sentencefiltering.ipynb** 🔄
- **Status**: NEEDS CONVERSION to .py
- **Action**: 
  - Convert to `scripts/04_filtering/sentencefiltering.py`
  - Then archive the notebook
- **Priority**: HIGH (needed to regenerate data)

### 3. **readingtexts.ipynb** 🔄
- **Status**: DEPRECATED (preprocessing done)
- **Action**: Move to `archive/notebooks/`
- **Reason**: This was early preprocessing, not needed in normal pipeline

### 4. **OCR_pipeline.ipynb** 🔄
- **Status**: SUPERSEDED by ocr_swedish_pdfs_improved.py
- **Action**: Move to `archive/notebooks/`
- **Reason**: Replaced by better .py implementation

---

## 📁 DATA FILES (Need Organization)

### 1. **sentence_vectors_index.csv** (2.5 MB)
- **Status**: Old data
- **Location**: Move to `data/vectors/` (gitignore)
- **Action**: Keep temporarily, regenerate with new categories

### 2. **sentence_vectors_metadata.csv** (9.5 MB)
- **Status**: Old data (2 categories only)
- **Location**: Move to `data/vectors/` (gitignore)
- **Action**: Regenerate after fixing expanded terms

### 3. **sentences_with_negation_flags.csv** (9.9 MB)
- **Status**: Old data (2 categories only)
- **Location**: Move to `data/vectors/` (gitignore)
- **Action**: Regenerate

### 4. **expanded_terms_lemmatized.csv** (12.8 KB)
- **Status**: OUTDATED - DELETE
- **Action**: Regenerate with vectoranalysis.py

---

## 📄 SAMPLE/REFERENCE FILES

### 1. **RSA_Arvidsjaur_2019_Maskad.pdf** (2.3 MB)
- **Status**: Sample document
- **Location**: Move to `data/raw/pdfs/samples/`
- **Action**: KEEP for testing

### 2. **Summary_creating_paragraph_dataset.pdf** (281 KB)
- **Status**: Reference document
- **Location**: Move to `docs/references/`
- **Action**: KEEP

### 3. **README_OCR.ipynb** (15 KB)
- **Status**: Documentation
- **Action**: Convert to `docs/README_OCR.md`

---

## 🚨 CRITICAL ISSUES IDENTIFIED

### Issue 1: Wrong Categories in Data ⚠️⚠️⚠️
**Problem**: Your expanded terms only have "resilience" and "risk"
- Should have: risk, accountability, complexity, efficiency, equality, agency
- Current: resilience, risk

**Why this happened**: 
- `expanded_terms_lemmatized.csv` was generated with OLD seed terms
- Used "resilience" instead of proper categories

**Impact**: 
- ALL filtered sentence data is incomplete
- Missing 4 out of 6 categories
- Cannot test hypotheses H2, H3, H4 properly

**Solution**:
```bash
# 1. Delete old expanded terms
rm expanded_terms_lemmatized.csv

# 2. Regenerate with new script (has V2.0 seed terms)
python vectoranalysis.py --model-path /path/to/cc.sv.300.bin

# 3. Convert sentencefiltering.ipynb to .py
# 4. Regenerate sentence vectors with new expanded terms
```

### Issue 2: Missing Scripts
- ❌ No `sentencefiltering.py` (only .ipynb)
- ❌ No `data_diagnostic.py` (needed for QA)
- ❌ Missing analysis scripts

**Solution**: Download from outputs/ and organize:
- `data_diagnostic.py` → `scripts/05_analysis/`
- Convert `sentencefiltering.ipynb` → `scripts/04_filtering/sentencefiltering.py`

### Issue 3: No Directory Structure
Everything is in root directory - hard to navigate

**Solution**: Create proper structure (see recommended organization below)

---

## 📋 RECOMMENDED ACTIONS (Priority Order)

### 🔥 CRITICAL (Do First)

1. **Create directory structure**
   ```bash
   mkdir -p scripts/{01_ocr,02_preprocessing,03_expansion,04_filtering,05_analysis}
   mkdir -p docs/{references,guides}
   mkdir -p data/{raw/pdfs,processed,expanded_terms,vectors}
   mkdir -p results/{cooccurrence,figures}
   mkdir -p archive/notebooks
   ```

2. **Move current files to proper locations**
   ```bash
   # Scripts
   mv vectoranalysis.py scripts/03_expansion/
   mv cooccurrence_analysis.py scripts/05_analysis/
   mv ocr_swedish_pdfs_improved.py scripts/01_ocr/
   mv run_ocr.py scripts/01_ocr/
   
   # Documentation
   mv METHODOLOGY_SUMMARY.md docs/
   mv SEED_TERMS_REFERENCE.md docs/
   mv SEED_TERMS_UPDATE_V2.md docs/
   
   # Sample data
   mv RSA_Arvidsjaur_2019_Maskad.pdf data/raw/pdfs/samples/
   mv Summary_creating_paragraph_dataset.pdf docs/references/
   
   # Old data (temporary)
   mv sentence_vectors_*.csv data/vectors/
   mv sentences_with_negation_flags.csv data/vectors/
   
   # Archive notebooks
   mv vectoranalysis.ipynb archive/notebooks/
   mv OCR_pipeline.ipynb archive/notebooks/
   mv readingtexts.ipynb archive/notebooks/
   mv README_OCR.ipynb archive/notebooks/
   ```

3. **Delete outdated data**
   ```bash
   # This has wrong categories!
   rm expanded_terms_lemmatized.csv
   ```

4. **Download missing scripts from outputs/**
   - `data_diagnostic.py` → `scripts/05_analysis/`
   - `cooccurrence_analysis_notebook.ipynb` → `notebooks/`
   - `COMPLETE_PIPELINE_GUIDE.md` → `docs/guides/`
   - `README_COOCCURRENCE.md` → `docs/guides/`

### ⚠️ HIGH PRIORITY (Do Soon)

5. **Regenerate expanded terms (CRITICAL)**
   ```bash
   python scripts/03_expansion/vectoranalysis.py \
       --model-path /path/to/cc.sv.300.bin \
       --output data/expanded_terms/expanded_terms_lemmatized_complete.csv
   ```
   
   **Expected output**: ~1,000-1,500 terms across 6 categories

6. **Convert sentencefiltering.ipynb to .py**
   - Need this to regenerate sentence vectors
   - Should I create this script for you?

7. **Regenerate sentence vectors**
   ```bash
   python scripts/04_filtering/sentencefiltering.py
   ```

### 📝 MEDIUM PRIORITY

8. **Create .gitignore**
   ```bash
   cat > .gitignore << 'EOF'
   # Data files
   data/vectors/*.csv
   data/vectors/*.parquet
   data/processed/*.rds
   *.npy
   
   # Models
   *.bin
   
   # Results
   results/cooccurrence/*.csv
   results/figures/*.png
   
   # Python
   __pycache__/
   *.pyc
   .ipynb_checkpoints/
   
   # macOS
   .DS_Store
   EOF
   ```

9. **Create project README**
   - Download from outputs/

10. **Commit organized structure**
    ```bash
    git add .
    git commit -m "Reorganize project structure and remove outdated data"
    ```

---

## 🎯 EXPECTED STATE AFTER CLEANUP

```
Text-as-data-master-thesis-repo/
├── README.md
├── .gitignore
│
├── scripts/
│   ├── 01_ocr/
│   │   ├── ocr_swedish_pdfs_improved.py
│   │   └── run_ocr.py
│   ├── 03_expansion/
│   │   └── vectoranalysis.py
│   ├── 04_filtering/
│   │   └── sentencefiltering.py (NEEDS CREATION)
│   └── 05_analysis/
│       ├── cooccurrence_analysis.py
│       └── data_diagnostic.py (NEEDS DOWNLOAD)
│
├── docs/
│   ├── METHODOLOGY_SUMMARY.md
│   ├── SEED_TERMS_REFERENCE.md
│   ├── SEED_TERMS_UPDATE_V2.md
│   ├── guides/
│   │   ├── COMPLETE_PIPELINE_GUIDE.md
│   │   └── README_COOCCURRENCE.md
│   └── references/
│       └── Summary_creating_paragraph_dataset.pdf
│
├── data/
│   ├── raw/pdfs/samples/
│   │   └── RSA_Arvidsjaur_2019_Maskad.pdf
│   ├── expanded_terms/
│   │   └── expanded_terms_lemmatized_complete.csv (REGENERATE)
│   └── vectors/
│       └── (old CSV files, will be regenerated)
│
├── archive/
│   └── notebooks/
│       ├── vectoranalysis.ipynb
│       ├── sentencefiltering.ipynb (keep until converted)
│       ├── OCR_pipeline.ipynb
│       └── readingtexts.ipynb
│
└── notebooks/
    └── cooccurrence_analysis_notebook.ipynb
```

---

## 📊 File Statistics

| Category | Count | Status |
|----------|-------|--------|
| Current Python scripts | 4 | ✅ Good |
| Deprecated notebooks | 4 | 🔄 Archive |
| Outdated data files | 4 | ⚠️ Regenerate |
| Documentation | 3 | ✅ Good |
| Missing critical scripts | 2 | ❌ Need download |

---

## 🚀 Next Steps

**Immediate (This Session):**
1. Should I convert `sentencefiltering.ipynb` to a proper .py script?
2. Should I create a comprehensive .gitignore file?
3. Should I create a project README?

**After Cleanup:**
1. Run vectoranalysis.py to regenerate expanded terms
2. Run sentencefiltering.py to regenerate sentence vectors
3. Run data_diagnostic.py to verify all 6 categories present
4. Run cooccurrence_analysis.py for chi-square tests

---

**Summary**: You have most of the code, but your DATA is outdated. The expanded terms only have 2 categories instead of 6, so all downstream analysis is incomplete. Priority is to reorganize, regenerate expanded terms, and regenerate sentence vectors.
