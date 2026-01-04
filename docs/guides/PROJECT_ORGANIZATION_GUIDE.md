# Project Organization & Workflow Guide

## 🎯 Current Problem

**Your current workflow:**
1. I create files in `/mnt/user-data/outputs/` (temporary Claude storage)
2. You download them manually
3. You move them to iCloud folder
4. Files get scattered and hard to track

**Issues:**
- ❌ I can't see what's in your iCloud folder
- ❌ Duplicated files (project vs. outputs vs. iCloud)
- ❌ Unclear which files are current vs. outdated
- ❌ Manual copying is error-prone and time-consuming

## ✅ Proposed Solution: Git-Based Workflow

### **Option 1: Proper Git Repository (Recommended)**

**Setup once:**
```bash
cd "/Users/theodorselimovic/Library/CloudStorage/OneDrive-Personal/Sciences Po/Master Thesis/Text analysis code/Text-as-data-master-thesis-repo"

# Initialize git (if not already)
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Large data files
*.parquet
*.rds
*.bin
*.npy
*.pkl

# Model files
cc.sv.300.bin
cc.sv.300.vec

# Output directories (keep structure, ignore contents)
data/raw/*
data/processed/*
data/vectors/*
results/figures/*
results/tables/*

# But track .gitkeep files
!**/.gitkeep

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# macOS
.DS_Store

# R
.Rhistory
.RData

# Logs
*.log
EOF

# Commit initial structure
git add .
git commit -m "Initial project structure"
```

**Benefits:**
- ✅ Full version control
- ✅ I can see entire project history
- ✅ Easy to revert mistakes
- ✅ Collaborate efficiently
- ✅ Track what changed and why

---

### **Option 2: Organized Directory Structure (Simpler)**

If you don't want Git right now, at least organize the directory properly:

```
Text-as-data-master-thesis-repo/
│
├── README.md                          # Project overview
├── METHODOLOGY_SUMMARY.md             # Your existing methodology doc
├── COMPLETE_PIPELINE_GUIDE.md         # Complete workflow guide
│
├── scripts/                           # All executable scripts
│   ├── 01_ocr/
│   │   ├── run_ocr.py
│   │   └── ocr_swedish_pdfs_improved.py
│   │
│   ├── 02_preprocessing/
│   │   ├── readingtexts.py            # Convert from notebook
│   │   └── readingtexts.ipynb         # Original (deprecated)
│   │
│   ├── 03_expansion/
│   │   ├── vectoranalysis.py          # NEW: Main script
│   │   └── vectoranalysis.ipynb       # Original (deprecated)
│   │
│   ├── 04_filtering/
│   │   ├── sentencefiltering.py       # Convert from notebook
│   │   └── sentencefiltering.ipynb    # Original (deprecated)
│   │
│   └── 05_analysis/
│       ├── cooccurrence_analysis.py
│       ├── data_diagnostic.py
│       └── correspondence_analysis.py # Future
│
├── notebooks/                         # Interactive exploration only
│   ├── cooccurrence_analysis_notebook.ipynb
│   └── exploratory/
│       └── (ad-hoc analyses here)
│
├── docs/                              # Documentation
│   ├── README_OCR.md
│   ├── README_COOCCURRENCE.md
│   ├── SEED_TERMS_REFERENCE.md
│   ├── SEED_TERMS_UPDATE_V2.md
│   └── figures/
│       └── (methodology diagrams)
│
├── data/                              # Data files (gitignored except .gitkeep)
│   ├── raw/
│   │   ├── pdfs/                      # Original RSA PDFs
│   │   └── failed_files.txt
│   │
│   ├── processed/
│   │   ├── readtext_success.rds       # From R readtext
│   │   └── sentences_lemmatized.parquet
│   │
│   ├── expanded_terms/
│   │   └── expanded_terms_lemmatized_complete.csv
│   │
│   └── vectors/
│       ├── sentence_vectors_with_metadata.parquet
│       ├── sentence_vectors.npy
│       └── sentence_vectors_metadata.csv
│
├── results/                           # Analysis outputs
│   ├── cooccurrence/
│   │   ├── effect_cooccurrence.csv
│   │   ├── effect_actor_associations.csv
│   │   └── temporal_frequencies.csv
│   │
│   └── figures/
│       ├── effect_frequencies.png
│       └── effect_actor_heatmap.png
│
├── models/                            # External models (gitignored)
│   └── cc.sv.300.bin                  # FastText Swedish
│
└── archive/                           # Old/deprecated files
    └── (move old versions here)
```

---

## 🔄 New Workflow

### **During Claude Session**

**What I'll do:**
1. Read existing code from `/mnt/project/` (read-only snapshot from session start)
2. Create new/updated files in `/mnt/user-data/outputs/`
3. Suggest where files should go in your structure
4. Provide commit message suggestions

**What you'll do:**
1. Download files from outputs/
2. Place in appropriate locations in your local repo
3. Test the code
4. Commit if it works: `git add . && git commit -m "Add cooccurrence analysis"`
5. Push to remote (optional): `git push origin main`

**Important**: I cannot see git history, modify files in place, or write directly to your filesystem. The `/mnt/project/` directory is a read-only snapshot from session start.

**Example session:**
```bash
# I create a new file
Me: "I've created scripts/05_analysis/cooccurrence_analysis.py"

# You review
You: git diff scripts/05_analysis/cooccurrence_analysis.py

# You commit
You: git add scripts/05_analysis/cooccurrence_analysis.py
You: git commit -m "Add chi-square co-occurrence analysis script"
```

### **Benefits of This Workflow**

1. **I can see everything**: No more blind spots about what exists
2. **No manual copying**: Everything stays in one place
3. **Version control**: Track all changes
4. **Collaboration**: Easy to review what I changed
5. **Documentation**: Commit messages explain why changes were made

---

## 🚀 Migration Plan

### **Step 1: Organize Current Files (30 minutes)**

```bash
# Navigate to your repo
cd "/Users/theodorselimovic/Library/CloudStorage/OneDrive-Personal/Sciences Po/Master Thesis/Text analysis code/Text-as-data-master-thesis-repo"

# Create directory structure
mkdir -p scripts/{01_ocr,02_preprocessing,03_expansion,04_filtering,05_analysis}
mkdir -p notebooks/exploratory
mkdir -p docs/figures
mkdir -p data/{raw/pdfs,processed,expanded_terms,vectors}
mkdir -p results/{cooccurrence,figures}
mkdir -p models
mkdir -p archive

# Keep structure in git even when directories are empty
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/expanded_terms/.gitkeep
touch data/vectors/.gitkeep
touch results/cooccurrence/.gitkeep
touch results/figures/.gitkeep
touch models/.gitkeep

# Move files to proper locations
mv ocr_swedish_pdfs_improved.py scripts/01_ocr/
mv run_ocr.py scripts/01_ocr/
mv readingtexts.ipynb scripts/02_preprocessing/
mv vectoranalysis.ipynb scripts/03_expansion/
mv sentencefiltering.ipynb scripts/04_filtering/

# Move documentation
mv README_OCR.ipynb docs/README_OCR.md  # Convert to markdown
mv METHODOLOGY_SUMMARY.md docs/
mv failed_files.txt data/raw/

# Move sample PDF
mv RSA_Arvidsjaur_2019_Maskad.pdf data/raw/pdfs/
```

### **Step 2: Add New Scripts**

Download from outputs and place in proper locations:
```bash
# Analysis scripts
cp /path/to/downloads/vectoranalysis.py scripts/03_expansion/
cp /path/to/downloads/cooccurrence_analysis.py scripts/05_analysis/
cp /path/to/downloads/data_diagnostic.py scripts/05_analysis/

# Documentation
cp /path/to/downloads/COMPLETE_PIPELINE_GUIDE.md docs/
cp /path/to/downloads/README_COOCCURRENCE.md docs/
cp /path/to/downloads/SEED_TERMS_REFERENCE.md docs/
cp /path/to/downloads/SEED_TERMS_UPDATE_V2.md docs/

# Notebooks
cp /path/to/downloads/cooccurrence_analysis_notebook.ipynb notebooks/
```

### **Step 3: Create .gitignore**

```bash
cat > .gitignore << 'EOF'
# Large data files
*.parquet
*.rds
*.bin
*.npy
*.pkl
*.h5

# Data directories (keep structure)
data/raw/pdfs/*.pdf
data/processed/*
data/expanded_terms/*.csv
data/vectors/*
!**/.gitkeep

# Results (regenerable)
results/cooccurrence/*.csv
results/figures/*.png
results/figures/*.pdf

# Models (too large for git)
models/*.bin
models/*.vec

# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
.pytest_cache/
*.egg-info/
dist/
build/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# macOS
.DS_Store
.AppleDouble

# R
.Rhistory
.RData
.Rproj.user

# Logs
*.log
logs/

# Temporary files
*.tmp
*~
EOF
```

### **Step 4: Initialize Git**

```bash
# Initialize
git init

# Add everything
git add .

# First commit
git commit -m "Initial project structure with organized scripts and documentation"

# Check status
git status
git log --oneline
```

### **Step 5: Create Project README**

I'll create a comprehensive README.md for the root directory (see next file).

---

## 📝 Daily Workflow (After Setup)

### **Starting a Claude Session**

```bash
# Share project structure with me
Me: "Here's my current project structure"
You: tree -L 3 -I '__pycache__|.git|*.pyc'

# Or just show what's relevant
Me: "Show me the scripts directory"
You: ls -lh scripts/*/
```

### **During Work**

```bash
# I'll tell you: "I've updated scripts/05_analysis/cooccurrence_analysis.py"

# You check the changes
git diff scripts/05_analysis/cooccurrence_analysis.py

# If you approve
git add scripts/05_analysis/cooccurrence_analysis.py
git commit -m "Fix: Handle zero-frequency categories in chi-square tests"
```

### **Ending a Session**

```bash
# Review all changes
git status
git log --oneline -5

# Push to remote (if using GitHub/GitLab)
git push origin main
```

---

## 🎯 Immediate Action Items

### **For You (30 minutes)**

1. [ ] Reorganize files into proper structure
2. [ ] Create .gitignore
3. [ ] Initialize git repository
4. [ ] Create initial commit
5. [ ] Download latest scripts from /mnt/user-data/outputs/ and place in proper folders

### **For Next Claude Session**

1. Share the new project structure (just paste `tree` output)
2. I'll work directly in `/mnt/project/` going forward
3. You review changes and commit

---

## 💡 Pro Tips

### **Quick Commands**

```bash
# See what changed
git status

# See changes in a file
git diff path/to/file.py

# See commit history
git log --oneline --graph

# Undo uncommitted changes
git checkout -- path/to/file.py

# Create a new branch for experiments
git checkout -b experiment-new-analysis

# Go back to main branch
git checkout main
```

### **Commit Message Conventions**

Use conventional commits for clarity:
```
feat: Add correspondence analysis script
fix: Handle division by zero in Cramér's V
docs: Update seed terms reference
refactor: Convert vectoranalysis to .py script
test: Add unit tests for term expansion
chore: Reorganize project structure
```

---

## 🔒 What NOT to Commit

Never commit:
- ❌ Large data files (>100MB)
- ❌ Model files (FastText .bin)
- ❌ Processed datasets (.parquet, .npy)
- ❌ Generated results (can be recreated)
- ❌ Personal API keys or credentials
- ❌ Temporary files

Always commit:
- ✅ Python scripts (.py)
- ✅ Notebooks (small ones, <5MB)
- ✅ Documentation (.md)
- ✅ Configuration files
- ✅ Small reference data (<1MB)
- ✅ Tests

---

## ⚠️ Understanding Claude's File Access

### What `/mnt/project/` Actually Is

`/mnt/project/` is a **read-only snapshot** of your project uploaded at session start.

**I CAN:**
- ✅ Read all files in `/mnt/project/`
- ✅ Reference your existing code structure
- ✅ Create new files in `/mnt/user-data/outputs/`

**I CANNOT:**
- ❌ See git history or commits
- ❌ See live changes you make during session
- ❌ Write files directly to your local filesystem
- ❌ Access GitHub/GitLab remote repositories
- ❌ See branches or git status

### Actual Workflow

```
Your Local Repo                          Claude
     │                                      │
     ├─ scripts/                            ├─ /mnt/project/ (read-only)
     ├─ .git/ (history)                     │  └─ snapshot at session start
     │                                      │
     └─ [you work here]                     └─ /mnt/user-data/outputs/
                                                └─ [I create files here]
                                                └─ [you download these]
```

### Sharing Context With Me

At session start, share:
```bash
# Current structure
tree -L 2 -I '__pycache__|.git'

# Recent changes (optional)
git log --oneline -5
git status

# What you're working on
"I'm adding correspondence analysis..."
```

This helps me understand what's changed since last session!

---

## 🎓 Learning Resources

**Git Basics:**
- [Git Handbook (GitHub)](https://guides.github.com/introduction/git-handbook/)
- [Learn Git Branching (Interactive)](https://learngitbranching.js.org/)

**Project Organization:**
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Good Enough Practices for Scientific Computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)

---

## 🆘 Troubleshooting

### "I accidentally committed a large file"

```bash
# Remove from git but keep file locally
git rm --cached data/vectors/large_file.parquet
echo "data/vectors/*.parquet" >> .gitignore
git add .gitignore
git commit -m "Remove large file from git tracking"
```

### "I want to see what changed since last week"

```bash
git log --since="1 week ago" --oneline
git diff HEAD@{1.week.ago} HEAD
```

### "I want to revert a file to previous version"

```bash
# See file history
git log --oneline -- path/to/file.py

# Revert to specific commit
git checkout COMMIT_HASH -- path/to/file.py
```

---

**This new workflow will make our collaboration much more efficient!** 🚀
