# 📚 Documentation Index

Quick navigation for all project documentation.

---

## 🚀 Getting Started

| Document | Description | For |
|----------|-------------|-----|
| [README.md](README.md) | Project overview & quick start | Everyone |
| [EVALUATION_QUICK_REFERENCE.md](EVALUATION_QUICK_REFERENCE.md) | One-page command reference | Quick lookup |

---

## 📊 Evaluation Guides

| Document | Description | When to Use |
|----------|-------------|-------------|
| [HOW_TO_RUN_FULL_EVALUATION.md](HOW_TO_RUN_FULL_EVALUATION.md) | Complete evaluation guide | First time running |
| [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md) | Current status & results | Check progress |
| [EVALUATION_FIX.md](EVALUATION_FIX.md) | Root cause analysis | Understand bugs |
| [EVALUATION_QUICK_REFERENCE.md](EVALUATION_QUICK_REFERENCE.md) | Quick commands | Daily use |

---

## 🔧 Technical Documentation

| Document | Description | For |
|----------|-------------|-----|
| [MTUP_WORKFLOW.md](MTUP_WORKFLOW.md) | Visual workflow explanation | Understanding MTUP |
| [FIX_PREPROCESSING.md](FIX_PREPROCESSING.md) | Preprocessing analysis | Debugging training |
| [QUICK_COMMANDS.md](QUICK_COMMANDS.md) | Server command reference | Server operations |

---

## 📁 Code Files

### Evaluation Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `evaluate_mtup_model.py` | Main evaluation script | Core evaluation |
| `RUN_FULL_EVALUATION.sh` | Run full eval (foreground) | Direct run |
| `RUN_FULL_EVALUATION_TMUX.sh` | Run in tmux (recommended) | Long runs |
| `CHECK_EVALUATION_STATUS.sh` | Monitor progress | Status check |

### Training Scripts

| File | Purpose |
|------|---------|
| `src/train_mtup.py` | Main training script |
| `RUN_FULL_TRAINING.sh` | Full training runner |
| `CHECK_TRAINING_STATUS.sh` | Training monitor |

### Utilities

| File | Purpose |
|------|---------|
| `test_preprocessing.py` | Test preprocessing logic |
| `test_model_output.py` | Debug model generation |
| `fix_model_output.py` | AMR fixing utilities |

---

## 🎯 Quick Workflows

### ➤ Run Full Evaluation

```bash
# Read this:
HOW_TO_RUN_FULL_EVALUATION.md

# Run this:
bash RUN_FULL_EVALUATION_TMUX.sh

# Check with:
bash CHECK_EVALUATION_STATUS.sh
```

### ➤ Check Current Results

```bash
# Read:
EVALUATION_SUMMARY.md

# Or quick ref:
EVALUATION_QUICK_REFERENCE.md
```

### ➤ Understand MTUP

```bash
# Read:
MTUP_WORKFLOW.md
```

### ➤ Debug Issues

```bash
# Read:
EVALUATION_FIX.md

# Or:
FIX_PREPROCESSING.md
```

---

## 📖 Reading Order for New Users

1. **Start here**: [README.md](README.md)
   - Understand project overview
   - See current results

2. **Quick reference**: [EVALUATION_QUICK_REFERENCE.md](EVALUATION_QUICK_REFERENCE.md)
   - Get familiar with commands
   - Know how to run evaluation

3. **Run evaluation**: [HOW_TO_RUN_FULL_EVALUATION.md](HOW_TO_RUN_FULL_EVALUATION.md)
   - Step-by-step guide
   - Full details

4. **Understand results**: [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md)
   - Current status
   - Performance analysis

5. **Deep dive**: [MTUP_WORKFLOW.md](MTUP_WORKFLOW.md)
   - How MTUP works
   - Technical details

---

## 🔍 Find Information By Topic

### Training
- [RUN_FULL_TRAINING.sh](RUN_FULL_TRAINING.sh)
- [src/train_mtup.py](src/train_mtup.py)
- [FIX_PREPROCESSING.md](FIX_PREPROCESSING.md)

### Evaluation
- [evaluate_mtup_model.py](evaluate_mtup_model.py)
- [HOW_TO_RUN_FULL_EVALUATION.md](HOW_TO_RUN_FULL_EVALUATION.md)
- [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md)

### MTUP Format
- [config/prompt_templates.py](config/prompt_templates.py)
- [MTUP_WORKFLOW.md](MTUP_WORKFLOW.md)

### Debugging
- [EVALUATION_FIX.md](EVALUATION_FIX.md)
- [test_preprocessing.py](test_preprocessing.py)
- [test_model_output.py](test_model_output.py)

### Results
- [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md)
- `outputs/evaluation_results_full_*.json`
- `outputs/evaluation_full_*.log`

---

## 📊 Current Status (Quick Glance)

✅ **Training**: Complete
✅ **Quick Test**: F1 = 0.49 (10 samples)
⏳ **Full Test**: Ready to run
📁 **Docs**: Complete

---

## 💡 Tips

1. **Bookmark this page** - Central navigation hub
2. **Start with Quick Reference** - Fastest way to get started
3. **Read summaries first** - Then dive into details
4. **Check workflow diagram** - Understand the big picture

---

## 📞 Quick Commands

```bash
# Run evaluation
bash RUN_FULL_EVALUATION_TMUX.sh

# Check status
bash CHECK_EVALUATION_STATUS.sh

# View results
cat outputs/evaluation_results_full_*.json | python3 -m json.tool

# Read docs
cat EVALUATION_SUMMARY.md
cat MTUP_WORKFLOW.md
```

---

_Last updated: 2025-12-25_
_Total docs: 10+ files_
_All scripts tested and ready_
