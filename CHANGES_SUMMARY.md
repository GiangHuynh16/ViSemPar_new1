# CHANGES_SUMMARY.md - What's New & How to Use

## 🎉 Summary of Updates

I've updated your Vietnamese AMR Parser project with several important improvements to address the issues you mentioned and make training easier on your university server.

---

## ✅ What Was Fixed

### 1. **Tensor Dimension Mismatch Error** ✅
**Previous Error:**
```
ValueError: expected sequence of length 146 at dim 1 (got 113)
```

**Fix Applied:**
- Changed from `DataCollatorForLanguageModeling` to `DataCollatorForSeq2Seq`
- `DataCollatorForSeq2Seq` properly handles variable-length sequences with dynamic padding
- Added `label_pad_token_id=-100` to ignore padding in loss calculation
- Added `padding=True` and `max_length` parameters

**Location:** `train.py` lines 315-321

### 2. **Automatic Directory Creation Removed** ✅
**Previous Issue:**
- `config/config.py` automatically created directories
- Required sudo permissions on university server
- Training would fail with permission errors

**Fix Applied:**
- Removed automatic directory creation from `config/config.py`
- Added manual directory check before training starts
- Created `setup_directories.sh` helper script
- Clear error messages guide you to create missing directories

**Location:** `config/config.py` lines 9-16

### 3. **Single Unified Training Script** ✅
**Previous:**
- `train_standard.py` - Simple training
- `main.py` - Full pipeline
- Two separate files to maintain

**Fix Applied:**
- New single `train.py` that combines both
- All features in one file
- Easier to maintain and use
- Command-line arguments for flexibility

### 4. **Comprehensive tmux Guide** ✅
**Added:**
- Complete `TRAINING_GUIDE.md` with tmux instructions
- `QUICK_REFERENCE.md` for common commands
- Step-by-step workflows
- Troubleshooting section

---

## 📁 New Files Created

1. **`train.py`** ⭐ (Main file - USE THIS)
   - Unified training script
   - Fixes tensor dimension mismatch
   - Better error handling
   - No automatic file creation

2. **`TRAINING_GUIDE.md`**
   - Complete training guide
   - tmux commands and workflows
   - Step-by-step instructions
   - Troubleshooting tips

3. **`QUICK_REFERENCE.md`**
   - Quick command reference
   - tmux cheat sheet
   - Common workflows
   - Emergency commands

4. **`setup_directories.sh`**
   - Helper script to create directories
   - Run before training
   - Avoids permission issues

5. **`README.md`** (Updated)
   - New quick start guide
   - Updated documentation
   - File comparison table

---

## 🚀 How to Use (Simple Version)

### Step 1: Upload to Server
```bash
# Upload the updated ViSemPar folder to your server
scp -r ViSemPar_Updated your-username@server:/mnt/nghiepth/giangha/
```

### Step 2: Setup Directories
```bash
cd /mnt/nghiepth/giangha/ViSemPar_Updated
./setup_directories.sh
```

### Step 3: Add Your Data
```bash
cp /path/to/train_amr_1.txt data/
cp /path/to/train_amr_2.txt data/
```

### Step 4: Start Training in tmux
```bash
# Start tmux session
tmux new -s amr-training

# Activate environment
conda activate amr-parser

# Start training
python train.py --model 7b

# Detach with: Ctrl+b, then d
```

### Step 5: Check Progress Anytime
```bash
# Reattach to tmux
tmux attach -t amr-training

# Or check logs
tail -f logs/training.log
```

---

## 🔄 Migration from Old System

If you have the old version running:

### Quick Migration
```bash
# 1. Copy data from old project
cp -r old_ViSemPar/data/* ViSemPar_Updated/data/

# 2. Setup new directories
cd ViSemPar_Updated
./setup_directories.sh

# 3. Use new training script
python train.py --model 7b
```

### File Mapping
| Old File | New File | Notes |
|----------|----------|-------|
| `train_standard.py` | `train.py` | Use new unified script |
| `main.py` | `train.py` | Combined into one file |
| Manual directory setup | `setup_directories.sh` | Helper script provided |
| No tmux guide | `TRAINING_GUIDE.md` | Comprehensive guide |

---

## 💡 Key Differences

### Before (Old System)
```bash
# Old way - could fail
python train_standard.py
# Error: Permission denied creating outputs/
# Error: Tensor dimension mismatch
```

### After (New System)
```bash
# New way - safe and clear
./setup_directories.sh  # Create dirs first
python train.py --model 7b  # Fixed tensor issues

# Clear error messages if something is missing:
# ❌ Missing required directories
# 💡 Create them with: mkdir -p data outputs logs
```

---

## 🎯 What to Use When

### For Training
```bash
# Always use the new unified script
python train.py --model 7b

# NOT these old files
# ❌ train_standard.py (old)
# ❌ main.py (old)
```

### For Setup
```bash
# Run once before first training
./setup_directories.sh
```

### For Reference
- `TRAINING_GUIDE.md` - Full guide
- `QUICK_REFERENCE.md` - Quick commands
- `README.md` - Overview

---

## 🔧 Technical Changes Detail

### 1. Data Collator Change
**Before:**
```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

**After:**
```python
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,  # Ignore padding in loss
    padding=True,              # Dynamic padding
    max_length=MAX_SEQ_LENGTH
)
```

### 2. Directory Creation
**Before (config.py):**
```python
# Automatic creation
for dir_path in [DATA_DIR, OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
```

**After (train.py):**
```python
# Check before training, show helpful error
def check_directories():
    missing_dirs = []
    for dir_path in [DATA_DIR, OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR]:
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        logger.error("❌ Missing required directories:")
        logger.error("💡 Create them with: mkdir -p data outputs logs outputs/checkpoints")
        sys.exit(1)
```

### 3. Format Function
**Added explicit structure:**
```python
def format_example(example):
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,  # Let collator handle it
        return_tensors=None  # Return list, not tensor
    )
    
    # IMPORTANT: Labels must be list, not nested
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized
```

---

## 📊 Expected Behavior

### When Directories Don't Exist
**Old System:**
```
PermissionError: [Errno 13] Permission denied: 'outputs'
```

**New System:**
```
❌ Missing required directories:
   /mnt/nghiepth/giangha/ViSemPar/outputs
   /mnt/nghiepth/giangha/ViSemPar/logs

💡 Create them with:
   mkdir -p data outputs logs outputs/checkpoints
```

### When Training Starts
```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         VIETNAMESE AMR PARSER - UNIFIED TRAINING            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
⚡ Device: CUDA
🎮 GPU: NVIDIA RTX 4090
💾 GPU Memory: 24.00 GB
======================================================================
Checking environment...
✓ All required directories exist
✓ All training data files found
```

---

## 🚨 Important Notes

1. **Must Create Directories First**
   - Run `./setup_directories.sh` OR
   - Run `mkdir -p data outputs logs outputs/checkpoints`
   - Do this BEFORE first training

2. **Use tmux for Long Training**
   - SSH disconnections will kill training without tmux
   - Always start tmux first: `tmux new -s amr-training`
   - Detach safely: `Ctrl+b, then d`

3. **Start with 7B Model**
   - More stable and uses less memory
   - Good for validating setup
   - Use 14B later for better performance

4. **Monitor Early**
   - Check logs in first 5 minutes
   - Catch configuration errors early
   - Use `tail -f logs/training.log`

5. **Backup Results**
   - Copy trained model immediately after completion
   - Save to multiple locations
   - Don't rely on server storage alone

---

## 🎓 Recommended Workflow

```bash
# Day 1: Setup
cd /mnt/nghiepth/giangha/ViSemPar_Updated
./setup_directories.sh
cp /path/to/train_amr_*.txt data/
ls data/  # Verify files

# Day 1: First Training (Short test)
tmux new -s test-run
conda activate amr-parser
python train.py --model 7b --epochs 1  # Test run
# Monitor for errors
Ctrl+b, d  # Detach

# Day 2: Full Training
tmux new -s amr-training
conda activate amr-parser
python train.py --model 7b --epochs 15
Ctrl+b, d

# Day 3+: Monitor and Wait
tmux attach -t amr-training  # Check progress
tail -f logs/training.log     # View logs

# When Complete: Backup
cp -r outputs/vlsp_amr_qwen_improved_v2 ~/backup/
```

---

## 📞 If You Need Help

1. **Read the guides:**
   - `TRAINING_GUIDE.md` - Complete walkthrough
   - `QUICK_REFERENCE.md` - Quick commands
   - `README.md` - Overview

2. **Check the error message:**
   - New system gives clear, actionable errors
   - Follow the suggestions in error messages

3. **Review logs:**
   - `tail -f logs/training.log`
   - Shows what's happening in real-time

4. **Common fixes:**
   - Missing directories → `./setup_directories.sh`
   - Out of memory → `--batch-size 1` or `--model 7b`
   - Permission denied → Create directories manually first

---

## ✅ Checklist Before Training

- [ ] Downloaded updated ViSemPar_Updated folder
- [ ] Uploaded to server
- [ ] Ran `./setup_directories.sh` (or created manually)
- [ ] Copied training data to `data/` directory
- [ ] Activated conda environment: `conda activate amr-parser`
- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Started tmux: `tmux new -s amr-training`
- [ ] Ready to run: `python train.py --model 7b`

---

**You're all set! The new system is ready to use. Start with the TRAINING_GUIDE.md for detailed instructions.**

**Good luck with your training! 🚀**
