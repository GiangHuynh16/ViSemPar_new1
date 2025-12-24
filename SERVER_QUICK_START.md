# Server Quick Start Guide - Vietnamese AMR Parser

**Quick reference cho việc chạy training trên server**

---

## 🚀 **SETUP LẦN ĐẦU (One-time setup)**

```bash
# 1. SSH vào server
ssh your_username@server_address

# 2. Clone project
git clone <your-repo-url>
cd ViSemPar_new1

# 3. Run setup script
bash setup_server.sh

# Follow prompts to:
# - Install dependencies
# - Setup Hugging Face token
# - Verify installation
```

---

## 🔑 **SETUP HUGGING FACE TOKEN**

### **Method 1: CLI Login (RECOMMENDED ⭐)**
```bash
pip install --upgrade huggingface_hub
huggingface-cli login
# Paste token: hf_xxxxxxxxxxxxx
```

### **Method 2: Environment Variable**
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

### **Verify:**
```bash
huggingface-cli whoami
```

**Get token from:** https://huggingface.co/settings/tokens (chọn "Write" permission)

---

## 📊 **TRAINING COMMANDS**

### **Quick Test (5 epochs, 500 samples, 1.5B model)**
```bash
python3 train_mtup.py --use-case quick_test
```
- **Time:** ~30-60 minutes
- **Purpose:** Verify everything works

### **Fast Iteration (10 epochs, full data, 3B model) ⭐ RECOMMENDED**
```bash
python3 train_mtup.py --use-case fast_iteration
```
- **Time:** ~3-5 hours
- **Purpose:** Main training

### **Best Accuracy (15 epochs, full data, 7B model)**
```bash
python3 train_mtup.py --use-case best_accuracy
```
- **Time:** ~8-12 hours
- **Purpose:** Maximum performance

### **Custom Training**
```bash
# Custom model
python3 train_mtup.py --model qwen2.5-3b --epochs 12

# Custom batch size
python3 train_mtup.py --model qwen2.5-3b --batch-size 8

# Skip inference/evaluation
python3 train_mtup.py --skip-inference --skip-evaluation
```

---

## 🖥️ **TMUX - PERSISTENT SESSIONS**

**Why tmux?** SSH disconnects won't kill your training!

### **Start Training in tmux:**
```bash
# 1. Create new session
tmux new -s amr-training

# 2. Start training
python3 train_mtup.py --use-case fast_iteration

# 3. Detach (keep training running)
# Press: Ctrl+B, then D
```

### **Common tmux Commands:**
```bash
# List sessions
tmux ls

# Reattach to session
tmux attach -t amr-training

# Kill session (after training done)
tmux kill-session -t amr-training

# Scroll in tmux
# Press: Ctrl+B, then [
# Use arrow keys, then Q to exit scroll mode
```

---

## 📁 **FILE STRUCTURE ON SERVER**

```
ViSemPar_new1/
├── data/                          # ⚠️ Copy your data here!
│   ├── train_amr_1.txt           # Required
│   ├── train_amr_2.txt           # Required
│   ├── public_test.txt           # Optional
│   └── ...
│
├── outputs/                       # Training outputs
│   ├── checkpoints_mtup/         # Checkpoints saved here
│   └── vlsp_amr_mtup_v1/         # Final model
│
├── logs/
│   └── training_mtup.log         # Training logs
│
├── config/
│   ├── config_mtup.py            # MTUP configuration
│   └── prompt_templates.py       # Prompt templates
│
├── src/
│   └── preprocessor_mtup.py      # MTUP preprocessor
│
├── train_mtup.py                 # ⭐ Main training script (TODO)
├── setup_server.sh               # Setup script
└── requirements.txt
```

---

## 📊 **MONITORING TRAINING**

### **Method 1: Reattach tmux**
```bash
tmux attach -t amr-training
```

### **Method 2: Watch logs**
```bash
tail -f logs/training_mtup.log
```

### **Method 3: Check GPU usage**
```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi

# Or one-time check
nvidia-smi
```

### **Method 4: Check output files**
```bash
# List checkpoints
ls -lh outputs/checkpoints_mtup/

# Check latest checkpoint
ls -lt outputs/checkpoints_mtup/ | head -5
```

---

## ⚠️ **COMMON ISSUES & SOLUTIONS**

### **Issue 1: CUDA Out of Memory**
```bash
# Solution 1: Use smaller batch size
python3 train_mtup.py --batch-size 2

# Solution 2: Use smaller model
python3 train_mtup.py --model qwen2.5-1.5b

# Solution 3: Reduce gradient accumulation
# Edit config/config_mtup.py:
# gradient_accumulation_steps: 4 → 2
```

### **Issue 2: "Module not found"**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or specific package
pip install transformers==4.36.0
```

### **Issue 3: "HF token not found"**
```bash
# Re-login
huggingface-cli logout
huggingface-cli login
```

### **Issue 4: Training stopped/killed**
```bash
# Check if process still running
ps aux | grep train_mtup

# Check last checkpoint
ls -lt outputs/checkpoints_mtup/

# Resume from checkpoint (if script supports it)
# TODO: Add resume functionality
```

---

## 💾 **DATA TRANSFER TO SERVER**

### **Method 1: scp (from local)**
```bash
# Upload single file
scp data/train_amr_1.txt user@server:/path/to/ViSemPar_new1/data/

# Upload entire directory
scp -r data/ user@server:/path/to/ViSemPar_new1/
```

### **Method 2: rsync (better for large files)**
```bash
rsync -avz --progress data/ user@server:/path/to/ViSemPar_new1/data/
```

### **Method 3: Direct on server**
```bash
# If data already on server somewhere
cp /path/to/train_amr_1.txt ~/ViSemPar_new1/data/
```

---

## 📤 **DOWNLOAD RESULTS FROM SERVER**

### **Download trained model:**
```bash
# From local machine
scp -r user@server:/path/to/ViSemPar_new1/outputs/vlsp_amr_mtup_v1 ./

# Or use rsync
rsync -avz --progress user@server:/path/to/ViSemPar_new1/outputs/ ./outputs/
```

### **Download predictions:**
```bash
scp -r user@server:/path/to/ViSemPar_new1/outputs/*_results_*/ ./results/
```

### **Download logs:**
```bash
scp user@server:/path/to/ViSemPar_new1/logs/training_mtup.log ./logs/
```

---

## 🎯 **RECOMMENDED WORKFLOW**

### **Day 1: Setup & Quick Test**
```bash
# 1. Setup
bash setup_server.sh

# 2. Copy data
scp -r data/ user@server:~/ViSemPar_new1/

# 3. Quick test (verify everything works)
python3 train_mtup.py --use-case quick_test

# Check results
cat logs/training_mtup.log
```

### **Day 2: Main Training**
```bash
# 1. Start tmux session
tmux new -s amr-training

# 2. Run main training
python3 train_mtup.py --use-case fast_iteration

# 3. Detach
# Ctrl+B, D

# 4. Check periodically
tmux attach -t amr-training
# or
tail -f logs/training_mtup.log
```

### **Day 3: Collect Results**
```bash
# 1. Check training finished
tmux attach -t amr-training

# 2. Download results
scp -r user@server:~/ViSemPar_new1/outputs/ ./

# 3. Analyze locally
```

---

## 🔧 **CONFIGURATION SHORTCUTS**

### **Edit config on server:**
```bash
nano config/config_mtup.py
```

### **Quick config changes:**
```python
# Change model
MODEL_NAME = MODELS['qwen2.5-1.5b']  # Smaller, faster

# Change epochs
TRAINING_CONFIG['num_train_epochs'] = 20

# Change batch size
TRAINING_CONFIG['per_device_train_batch_size'] = 2

# Change template
MTUP_CONFIG['template_name'] = 'v4_compact'  # More token-efficient
```

---

## 📞 **QUICK HELP**

### **Check if training is running:**
```bash
ps aux | grep train_mtup
nvidia-smi
tmux ls
```

### **Stop training (if needed):**
```bash
# Find process ID
ps aux | grep train_mtup

# Kill process
kill <PID>

# Or kill tmux session
tmux kill-session -t amr-training
```

### **Disk space:**
```bash
df -h
du -sh outputs/
```

### **Memory usage:**
```bash
free -h
```

---

## 📚 **DOCUMENTATION LINKS**

- **MTUP Implementation:** [MTUP_IMPLEMENTATION.md](MTUP_IMPLEMENTATION.md)
- **Hugging Face Setup:** [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md)
- **Main README:** [README.md](README.md)

---

## 🎓 **TRAINING TIMELINE ESTIMATES**

| Use Case | Model | Data | Epochs | Time (V100) | Time (A100) |
|----------|-------|------|--------|-------------|-------------|
| quick_test | 1.5B | 500 | 5 | ~40 min | ~20 min |
| fast_iteration | 3B | Full | 10 | ~4 hours | ~2 hours |
| best_accuracy | 7B | Full | 15 | ~10 hours | ~5 hours |

*Times are estimates and may vary based on hardware and data size*

---

**Happy Training! 🚀**

For issues, check logs first: `tail -100 logs/training_mtup.log`
