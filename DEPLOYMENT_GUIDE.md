# Complete Deployment Guide - Vietnamese AMR Parser with MTUP

**Hướng dẫn đầy đủ: Local Setup → Git Push → Server Deployment → HuggingFace Login → Training → Evaluation**

---

## 📋 **TABLE OF CONTENTS**

1. [Local Preparation](#step-1-local-preparation)
2. [Git Push](#step-2-git-push)
3. [Server Setup](#step-3-server-setup)
4. [HuggingFace Login](#step-4-huggingface-login-critical)
5. [Data Verification](#step-5-data-verification)
6. [Training](#step-6-training-with-mtup)
7. [Monitoring](#step-7-monitoring-training)
8. [Evaluation](#step-8-evaluation)
9. [Troubleshooting](#troubleshooting)

---

## 🏠 **STEP 1: LOCAL PREPARATION**

### **1.1. Verify Local Setup**

```bash
cd ~/ViSemPar_new1

# Check all MTUP files exist
ls -l config/prompt_templates.py config/config_mtup.py src/preprocessor_mtup.py train_mtup.py

# Run local tests
python3 test_mtup_simple.py
python3 quick_test_mtup.py
```

**Expected:**
- ✅ All test scripts pass
- ✅ No errors in preprocessing
- ✅ MTUP format looks correct

---

## 📤 **STEP 2: GIT PUSH**

### **2.1. Check Git Status**

```bash
git status
git log --oneline -5
```

### **2.2. Ensure Latest Code**

```bash
# If you made changes
git add .
git commit -m "Ready for server deployment"
git push origin main
```

### **2.3. Verify Push**

```bash
git log --oneline -1
# Should show your latest commit
```

---

## 🖥️ **STEP 3: SERVER SETUP**

### **3.1. SSH to Server**

```bash
ssh your_username@server_address
```

**If using SSH key:**
```bash
ssh -i ~/.ssh/your_key.pem your_username@server_address
```

---

### **3.2. Clone or Pull Repository**

**First Time (Clone):**
```bash
git clone https://github.com/your-username/ViSemPar_new1.git
cd ViSemPar_new1
```

**Already Cloned (Pull):**
```bash
cd ViSemPar_new1
git pull origin main
```

**Verify:**
```bash
git log --oneline -3
# Should match your local commits
```

---

### **3.3. Run Setup Script**

```bash
bash setup_server.sh
```

**Setup script will:**
1. ✅ Check Python 3.8+
2. ✅ Create directories (data, outputs, logs, checkpoints)
3. ✅ Install dependencies from requirements.txt
4. ✅ Ask about HuggingFace setup

**When asked about HuggingFace:**
```
Choose an option:
  1. CLI Login (RECOMMENDED - one-time setup)
  2. Environment Variable (.env file)
  3. Skip for now

→ Choose 1: CLI Login
```

---

## 🔑 **STEP 4: HUGGINGFACE LOGIN (CRITICAL)**

**Why needed?**
- ✅ Download pretrained models (Qwen, Gemma, Phi)
- ✅ Save model checkpoints
- ✅ Avoid rate limits

---

### **4.1. Get HuggingFace Token**

**On your browser (can do on local machine):**

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Token name: `vlsp-amr-server`
4. Permission: **"Write"** (IMPORTANT!)
5. Click **"Generate a new token"**
6. **COPY TOKEN** immediately (only shown once!)
   - Format: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

---

### **4.2. Login on Server**

**Method 1: CLI Login (RECOMMENDED) ⭐**

```bash
# This is the easiest and most secure method
huggingface-cli login
```

**When prompted:**
```
Token (input will not be visible): [paste your token here]
Add token as git credential? (Y/n) n
```

**Verify:**
```bash
huggingface-cli whoami
# Should show: your-username
```

**Token saved at:** `~/.cache/huggingface/token`

✅ **DONE!** Now training will auto-use this token!

---

**Method 2: Environment Variable (Alternative)**

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx

# Verify
echo $HF_TOKEN

# Make permanent (optional)
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

---

**Method 3: .env File (Alternative)**

```bash
cd ~/ViSemPar_new1

# Create .env file
echo 'HF_TOKEN=hf_xxxxxxxxxxxxx' > .env

# Verify
cat .env
```

**Note:** .env file is auto git-ignored (safe)

---

**Method 4: Python Script (Alternative)**

```bash
# Interactive login
python3 hf_login.py

# Or with token directly
python3 hf_login.py --token hf_xxxxxxxxxxxxx

# Check status
python3 hf_login.py --check
```

---

### **4.3. Verify Login**

**Quick check:**
```bash
huggingface-cli whoami
```

**Detailed check:**
```bash
python3 hf_login.py --check
```

**In training script:**
```bash
python3 -c "from hf_auth import ensure_hf_login; ensure_hf_login()"
```

**Expected output:**
```
✅ Already logged in to HuggingFace
   User: your-username
```

---

### **4.4. Login Method Comparison**

| Method | Ease | Security | Persistent | Auto-detected | Recommended |
|--------|------|----------|------------|---------------|-------------|
| **CLI Login** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | **YES** ⭐ |
| **Env Var** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Session | ✅ | Testing |
| **.env File** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ | Alternative |
| **Python Script** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ | Advanced |

**Training script auto-detects login from ALL methods!**

---

## 📊 **STEP 5: DATA VERIFICATION**

### **5.1. Check Data Files**

```bash
ls -lh data/
```

**Should have:**
```
train_amr_1.txt          (~1.2 MB, ~1200 examples)
train_amr_2.txt          (~1.3 MB, ~1300 examples)
public_test.txt          (optional)
public_test_ground_truth.txt (optional)
```

---

### **5.2. Count Examples**

```bash
# Count training examples
grep -c "^#::snt" data/train_amr_1.txt
grep -c "^#::snt" data/train_amr_2.txt

# Total should be ~2500 examples
```

**Sample first example:**
```bash
head -10 data/train_amr_1.txt
```

---

### **5.3. Test Preprocessing**

```bash
python3 test_mtup_simple.py
```

**Expected output:**
```
================================================================================
MTUP PIPELINE ANALYSIS & VERIFICATION
================================================================================

✓ Loaded 1262 examples from train_amr_1.txt

Testing preprocessing...
✓ Task 1 (no vars): (bi_kịch :domain(chỗ :mod(đó)))
✓ Task 2 (with vars): (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))

Vietnamese character handling:
✓ đ, ô, ê, etc. properly handled

✅ ALL CHECKS PASSED!
```

---

### **5.4. Test MTUP Data Preparation**

```bash
python3 quick_test_mtup.py
```

**Expected output:**
```
MTUP DATA PREPARATION TEST
✓ Loaded 1262 examples
✓ Processed 10/10 examples

VALIDATION CHECKS
✓ Has input section
✓ Has Task 1
✓ Has Task 2
✓ Has instructions

✅ ALL CHECKS PASSED - MTUP PREPROCESSING READY!
```

---

### **5.5. Test SMATCH Evaluation**

```bash
python3 test_smatch.py
```

**Expected output:**
```
SMATCH EVALUATION TEST
Example 1:
  Precision: 1.0000
  Recall:    1.0000
  F1:        1.0000

AVERAGE SCORES
Precision: 1.0000
Recall:    1.0000
F1:        1.0000

✅ SMATCH working correctly!
```

---

## 🚀 **STEP 6: TRAINING WITH MTUP**

### **6.1. Quick Test (ALWAYS RUN FIRST)**

**Purpose:** Verify entire pipeline works

```bash
python3 train_mtup.py --use-case quick_test --show-sample
```

**What it does:**
- Processes 100 samples
- Trains for 1 epoch
- Shows MTUP format example
- Verifies HF login
- Tests model loading
- ~10 minutes

**Expected output:**
```
╔══════════════════════════════════════════════════════════════╗
║     VIETNAMESE AMR PARSER - MTUP TRAINING                   ║
╚══════════════════════════════════════════════════════════════╝

======================================================================
CHECKING HUGGINGFACE LOGIN
======================================================================
✅ Already logged in to HuggingFace
   User: your-username
======================================================================

ENVIRONMENT CHECK
✓ Data directory: /path/to/data
✓ Output directory: /path/to/outputs
...

STEP 1: LOADING DATA WITH MTUP PREPROCESSING
✓ Loaded 100 samples
...

SAMPLE MTUP EXAMPLE
### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)
...

STEP 2: LOADING MODEL AND TOKENIZER
Model: Qwen/Qwen2.5-3B-Instruct
...

STEP 3: TRAINING WITH MTUP STRATEGY
🚀 Starting training...
...

✅ TRAINING COMPLETED
```

**If quick test fails → FIX before full training!**

---

### **6.2. Fast Iteration (Optional - For Tuning)**

**Purpose:** Test hyperparameters, quick experiments

```bash
python3 train_mtup.py --use-case fast_iteration
```

**What it does:**
- Processes 500 samples
- Trains for 3 epochs
- ~30-40 minutes

**Use when:**
- Testing different learning rates
- Trying different templates
- Quick accuracy check

---

### **6.3. Full Training (Production) ⭐**

**Purpose:** Production model with all data

**IMPORTANT: Use tmux to avoid disconnection!**

```bash
# Create tmux session
tmux new -s amr-training

# Run training
python3 train_mtup.py --use-case full_training

# Detach from tmux: Press Ctrl+B, then D
# Training continues in background!
```

**What it does:**
- Processes ~2500 samples (all data)
- Trains for 10 epochs (OPTIMIZED)
- Validation every 100 steps
- Saves checkpoints every 200 steps
- ~2-3 hours (3B model)

**Training configuration:**
```
Model:           Qwen/Qwen2.5-3B-Instruct
Learning rate:   2e-4 (optimized)
Batch size:      4
Grad accumulation: 4 (effective batch = 16)
Epochs:          10 (optimized for MTUP)
Val split:       10% (~250 examples)
```

---

### **6.4. Custom Training**

**Optimized 3B (Recommended):**
```bash
python3 train_mtup.py \
  --model qwen2.5-3b \
  --epochs 10 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --val-split 0.1
```

**Best Accuracy 7B (Slower):**
```bash
python3 train_mtup.py \
  --model qwen2.5-7b \
  --epochs 15 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 1e-4 \
  --val-split 0.1
```

**Quick Prototyping 1.5B (Fastest):**
```bash
python3 train_mtup.py \
  --model qwen2.5-1.5b \
  --epochs 8 \
  --batch-size 8 \
  --lr 3e-4
```

---

## 📈 **STEP 7: MONITORING TRAINING**

### **7.1. Reattach to Tmux**

```bash
# List tmux sessions
tmux ls

# Attach to training session
tmux attach -t amr-training

# Detach again: Ctrl+B, D
```

---

### **7.2. Monitor GPU Usage**

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or
nvidia-smi -l 1
```

**What to check:**
- GPU Utilization: Should be 80-95%
- GPU Memory: 12-14GB for 3B, 18-22GB for 7B
- GPU Temperature: < 85°C

---

### **7.3. Watch Training Logs**

```bash
# Follow training log
tail -f outputs/logs/mtup_*/training.log

# Or grep for important info
tail -f outputs/logs/mtup_*/training.log | grep -E "(loss|epoch|validation)"
```

---

### **7.4. TensorBoard (Optional)**

**In another terminal:**
```bash
# SSH with port forwarding
ssh -L 6006:localhost:6006 user@server

# Start TensorBoard
tensorboard --logdir outputs/logs

# Open browser: http://localhost:6006
```

---

### **7.5. Monitor Disk Space**

```bash
# Check disk usage
df -h

# Check output directory size
du -sh outputs/

# Watch it grow
watch -n 5 'du -sh outputs/'
```

---

## 🎯 **STEP 8: EVALUATION**

### **8.1. After Training Completes**

**Check final model:**
```bash
ls -lh outputs/checkpoints_mtup/mtup_full_training_final/
```

**Should have:**
```
adapter_config.json
adapter_model.safetensors
tokenizer files
training_args.bin
```

---

### **8.2. Evaluate on Test Data**

```bash
python3 evaluate_test_data.py
```

**Expected output:**
```
REAL DATA SMATCH EVALUATION

✓ Loaded 250 examples from test data
✓ SMATCH available

Evaluating...
Example  1: P=0.8500 R=0.8200 F1=0.8347
Example  2: P=0.7800 R=0.7600 F1=0.7698
...

AVERAGE SMATCH SCORES
Precision: 0.7234
Recall:    0.7156
F1:        0.7195

✅ Evaluation complete!
```

**Target scores:**
- **Excellent**: F1 > 75%
- **Good**: F1 > 70%
- **Acceptable**: F1 > 65%
- **Need improvement**: F1 < 65%

---

### **8.3. Analyze Results**

**Check Task 1 vs Task 2 accuracy:**
- Task 1 (structure): Should be higher (~80-85%)
- Task 2 (variables): Slightly lower (~75-80%)
- This is expected with MTUP approach

**If F1 < 65%:**
1. Increase epochs: `--epochs 15`
2. Try 7B model: `--model qwen2.5-7b`
3. Try different template: Edit config → `template_name: "v5_cot"`
4. Check for overfitting (val loss vs train loss)

---

## 🔧 **TROUBLESHOOTING**

### **Common Issues & Solutions**

---

#### **Issue 1: HuggingFace Login Failed**

**Error:**
```
❌ LOGIN FAILED
Error: 401 Unauthorized
```

**Solution:**
```bash
# Token invalid or expired
huggingface-cli logout
huggingface-cli login
# Paste NEW token with "Write" permission
```

---

#### **Issue 2: Out of Memory (OOM)**

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Reduce batch size
python3 train_mtup.py --batch-size 2 --grad-accum 8

# Or use smaller model
python3 train_mtup.py --model qwen2.5-1.5b

# Or reduce sequence length
python3 train_mtup.py --max-length 1536
```

---

#### **Issue 3: Data Files Not Found**

**Error:**
```
❌ Missing training files: train_amr_1.txt
```

**Solution:**
```bash
# Check data directory
ls -la data/

# Copy from local (from your local machine)
scp -r data/*.txt user@server:~/ViSemPar_new1/data/

# Or download from backup
```

---

#### **Issue 4: Training Too Slow**

**Solutions:**

1. **Use smaller model:**
```bash
python3 train_mtup.py --model qwen2.5-3b  # Instead of 7b
```

2. **Increase batch size (if GPU allows):**
```bash
python3 train_mtup.py --batch-size 8
```

3. **Reduce gradient accumulation:**
```bash
python3 train_mtup.py --grad-accum 2
```

---

#### **Issue 5: Validation Loss Increasing**

**Symptoms:**
- Training loss decreasing
- Validation loss increasing
- Model is overfitting

**Solutions:**

1. **Reduce learning rate:**
```bash
python3 train_mtup.py --lr 1e-4
```

2. **Reduce epochs:**
```bash
python3 train_mtup.py --epochs 8
```

3. **Increase validation split:**
```bash
python3 train_mtup.py --val-split 0.15
```

---

#### **Issue 6: SMATCH Score Low (< 65%)**

**Analysis steps:**

1. **Check preprocessing:**
```bash
python3 test_mtup_simple.py
# Verify Task 1 has no variables
# Verify Task 2 has variables
```

2. **Check Task 1 vs Task 2 separately**

3. **Try solutions:**
```bash
# Increase epochs
python3 train_mtup.py --epochs 15

# Use larger model
python3 train_mtup.py --model qwen2.5-7b

# Try different template
# Edit config/config_mtup.py: template_name = "v5_cot"
```

---

#### **Issue 7: Import Errors**

**Error:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or specific package
pip install transformers accelerate peft bitsandbytes
```

---

#### **Issue 8: Tmux Session Lost**

**Find session:**
```bash
tmux ls
```

**Attach to session:**
```bash
tmux attach -t amr-training
```

**If session killed:**
- Training stopped
- Need to restart training
- Use checkpoints to resume (if available)

---

## 📝 **COMPLETE WORKFLOW CHECKLIST**

### **Local (One-time):**
```
☐ Code ready and tested locally
☐ Push to git
```

### **Server Setup (One-time):**
```
☐ SSH to server
☐ Clone repository
☐ Run setup_server.sh
☐ HuggingFace login (CLI method)
☐ Verify: huggingface-cli whoami
```

### **Data Verification:**
```
☐ Data files in data/ directory
☐ python3 test_mtup_simple.py → PASS
☐ python3 quick_test_mtup.py → PASS
☐ python3 test_smatch.py → PASS
```

### **Training:**
```
☐ Quick test: python3 train_mtup.py --use-case quick_test → PASS
☐ Create tmux session
☐ Full training: python3 train_mtup.py --use-case full_training
☐ Detach from tmux
```

### **Monitoring:**
```
☐ GPU usage ~80-95%
☐ Training loss decreasing
☐ Validation loss stable/decreasing
☐ No OOM errors
```

### **Evaluation:**
```
☐ Training completed
☐ Model saved to outputs/checkpoints_mtup/
☐ python3 evaluate_test_data.py
☐ SMATCH F1 > 65% (acceptable)
☐ SMATCH F1 > 70% (good)
☐ SMATCH F1 > 75% (excellent)
```

---

## 🎓 **BEST PRACTICES**

### **1. Always Start with Quick Test**
```bash
python3 train_mtup.py --use-case quick_test --show-sample
```
- Catches errors early
- Verifies entire pipeline
- Shows MTUP format
- Only takes ~10 minutes

### **2. Use Tmux for Long Training**
```bash
tmux new -s amr-training
python3 train_mtup.py --use-case full_training
# Ctrl+B, D to detach
```
- Training continues if SSH disconnects
- Can reattach anytime
- Multiple terminals possible

### **3. Monitor During Training**
```bash
# Terminal 1: Training (in tmux)
# Terminal 2: GPU monitoring
watch -n 1 nvidia-smi

# Terminal 3: Logs
tail -f outputs/logs/mtup_*/training.log
```

### **4. Save Checkpoints Regularly**
- Default: Every 200 steps
- Keeps last 3 checkpoints
- Can resume if training stops

### **5. Use CLI Login for HuggingFace**
- One-time setup
- Secure token storage
- Auto-detected by all scripts
- No configuration needed

### **6. Check Validation Loss**
- Should track with training loss
- If diverging → overfitting
- Reduce epochs or learning rate

### **7. Start with 3B Model**
- MTUP makes 3B effective
- 2-3x faster than 7B
- Good balance of speed and accuracy
- Upgrade to 7B only if needed

---

## 🎯 **QUICK REFERENCE**

### **Essential Commands:**

```bash
# SSH
ssh user@server

# Pull latest code
cd ~/ViSemPar_new1 && git pull

# HuggingFace login (one-time)
huggingface-cli login

# Verify login
huggingface-cli whoami

# Quick test (always first)
python3 train_mtup.py --use-case quick_test

# Full training (in tmux)
tmux new -s amr-training
python3 train_mtup.py --use-case full_training

# Detach tmux
# Ctrl+B, then D

# Reattach tmux
tmux attach -t amr-training

# Monitor GPU
watch -n 1 nvidia-smi

# Check logs
tail -f outputs/logs/mtup_*/training.log

# Evaluate
python3 evaluate_test_data.py
```

---

## 🚀 **READY TO DEPLOY!**

**One-time setup:**
1. `huggingface-cli login` (paste token)
2. `python3 train_mtup.py --use-case quick_test`

**Every training run:**
1. `git pull`
2. `tmux new -s amr-training`
3. `python3 train_mtup.py --use-case full_training`

**That's it!** Training runs automatically with all optimizations! 🎉

---

**See also:**
- [MTUP_TRAINING_GUIDE.md](MTUP_TRAINING_GUIDE.md) - Detailed training guide
- [OPTIMIZATION_APPLIED.md](OPTIMIZATION_APPLIED.md) - What's been optimized
- [QUICK_COMMANDS.md](QUICK_COMMANDS.md) - Copy-paste commands
