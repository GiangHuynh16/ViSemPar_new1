# Quick Commands Reference

**Copy-paste commands cho deployment**

---

## 🏠 **LOCAL - PUSH TO GIT**

```bash
cd ~/ViSemPar_new1

# Add all new files
git add .

# Commit
git commit -m "Add MTUP implementation with Vietnamese char support"

# Push
git push origin main
```

---

## 🖥️ **SERVER - INITIAL SETUP**

```bash
# 1. SSH
ssh your_username@server_address

# 2. Clone (first time) or Pull (update)
git clone https://github.com/your-username/ViSemPar_new1.git
# OR
cd ViSemPar_new1 && git pull

# 3. Run setup
cd ViSemPar_new1
bash setup_server.sh
# → Select Option 1: CLI Login
# → Paste HF token when asked

# 4. Verify
huggingface-cli whoami
```

---

## 🔑 **HUGGING FACE TOKEN**

**Get token:**
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Write" permission
4. Copy token: `hf_xxxxxxxxxxxxx`

**Login on server:**
```bash
huggingface-cli login
# Paste token
```

---

## 🧪 **RUN TESTS**

```bash
# Test 1: MTUP Preprocessing
python3 test_mtup_simple.py

# Test 2: MTUP Data Preparation
python3 quick_test_mtup.py

# Test 3: SMATCH Evaluation
python3 test_smatch.py

# Test 4: Real Data Evaluation
python3 evaluate_test_data.py
```

**Expected results:**
- ✅ All tests pass
- ✅ SMATCH F1 = 1.0 (self-match)
- ✅ Preprocessing ready

---

## 📊 **CHECK DATA**

```bash
# Count examples
grep -c "^#::snt" data/train_amr_1.txt
grep -c "^#::snt" data/train_amr_2.txt

# View first example
head -10 data/train_amr_1.txt

# Check file sizes
ls -lh data/
```

---

## 🚀 **TRAINING (Future)**

```bash
# Quick test (when train_mtup.py available)
python3 train_mtup.py --use-case quick_test

# Fast iteration
tmux new -s amr-training
python3 train_mtup.py --use-case fast_iteration
# Ctrl+B, D (detach)

# Reattach
tmux attach -t amr-training
```

---

## 📈 **MONITOR**

```bash
# GPU status
nvidia-smi

# Logs
tail -f logs/training_mtup.log

# Disk space
df -h
du -sh outputs/
```

---

## 🔧 **TROUBLESHOOTING**

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Re-login HF
huggingface-cli logout
huggingface-cli login

# Check Python
python3 --version

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 📝 **COMMON TASKS**

**Upload data to server:**
```bash
# From local
scp -r data/ user@server:~/ViSemPar_new1/
```

**Download results:**
```bash
# From local
scp -r user@server:~/ViSemPar_new1/outputs/ ./outputs/
```

**Check tmux sessions:**
```bash
tmux ls
tmux attach -t amr-training
tmux kill-session -t amr-training
```

---

## ✅ **VERIFICATION CHECKLIST**

```bash
# Run all checks
python3 test_mtup_simple.py && \
python3 quick_test_mtup.py && \
python3 test_smatch.py && \
python3 evaluate_test_data.py && \
echo "✅ ALL TESTS PASSED"
```

---

## 🎯 **ONE-LINER SETUP**

```bash
# Complete setup in one go
git pull && \
bash setup_server.sh && \
python3 test_mtup_simple.py && \
echo "✅ Setup complete!"
```

---

**Questions? See:**
- [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) - Full guide
- [`SERVER_QUICK_START.md`](SERVER_QUICK_START.md) - Server commands
- [`HUGGINGFACE_SETUP.md`](HUGGINGFACE_SETUP.md) - HF setup
