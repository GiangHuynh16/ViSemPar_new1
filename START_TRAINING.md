# Start MTUP Training - Complete Guide

## 🚀 Quick Start (Copy-Paste Commands)

### Step 1: SSH to Server & Setup tmux
```bash
# SSH to server
ssh your_username@server_ip
cd /mnt/nghiepth/giang/ViSemPar

# Pull latest code with all fixes
git pull origin main

# Start tmux session (IMPORTANT - prevents disconnection)
tmux new -s mtup_training

# If tmux session already exists, attach to it:
# tmux attach -t mtup_training
```

### Step 2: Activate Environment
```bash
# Activate conda environment
conda activate lora_py310

# Verify environment
which python3
# Should show: /home/islabworker1/anaconda3/envs/lora_py310/bin/python3

python3 --version
# Should show: Python 3.10.x
```

### Step 3: Verify Cleanup & Setup
```bash
# Check disk space
df -h /mnt/nghiepth | tail -1
du -sh outputs/

# Verify outputs directory is clean
ls -lh outputs/
ls -lh outputs/checkpoints_mtup/  # Should be empty

# Check GPU
nvidia-smi
```

### Step 4: Start Training
```bash
# Full training with all fixes (15 epochs, ~9 hours)
python3 train_mtup.py --use-case full_training --model qwen2.5-3b

# The script will show:
# ✅ Template loaded: v2_natural (FIXED - no parentheses leakage)
# ✅ Training: 15 epochs, batch_size=4, grad_accum=4
# ✅ Saving checkpoints every 200 steps
# ✅ Keeping best 5 checkpoints
# ✅ Will save to: outputs/checkpoints_mtup/mtup_full_training_final
```

### Step 5: Detach from tmux (Keep Training Running)
```bash
# Press: Ctrl+B, then D
# This detaches from tmux but training continues in background
```

### Step 6: Monitor Training Progress
```bash
# Re-attach to tmux session anytime
tmux attach -t mtup_training

# Or check logs without tmux
tail -f logs/training_mtup.log

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## 📊 Training Timeline

### Expected Duration: ~9 hours
```
Epoch  1/15:  ~36 min  [=====>                    ]
Epoch  2/15:  ~36 min  [==========>               ]
Epoch  3/15:  ~36 min  [===============>          ]
...
Epoch 15/15:  ~36 min  [==========================]

Total: ~9 hours
```

### Checkpoints Saved:
- Every 200 steps → `outputs/mtup_full_training_TIMESTAMP/checkpoint-200/`
- Every 200 steps → `outputs/mtup_full_training_TIMESTAMP/checkpoint-400/`
- ...
- Final model → `outputs/checkpoints_mtup/mtup_full_training_final/`

---

## ✅ Success Indicators

During training, watch for:
```
✓ Loading config_mtup.py...
✓ Model: Qwen/Qwen2.5-3B-Instruct
✓ Template: v2_natural (FIXED)
✓ Training samples: ~2500
✓ Validation samples: ~250
✓ Gradient checkpointing enabled

... training ...

{'loss': 0.8234, 'learning_rate': 0.00019, 'epoch': 1.0}
Checkpoint saved: checkpoint-200
{'loss': 0.7123, 'learning_rate': 0.00018, 'epoch': 2.0}
Checkpoint saved: checkpoint-400
...

✅ TRAINING COMPLETED
💾 Saving model to: /mnt/.../outputs/checkpoints_mtup/mtup_full_training_final
✅ All required files present
✅ Model ready for evaluation/upload
✅ Model size: 945.2 MB
```

---

## 🔧 Troubleshooting

### If Training Crashes:
```bash
# Check what happened
tail -100 logs/training_mtup.log

# Check GPU memory
nvidia-smi

# If OOM (Out of Memory), reduce batch size in train_mtup.py:
# --batch-size 2 --grad-accum 8  # Instead of 4/4
```

### If tmux Session Lost:
```bash
# List tmux sessions
tmux ls

# Attach to training session
tmux attach -t mtup_training

# If session doesn't exist but training is running:
ps aux | grep train_mtup
# Note the PID

# Create new tmux and attach to process
tmux new -s recovery
# Then monitor via logs
```

### If Disconnected from SSH:
```bash
# Just reconnect and reattach
ssh your_username@server_ip
tmux attach -t mtup_training

# Training will still be running!
```

---

## 📋 tmux Cheat Sheet

```bash
# Create new session
tmux new -s session_name

# List sessions
tmux ls

# Attach to session
tmux attach -t session_name

# Detach from session (keep it running)
Ctrl+B, then D

# Kill session (stop training)
tmux kill-session -t session_name

# Split window horizontally
Ctrl+B, then "

# Split window vertically
Ctrl+B, then %

# Switch between panes
Ctrl+B, then arrow keys

# Scroll in tmux
Ctrl+B, then [
# Then use arrow keys or Page Up/Down
# Press Q to exit scroll mode
```

---

## 🎯 After Training Completes

### 1. Verify Model Saved
```bash
ls -lh outputs/checkpoints_mtup/mtup_full_training_final/

# Should show:
# adapter_model.safetensors  (~945MB)
# adapter_config.json
# tokenizer_config.json
# special_tokens_map.json
# tokenizer.json
# ...
```

### 2. Evaluate Model
```bash
python3 evaluate_mtup_model.py \
  --checkpoint outputs/checkpoints_mtup/mtup_full_training_final \
  --output evaluation_results_v2.txt

# Expected: F1 > 0.50 (improved from 0.4751)
```

### 3. Push to HuggingFace
```bash
# Login once
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens

# Push model
python3 push_to_hf_cli.py --model-type mtup

# Will upload to: your-username/vietnamese-amr-mtup-7b
```

### 4. Cleanup (Optional)
```bash
# After pushing to HuggingFace, you can delete local checkpoints
# Keep only final model
rm -rf outputs/mtup_full_training_*/checkpoint-*

# Or delete everything except final
rm -rf outputs/mtup_full_training_*
# (Final model is in outputs/checkpoints_mtup/)
```

---

## 🔄 If You Need to Stop & Resume

### Stop Training:
```bash
# In tmux session, press Ctrl+C
# Or kill the session:
tmux kill-session -t mtup_training
```

### Resume Training (if using checkpoints):
```bash
# Training will auto-resume from last checkpoint if you restart
python3 train_mtup.py --use-case full_training --model qwen2.5-3b

# Or specify a checkpoint:
python3 train_mtup.py \
  --use-case full_training \
  --model qwen2.5-3b \
  --resume-from outputs/mtup_full_training_TIMESTAMP/checkpoint-1000
```

---

## 📞 Support

If anything goes wrong:
1. Check logs: `tail -100 logs/training_mtup.log`
2. Check GPU: `nvidia-smi`
3. Check disk: `df -h /mnt/nghiepth`
4. Check process: `ps aux | grep train_mtup`
5. Review error in tmux session

---

**Good luck with training! 🚀**

Expected completion: ~9 hours from start
Target F1 score: 0.50-0.52 (up from 0.4751)
