# Vietnamese AMR Parser - Complete Training Guide

## 🚀 Quick Start

This guide will walk you through training the Vietnamese AMR parser on your university server using a single unified script.

---

## 📋 Prerequisites

1. **GPU Access**: CUDA-enabled GPU (recommended: 16GB+ VRAM)
2. **Python**: Version 3.8-3.10
3. **Conda/Miniconda**: For environment management
4. **tmux**: For persistent sessions (in case SSH disconnects)

---

## 🔧 Step 1: Initial Setup

### 1.1 Create Project Directory

```bash
# Navigate to your workspace
cd /mnt/nghiepth/giangha

# If ViSemPar doesn't exist, clone or create it
# git clone <your-repo-url> ViSemPar
cd ViSemPar
```

### 1.2 Create Required Directories

**IMPORTANT**: Create these directories BEFORE running training to avoid permission issues:

```bash
# Create all required directories at once
mkdir -p data outputs logs outputs/checkpoints

# Verify they were created
ls -la
# You should see: data/ outputs/ logs/ and others
```

### 1.3 Place Training Data

```bash
# Copy your training data files into the data directory
# Your data files should be:
# - train_amr_1.txt
# - train_amr_2.txt
# - public_test.txt (optional)
# - public_test_ground_truth.txt (optional)
# - private_test.txt (optional)

# Example:
# cp /path/to/your/train_amr_1.txt data/
# cp /path/to/your/train_amr_2.txt data/

# Verify files exist
ls -la data/
```

---

## 🐍 Step 2: Python Environment Setup

### 2.1 Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n amr-parser python=3.10 -y

# Activate environment
conda activate amr-parser
```

### 2.2 Install PyTorch with CUDA

```bash
# Install PyTorch 2.0+ with CUDA 11.8 (adjust for your server's CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Verify CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

### 2.3 Install Dependencies

```bash
# Install all required packages
pip install --break-system-packages transformers==4.36.0 \
    datasets==2.14.6 \
    accelerate==0.24.1 \
    peft==0.7.1 \
    bitsandbytes==0.41.3 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    tqdm \
    pandas \
    smatch

# Verify installation
python -c "from transformers import AutoTokenizer; print('✓ Transformers installed')"
```

---

## 🖥️ Step 3: Using tmux for Persistent Training

**Why tmux?** 
- Training can take hours or days
- SSH disconnections won't stop training
- You can detach and reattach anytime

### 3.1 Start tmux Session

```bash
# Start a new tmux session named "amr-training"
tmux new -s amr-training
```

### 3.2 tmux Essential Commands

**Inside tmux:**
- `Ctrl+b` then `d` - Detach from session (training continues)
- `Ctrl+b` then `[` - Scroll mode (use arrow keys, press `q` to exit)
- `Ctrl+b` then `c` - Create new window
- `Ctrl+b` then `n` - Next window
- `Ctrl+b` then `p` - Previous window

**Outside tmux:**
```bash
# List all tmux sessions
tmux ls

# Reattach to your session
tmux attach -t amr-training

# Or shorthand
tmux a -t amr-training

# Kill a session (if needed)
tmux kill-session -t amr-training
```

---

## 🎯 Step 4: Training

### 4.1 Basic Training (Recommended)

```bash
# Inside tmux session
cd /mnt/nghiepth/giangha/ViSemPar

# Activate environment
conda activate amr-parser

# Start training with 7B model (more stable, less memory)
python train.py --model 7b

# Training will show progress like:
# ============================================================
# STEP 1: LOADING DATA
# ✓ Training samples: 3500
# ✓ Validation samples: 184
# ============================================================
# STEP 2: LOADING MODEL
# Using: Qwen/Qwen2.5-7B-Instruct (7B - More stable)
# ...
```

### 4.2 Training with Custom Parameters

```bash
# Use 14B model (better performance, needs more memory)
python train.py --model 14b

# Override number of epochs
python train.py --model 7b --epochs 10

# Custom batch size
python train.py --model 7b --batch-size 4

# Skip inference after training
python train.py --model 7b --skip-inference

# Combine options
python train.py --model 7b --epochs 20 --batch-size 2
```

### 4.3 View All Options

```bash
python train.py --help
```

### 4.4 Detach from tmux

**After starting training, detach safely:**
```bash
# Press: Ctrl+b, then d
# You'll see: [detached (from session amr-training)]
```

Now you can:
- Close your terminal
- Disconnect from SSH
- Training continues in background

---

## 📊 Step 5: Monitoring Training

### 5.1 Reattach to tmux

```bash
# Reconnect to your session anytime
tmux attach -t amr-training
```

### 5.2 Check Logs

```bash
# In another terminal window (or tmux window):
cd /mnt/nghiepth/giangha/ViSemPar

# Watch training logs in real-time
tail -f logs/training.log

# Check latest log entries
tail -n 50 logs/training.log
```

### 5.3 Monitor GPU Usage

```bash
# In another tmux window (Ctrl+b then c):
watch -n 1 nvidia-smi

# Or one-time check:
nvidia-smi
```

---

## 📁 Step 6: Output Files

After training completes, you'll find:

```
ViSemPar/
├── outputs/
│   ├── vlsp_amr_qwen_improved_v2/        # Final trained model
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin
│   │   └── ...
│   ├── checkpoints/                       # Training checkpoints
│   │   └── vlsp_amr_qwen_improved_v2_20231215_143022/
│   ├── public_results_20231215_150030/    # Public test results (if available)
│   │   ├── vietnamese_amr_public.txt
│   │   └── evaluation_metrics.json
│   └── private_results_20231215_150045/   # Private test results
│       └── vietnamese_amr_private.txt
└── logs/
    └── training.log                       # Complete training log
```

---

## 🔄 Step 7: Common Workflows

### 7.1 Training from Scratch

```bash
# Start tmux
tmux new -s amr-training

# Activate environment
conda activate amr-parser

# Go to project
cd /mnt/nghiepth/giangha/ViSemPar

# Start training
python train.py --model 7b --epochs 15

# Detach: Ctrl+b, d
```

### 7.2 Resume Checking Training

```bash
# Reattach to session
tmux attach -t amr-training

# You'll see current training progress
```

### 7.3 Run Only Inference (After Training)

```bash
# If you already have a trained model and want to run inference only
python train.py --skip-training --model-path outputs/vlsp_amr_qwen_improved_v2

# Note: Model loading feature needs to be implemented
```

---

## 🐛 Troubleshooting

### Issue 1: "Missing required directories"

**Error:**
```
❌ Missing required directories:
   /mnt/nghiepth/giangha/ViSemPar/outputs
```

**Solution:**
```bash
cd /mnt/nghiepth/giangha/ViSemPar
mkdir -p data outputs logs outputs/checkpoints
```

### Issue 2: "CUDA out of memory"

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Option 1: Use smaller batch size
python train.py --model 7b --batch-size 1

# Option 2: Use 7B model instead of 14B
python train.py --model 7b

# Option 3: Reduce sequence length in config/config.py
# Change MAX_SEQ_LENGTH from 2048 to 1024
```

### Issue 3: "Missing training files"

**Error:**
```
❌ Missing training files:
   train_amr_1.txt
```

**Solution:**
```bash
# Copy your data files to the data/ directory
cp /path/to/train_amr_1.txt data/
cp /path/to/train_amr_2.txt data/
```

### Issue 4: Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'outputs/...'
```

**Solution:**
```bash
# Check directory permissions
ls -la outputs/

# If needed, change permissions
chmod -R 755 outputs/

# If you need sudo, create directories beforehand:
sudo mkdir -p outputs/checkpoints
sudo chown -R $USER:$USER outputs/
```

### Issue 5: tmux session lost

**Can't find your session?**
```bash
# List all sessions
tmux ls

# If it doesn't exist, it may have crashed
# Check if training process is still running:
ps aux | grep python | grep train.py

# If running, you can monitor it directly:
tail -f logs/training.log
```

### Issue 6: Tensor Dimension Mismatch

**Error:**
```
ValueError: expected sequence of length 146 at dim 1 (got 113)
```

**Solution:**
This is **FIXED** in the new `train.py` file. The script now uses `DataCollatorForSeq2Seq` instead of `DataCollatorForLanguageModeling`, which properly handles variable-length sequences.

If you still see this error, make sure you're using the updated `train.py` file.

---

## ⚙️ Advanced Configuration

### Modify Training Parameters

Edit `config/config.py` to adjust:

```python
# Training Configuration
TRAINING_CONFIG = {
    "learning_rate": 2e-4,           # Adjust learning rate
    "num_train_epochs": 15,          # Change default epochs
    "per_device_train_batch_size": 2, # Change batch size
    "gradient_accumulation_steps": 8, # Effective batch = batch_size × this
    # ... other parameters
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 128,                        # LoRA rank
    "lora_alpha": 256,               # LoRA alpha
    # ... other parameters
}
```

### Change Model

In `train.py`, the default models are:
- `--model 7b`: Qwen2.5-7B-Instruct (more stable)
- `--model 14b`: Qwen2.5-14B-Instruct (better performance)

You can modify the code to use other models.

---

## 📝 Training Checklist

Before starting training, verify:

- [ ] Directories created: `data/`, `outputs/`, `logs/`, `outputs/checkpoints/`
- [ ] Training data files in `data/` directory
- [ ] Conda environment activated: `conda activate amr-parser`
- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] tmux session started: `tmux new -s amr-training`
- [ ] In correct directory: `cd /mnt/nghiepth/giangha/ViSemPar`

Start training:
```bash
python train.py --model 7b
```

Detach tmux:
```
Ctrl+b, then d
```

---

## 🎓 Tips & Best Practices

### 1. **Always Use tmux**
- Never run long training jobs directly in terminal
- SSH can disconnect, killing your training
- tmux keeps processes alive

### 2. **Monitor Progress**
- Check logs regularly: `tail -f logs/training.log`
- Monitor GPU: `watch -n 1 nvidia-smi`
- Look for decreasing loss values

### 3. **Save Checkpoints**
- The script auto-saves every 500 steps (configurable)
- Don't delete `outputs/checkpoints/` until training is done
- You can resume from checkpoints if training crashes

### 4. **Start Small**
- First run: Use `--model 7b` with default settings
- If it works, try `--model 14b` or more epochs
- Don't start with 50 epochs on first try

### 5. **Backup Your Work**
- Copy final model to safe location after training
- Save important output files
- Keep training logs for reference

### 6. **Resource Management**
- Check GPU memory before starting: `nvidia-smi`
- One training job per GPU
- Use `--batch-size 1` if memory is tight

---

## 🚀 Example Training Session

Complete example from start to finish:

```bash
# 1. SSH into server
ssh your-username@your-server.edu

# 2. Navigate to project
cd /mnt/nghiepth/giangha/ViSemPar

# 3. Verify directories
ls -la
# Should see: data/ outputs/ logs/

# 4. Check data files
ls data/
# Should see: train_amr_1.txt, train_amr_2.txt

# 5. Start tmux
tmux new -s amr-training

# 6. Activate environment
conda activate amr-parser

# 7. Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True

# 8. Start training
python train.py --model 7b --epochs 15

# 9. Wait for training to start (you'll see the banner and "STEP 1: LOADING DATA")

# 10. Detach from tmux
# Press: Ctrl+b, then d

# 11. Training continues in background. Disconnect from SSH if needed.

# 12. Later, reconnect
ssh your-username@your-server.edu
tmux attach -t amr-training

# 13. Check progress, see if training is complete

# 14. When done, check outputs
ls outputs/vlsp_amr_qwen_improved_v2/

# 15. Exit tmux
exit

# 16. Copy results
cp -r outputs/vlsp_amr_qwen_improved_v2 ~/backup/
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check the error message** - Often tells you what's wrong
2. **Read the logs**: `tail -f logs/training.log`
3. **Check GPU memory**: `nvidia-smi`
4. **Verify data files**: `ls data/`
5. **Review this guide** - Most issues are covered here

---

## 📚 Additional Resources

- **Transformers Documentation**: https://huggingface.co/docs/transformers
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **tmux Cheat Sheet**: https://tmuxcheatsheet.com/
- **VLSP 2025**: https://vlsp.org.vn/

---

## ✅ Success Indicators

Your training is successful if you see:

```
✅ TRAINING COMPLETED
Final train loss: 0.3456
✓ Model saved to outputs/vlsp_amr_qwen_improved_v2
🎉 PIPELINE COMPLETE!
```

And you have files in:
- `outputs/vlsp_amr_qwen_improved_v2/` (trained model)
- `outputs/checkpoints/` (training checkpoints)
- `outputs/public_results_*/` (test predictions, if available)

---

**Good luck with your training! 🚀**
