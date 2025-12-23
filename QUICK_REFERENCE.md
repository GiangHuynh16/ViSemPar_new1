# Quick Reference Card - Vietnamese AMR Parser

## 🚀 Quick Start Commands

```bash
# 1. Setup (run once)
./setup_directories.sh

# 2. Start tmux session
tmux new -s amr-training

# 3. Activate environment
conda activate amr-parser

# 4. Start training
python train.py --model 7b

# 5. Detach from tmux
Ctrl+b, then d
```

---

## 📺 tmux Commands

### Starting tmux
```bash
tmux new -s amr-training          # Create new session
tmux new -s <name>                # Create named session
```

### Attaching/Detaching
```bash
tmux attach -t amr-training       # Attach to session
tmux a -t amr-training            # Shorthand
Ctrl+b, d                         # Detach (inside tmux)
```

### Managing Sessions
```bash
tmux ls                           # List all sessions
tmux kill-session -t amr-training # Kill a session
Ctrl+b, :kill-session            # Kill current session (inside tmux)
```

### Window Management (inside tmux)
```
Ctrl+b, c        # Create new window
Ctrl+b, n        # Next window
Ctrl+b, p        # Previous window
Ctrl+b, 0-9      # Switch to window by number
Ctrl+b, ,        # Rename current window
Ctrl+b, &        # Close current window
```

### Pane Management (inside tmux)
```
Ctrl+b, %        # Split vertically
Ctrl+b, "        # Split horizontally
Ctrl+b, arrow    # Navigate between panes
Ctrl+b, x        # Close current pane
Ctrl+b, z        # Zoom/unzoom current pane
```

### Scrolling (inside tmux)
```
Ctrl+b, [        # Enter scroll mode
Arrow keys       # Scroll up/down
q                # Exit scroll mode
```

---

## 🎯 Training Commands

### Basic Training
```bash
python train.py --model 7b                          # 7B model (stable)
python train.py --model 14b                         # 14B model (better)
```

### Custom Parameters
```bash
python train.py --model 7b --epochs 20              # Custom epochs
python train.py --model 7b --batch-size 4           # Custom batch size
python train.py --model 7b --epochs 15 --batch-size 2  # Combined
```

### Pipeline Control
```bash
python train.py --model 7b --skip-inference         # Skip inference
python train.py --model 7b --skip-evaluation        # Skip evaluation
python train.py --model 7b --inference-batch-size 8 # Custom inference batch
```

### Get Help
```bash
python train.py --help                              # Show all options
```

---

## 📊 Monitoring Commands

### Check Logs
```bash
tail -f logs/training.log                          # Watch logs in real-time
tail -n 100 logs/training.log                      # Last 100 lines
cat logs/training.log                              # View full log
```

### Monitor GPU
```bash
nvidia-smi                                         # Check GPU once
watch -n 1 nvidia-smi                             # Update every second
watch -n 5 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

### Check Training Process
```bash
ps aux | grep python | grep train.py              # Find training process
htop                                               # Interactive process monitor
```

---

## 📁 File Management

### Check Outputs
```bash
ls outputs/                                        # List outputs
ls outputs/vlsp_amr_qwen_improved_v2/             # Check model files
ls outputs/checkpoints/                            # Check checkpoints
```

### Check Data
```bash
ls data/                                           # List data files
wc -l data/train_amr_1.txt                        # Count lines in file
head -n 5 data/train_amr_1.txt                    # View first 5 lines
```

### Disk Space
```bash
df -h                                              # Check disk space
du -sh outputs/                                    # Check outputs size
du -sh outputs/checkpoints/                        # Check checkpoints size
```

---

## 🔄 Common Workflows

### Workflow 1: Start Training
```bash
tmux new -s amr-training
conda activate amr-parser
cd /mnt/nghiepth/giangha/ViSemPar
python train.py --model 7b
# Wait for "STEP 1: LOADING DATA" to appear
Ctrl+b, d
```

### Workflow 2: Check Training Progress
```bash
tmux attach -t amr-training
# View progress
Ctrl+b, d
# Or check logs
tail -f logs/training.log
```

### Workflow 3: Monitor Resources
```bash
# Start tmux with monitoring
tmux new -s amr-monitoring
# Split screen horizontally
Ctrl+b, "
# In top pane: watch training logs
tail -f logs/training.log
# In bottom pane (navigate with Ctrl+b, arrow)
watch -n 1 nvidia-smi
```

### Workflow 4: Resume After Disconnection
```bash
# SSH back into server
ssh your-username@your-server.edu
# Check if session exists
tmux ls
# Reattach
tmux attach -t amr-training
```

---

## 🐛 Quick Troubleshooting

### Can't find tmux session
```bash
tmux ls                          # List sessions
ps aux | grep train.py          # Check if training is running
tail -f logs/training.log       # Check logs directly
```

### CUDA Out of Memory
```bash
# Option 1: Smaller batch size
python train.py --model 7b --batch-size 1

# Option 2: Use 7B model
python train.py --model 7b

# Option 3: Clear GPU memory
nvidia-smi --gpu-reset
```

### Training Seems Stuck
```bash
# Check if process is running
ps aux | grep train.py

# Check GPU usage
nvidia-smi

# Check last log entries
tail -n 50 logs/training.log

# Check if disk is full
df -h
```

### Permission Denied
```bash
# Check permissions
ls -la outputs/

# Fix if needed (no sudo)
chmod -R 755 outputs/

# Or with sudo if you have access
sudo chown -R $USER:$USER outputs/
```

---

## 📝 File Locations

```
ViSemPar/
├── train.py                    # Main training script (USE THIS)
├── setup_directories.sh        # Run this first
├── TRAINING_GUIDE.md          # Complete guide
├── QUICK_REFERENCE.md         # This file
│
├── data/                       # Place training data here
│   ├── train_amr_1.txt
│   ├── train_amr_2.txt
│   ├── public_test.txt
│   └── private_test.txt
│
├── outputs/                    # Training outputs
│   ├── vlsp_amr_qwen_improved_v2/    # Final model
│   ├── checkpoints/                   # Training checkpoints
│   └── *_results_*/                   # Test predictions
│
└── logs/
    └── training.log            # Training logs
```

---

## ⚡ Essential Python One-Liners

```python
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Check Transformers version
python -c "import transformers; print(transformers.__version__)"
```

---

## 💡 Pro Tips

1. **Always start with tmux** - Never train without it
2. **Monitor early** - Check logs in first 5 minutes to catch errors
3. **Use 7B first** - More stable, validate setup before using 14B
4. **Save early** - Copy important results immediately after training
5. **Clean checkpoints** - Delete old checkpoints to save disk space

---

## 📞 Emergency Commands

### Kill Training (if needed)
```bash
# Find process ID
ps aux | grep train.py

# Kill by PID
kill <PID>

# Or force kill
kill -9 <PID>

# Kill tmux session
tmux kill-session -t amr-training
```

### Clear GPU Memory
```bash
# Kill all Python processes (CAUTION!)
pkill python

# Or specific process
kill $(ps aux | grep train.py | awk '{print $2}')
```

---

**Keep this file open in another terminal for quick reference!**
