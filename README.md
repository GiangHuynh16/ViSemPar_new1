# Vietnamese AMR Parser - Unified Training System

**Complete training pipeline for Vietnamese Abstract Meaning Representation (AMR) parsing using Qwen2.5 models with LoRA fine-tuning.**

---

## ✨ What's New (Updated Version)

### 🔧 Key Improvements

1. **Single Unified Training Script** (`train.py`)
   - Combines training, inference, and evaluation
   - Replaces separate `train_standard.py` and `main.py`
   - Cleaner, more maintainable code

2. **Fixed Tensor Dimension Mismatch Error**
   - Uses `DataCollatorForSeq2Seq` instead of `DataCollatorForLanguageModeling`
   - Properly handles variable-length sequences
   - No more "expected sequence of length X at dim 1 (got Y)" errors

3. **No Automatic Directory Creation**
   - Removed auto-creation that requires sudo permissions
   - Manual directory setup before training (see below)
   - Better for university server environments

4. **Comprehensive tmux Guide**
   - Complete instructions for persistent training sessions
   - Handles SSH disconnections gracefully
   - Essential commands and workflows

5. **Better Error Messages**
   - Clear guidance when directories/files are missing
   - Helpful suggestions for common issues
   - Improved logging throughout

---

## 🚀 Quick Start (3 Steps)

### Step 1: Setup Directories

```bash
cd ViSemPar
./setup_directories.sh
```

Or manually:
```bash
mkdir -p data outputs logs outputs/checkpoints
```

### Step 2: Add Your Data

```bash
cp /path/to/train_amr_1.txt data/
cp /path/to/train_amr_2.txt data/
# Optional: test files
cp /path/to/public_test.txt data/
```

### Step 3: Start Training

```bash
# Start tmux session
tmux new -s amr-training

# Activate environment
conda activate amr-parser

# Train!
python train.py --model 7b

# Detach: Ctrl+b, then d
```

---

## 📚 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete step-by-step training guide with tmux
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference for common commands
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Technical improvements for SMATCH scores
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Overall project information

---

## 🎯 Training Options

### Basic Usage

```bash
# 7B model (recommended for first run)
python train.py --model 7b

# 14B model (better performance, needs more memory)
python train.py --model 14b
```

### Custom Parameters

```bash
# Custom epochs
python train.py --model 7b --epochs 20

# Custom batch size
python train.py --model 7b --batch-size 4

# Combined
python train.py --model 7b --epochs 15 --batch-size 2
```

### Pipeline Control

```bash
# Skip inference
python train.py --model 7b --skip-inference

# Skip evaluation
python train.py --model 7b --skip-evaluation

# Custom inference batch size
python train.py --model 7b --inference-batch-size 8
```

### View All Options

```bash
python train.py --help
```

---

## 📁 Project Structure

```
ViSemPar/
├── train.py                    # 🌟 Main training script (USE THIS)
├── setup_directories.sh        # Setup helper script
│
├── config/
│   └── config.py              # Configuration settings
│
├── src/
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessor.py        # AMR preprocessing
│   ├── postprocessor.py       # AMR postprocessing
│   ├── model.py               # Model trainer
│   ├── inference.py           # Inference engine
│   └── evaluation.py          # SMATCH evaluation
│
├── data/                       # Place training data here
│   ├── train_amr_1.txt        # Training file 1
│   ├── train_amr_2.txt        # Training file 2
│   └── *.txt                  # Test files (optional)
│
├── outputs/                    # Training outputs
│   ├── vlsp_amr_qwen_improved_v2/  # Final model
│   ├── checkpoints/                 # Training checkpoints
│   └── *_results_*/                 # Test predictions
│
└── logs/
    └── training.log           # Complete training log
```

---

## 🔧 Installation

### 1. Create Environment

```bash
conda create -n amr-parser python=3.10 -y
conda activate amr-parser
```

### 2. Install PyTorch

```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install Dependencies

```bash
pip install --break-system-packages \
    transformers==4.36.0 \
    datasets==2.14.6 \
    accelerate==0.24.1 \
    peft==0.7.1 \
    bitsandbytes==0.41.3 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    tqdm pandas smatch
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 🎮 Training Configuration

Edit `config/config.py` to customize:

### Model Configuration
- `MODEL_NAME`: Model to use
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: 2048)

### LoRA Settings
- `r`: LoRA rank (default: 128)
- `lora_alpha`: LoRA alpha (default: 256)
- `lora_dropout`: Dropout rate (default: 0.05)

### Training Settings
- `learning_rate`: Learning rate (default: 2e-4)
- `num_train_epochs`: Number of epochs (default: 15)
- `per_device_train_batch_size`: Batch size (default: 2)
- `gradient_accumulation_steps`: Accumulation steps (default: 8)

---

## 📊 Monitoring Training

### Reattach to tmux Session
```bash
tmux attach -t amr-training
```

### Watch Logs
```bash
tail -f logs/training.log
```

### Monitor GPU
```bash
watch -n 1 nvidia-smi
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Missing Directories
**Error:** `❌ Missing required directories`

**Solution:**
```bash
./setup_directories.sh
# Or manually:
mkdir -p data outputs logs outputs/checkpoints
```

### Issue 2: CUDA Out of Memory
**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Use smaller batch size
python train.py --model 7b --batch-size 1

# Use 7B model
python train.py --model 7b
```

### Issue 3: Tensor Dimension Mismatch
**Error:** `ValueError: expected sequence of length 146 at dim 1 (got 113)`

**Solution:** This is **FIXED** in the new `train.py`. Make sure you're using the updated version.

### Issue 4: Permission Denied
**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# Create directories beforehand
mkdir -p outputs logs
chmod -R 755 outputs logs

# Or with sudo if available
sudo mkdir -p outputs logs
sudo chown -R $USER:$USER outputs logs
```

---

## 📈 Expected Results

### Training Progress
```
============================================================
STEP 1: LOADING DATA
✓ Training samples: 3500
✓ Validation samples: 184
============================================================
STEP 2: LOADING MODEL
Using: Qwen/Qwen2.5-7B-Instruct (7B - More stable)
✓ Tokenizer loaded
✓ Model loaded
============================================================
STEP 3: CONFIGURING LORA
✓ LoRA configured
  Trainable parameters: 67,108,864 / 7,000,000,000
  Trainable: 0.96%
============================================================
...
```

### Output Files
After training completes:
- `outputs/vlsp_amr_qwen_improved_v2/` - Final trained model
- `outputs/checkpoints/` - Training checkpoints
- `outputs/public_results_*/` - Test predictions (if available)
- `logs/training.log` - Complete training log

---

## 🎓 Usage Tips

1. **Always use tmux** for long training sessions
2. **Start with 7B model** to validate setup
3. **Monitor logs** in first 5 minutes to catch errors early
4. **Check GPU memory** before starting: `nvidia-smi`
5. **Backup results** immediately after training completes

---

## 📝 Files Comparison

| Old Files | New Unified File | Purpose |
|-----------|------------------|---------|
| `train_standard.py` | ✅ `train.py` | Simplified training |
| `main.py` | ✅ `train.py` | Full pipeline |
| Both separate | ✅ One file | Easier maintenance |

**Use `train.py` for all training tasks.**

---

## 🔗 Related Documentation

- **Transformers**: https://huggingface.co/docs/transformers
- **PEFT/LoRA**: https://huggingface.co/docs/peft
- **Qwen Models**: https://huggingface.co/Qwen
- **VLSP 2025**: https://vlsp.org.vn/

---

## 🙏 Acknowledgments

- VLSP 2025 Competition
- Qwen Model Team
- Hugging Face Community
- University of Information Technology

---

## 📄 License

This project is for academic purposes (VLSP 2025 Competition).

---

## 👩‍💻 Author

**Giangiu**
- Final-year Information Systems Student
- University of Information Technology
- VLSP 2025 Participant (3rd Place - Semantic Parsing Task)

---

## 🆘 Support

For issues or questions:
1. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions
2. See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common commands
3. Review error messages and logs
4. Check [IMPROVEMENTS.md](IMPROVEMENTS.md) for technical details

---

**Happy Training! 🚀**

*Remember: Always start training in tmux to prevent loss from SSH disconnections!*
