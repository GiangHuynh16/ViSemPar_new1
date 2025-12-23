# 🚀 QUICK START GUIDE

Get your Vietnamese AMR Parser running in 5 minutes!

## Prerequisites

- **Python 3.8+** 
- **CUDA-capable GPU** (highly recommended)
- **16GB+ GPU memory** (for 14B model)
- **50GB+ disk space** (for model and outputs)

## Step 1: Setup Environment

```bash
# Clone or download the project
git clone [your-repo-url]
cd vietnamese-amr-parser

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Unsloth installation might take 5-10 minutes.

## Step 2: Prepare Data

Place your data files in the `data/` directory:

```
data/
├── train_amr_1.txt          ← Required
├── train_amr_2.txt          ← Required  
├── public_test.txt          ← Optional
├── public_test_ground_truth.txt  ← Optional
└── private_test.txt         ← Optional
```

The project comes with sample data already in place!

## Step 3: Verify Setup

```bash
python test_setup.py
```

This will check:
- Python version
- Dependencies
- CUDA availability
- Data files
- Module imports

If you see "✅ SETUP VERIFICATION COMPLETE!", you're ready!

## Step 4: Run Training

### Option A: Simple Run
```bash
python main.py
```

### Option B: With Setup Checks
```bash
./run.sh
```

### Option C: Custom Options
```bash
# Skip public test
python main.py --skip-public-test

# Skip private test  
python main.py --skip-private-test

# Skip training (inference only)
python main.py --skip-training --model-path outputs/your_model
```

## What Happens During Training?

```
1. 📊 LOADING DATA
   - Parses VLSP format AMR files
   - Validates syntax
   - Splits train/validation (95%/5%)
   - Shows statistics

2. 🤖 TRAINING MODEL
   - Loads Qwen 2.5 14B with 4-bit quantization
   - Applies LoRA for efficient fine-tuning
   - Trains for 15 epochs (~2-4 hours on T4 GPU)
   - Saves checkpoints every 500 steps
   - Logs to logs/training.log

3. 🔮 INFERENCE
   - Loads best checkpoint
   - Generates AMRs for test sets
   - Applies smart postprocessing
   - Saves in multiple formats

4. 📊 EVALUATION
   - Computes SMATCH scores (if ground truth available)
   - Generates detailed metrics
   - Saves evaluation reports
```

## Expected Training Time

| Hardware | Time |
|----------|------|
| T4 GPU (Colab) | 2-3 hours |
| V100 GPU | 1-2 hours |
| A100 GPU | 30-60 min |
| CPU only | ⚠️ Not recommended (days!) |

## Output Files

After training completes, check:

```
outputs/
├── vlsp_amr_qwen_improved_v2/
│   ├── merged_16bit/          ← Full precision model
│   └── merged_4bit/           ← Quantized for inference
├── public_test_results_YYYYMMDD_HHMMSS/
│   ├── vietnamese_amr_public_test_full.csv
│   ├── vietnamese_amr_public_test_submission.csv
│   ├── vietnamese_amr_public_test_vlsp.txt
│   └── evaluation_metrics.txt
└── private_test_results_YYYYMMDD_HHMMSS/
    └── [same structure]

logs/
└── training.log               ← Complete training log
```

## Configuration

Want to customize? Edit `config/config.py`:

```python
# Use smaller model for faster training
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

# Reduce epochs for quick test
TRAINING_CONFIG = {
    "num_train_epochs": 3,  # Instead of 15
}

# Adjust batch size for your GPU
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,  # Reduce if OOM
}
```

## Troubleshooting

### CUDA Out of Memory
```python
# In config/config.py
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,  # Reduce from 2
    "gradient_accumulation_steps": 16,  # Increase from 8
}
```

### Slow Training
- Check GPU is being used: `nvidia-smi`
- Ensure CUDA is available in PyTorch
- Consider using smaller model or fewer epochs

### Module Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Data Format Errors
- Ensure AMR files use VLSP format:
  ```
  #::snt sentence text
  (variable / concept
      :relation(other_concept))
  ```
- Check balanced parentheses
- Validate with provided test files

## Next Steps

After training:

1. **Check Results**
   ```bash
   cat outputs/public_test_results_*/evaluation_metrics.txt
   ```

2. **Review Predictions**
   ```bash
   cat outputs/public_test_results_*/vietnamese_amr_public_test_vlsp.txt | head -20
   ```

3. **Submit to VLSP**
   - Use `*_submission.csv` file
   - Contains: sentence | amr columns

4. **Push to Hugging Face** (optional)
   ```bash
   huggingface-cli login
   # Then model is automatically pushed during training
   ```

## Getting Help

- Check `README.md` for detailed documentation
- Review `logs/training.log` for errors
- Examine sample outputs in `outputs/`
- Open an issue on GitHub

## Performance Expectations

With this improved pipeline:

| Metric | Target | Previous |
|--------|--------|----------|
| Valid AMRs | 95%+ | ~85% |
| SMATCH F1 | 0.54-0.58 | 0.30 |
| Training Time | 2-3 hours | 2-3 hours |

---

**Ready? Let's go!** 🚀

```bash
python main.py
```
