# Comprehensive Fix for NaN Loss Issue

**Status**: Training shows `loss: 0.0, grad_norm: nan, learning_rate: 0.0` despite all fixes

**Fixes already applied**:
- ✅ Changed from `DataCollatorForLanguageModeling` to `default_data_collator`
- ✅ Added padding token masking (`labels[pad_tokens] = -100`)
- ✅ Switched from FP16 to BF16 in config
- ✅ Removed device_map CPU offload
- ✅ Using gradient checkpointing

**Current issue**: Loss is STILL NaN after all fixes

---

## Step 1: Stop Current Training

```bash
# SSH to server
ssh islabworker2@islab-server2

# Find and kill training process
pkill -f train_baseline.py
```

---

## Step 2: Pull Latest Debugging Tools

```bash
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Discard any local changes and pull latest code
git reset --hard origin/main
git pull origin main

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
```

---

## Step 3: Run Diagnostic Script

```bash
# Activate environment
conda activate baseline_final

# Run debug script
bash DEBUG_NAN_LOSS.sh
```

This will:
1. Check current config (verify BF16 is enabled)
2. Clear Python cache
3. Test BF16 forward pass with gradient checkpointing
4. Show PyTorch/CUDA versions and BF16 support

---

## Step 4: Interpret Results

### If BF16 test PASSES (✅):

BF16 is working, but TrainingArguments might not be using it. Solutions:

**Solution A: Verify config is loaded**
```bash
# Check if config has BF16
grep -A 3 "bf16" config/config.py

# Should show:
#   "bf16": True,
```

**Solution B: Add explicit BF16 to training script**

Edit `train_baseline.py` around line 370 and explicitly set:
```python
training_args = TrainingArguments(
    # ... other args ...
    fp16=False,
    bf16=True,  # Explicitly set here, not just from config
    # ... rest ...
)
```

**Solution C: Check if TrainingArguments overrides config**

Look for any code that might override the config after it's loaded.

---

### If BF16 test FAILS (❌):

BF16 is NOT supported on this system. Choose one of these solutions:

#### Option 1: Disable Gradient Checkpointing (Recommended)

Edit [train_baseline.py:311](train_baseline.py#L311):
```python
# BEFORE:
model.gradient_checkpointing_enable()

# AFTER:
# model.gradient_checkpointing_enable()  # DISABLED - causes NaN with FP16
```

Edit [config/config.py](config/config.py):
```python
TRAINING_CONFIG = {
    # ... other settings ...
    "fp16": True,   # Re-enable FP16
    "bf16": False,  # Disable BF16
    "gradient_checkpointing": False,  # Add this line
}
```

**Why this works**: FP16 alone works fine. FP16 + gradient checkpointing causes NaN. Disabling gradient checkpointing fixes it.

**Memory impact**: May need to reduce batch size or sequence length:
```python
MAX_SEQ_LENGTH = 1536  # Reduce from 2048
```

---

#### Option 2: Use FP32 (Most Stable, Slowest)

Edit [config/config.py](config/config.py):
```python
TRAINING_CONFIG = {
    # ... other settings ...
    "fp16": False,  # Disable FP16
    "bf16": False,  # Disable BF16
    # Model will use FP32 by default
}
```

Edit [train_baseline.py:292](train_baseline.py#L292):
```python
# Change torch_dtype to float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Use FP32
    device_map=None,
    trust_remote_code=True
)
```

**Memory impact**: Will need to reduce batch size and possibly sequence length:
```python
MAX_SEQ_LENGTH = 1024  # Reduce from 2048
"per_device_train_batch_size": 1,  # Reduce from 2
"gradient_accumulation_steps": 16,  # Increase to keep effective batch = 16
```

---

#### Option 3: Upgrade PyTorch (If Allowed)

BF16 requires PyTorch >= 1.10 and may not be supported on all GPUs.

Check current version:
```bash
python -c "import torch; print(torch.__version__)"
```

If version is < 2.0, upgrade:
```bash
pip install --upgrade torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Then retry BF16 test.

---

## Step 5: Apply Chosen Fix

After choosing Option 1, 2, or 3 above:

1. Make the code changes
2. Clear cache again:
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```

3. Test with quick run:
```bash
# Edit train_baseline.py and change epochs to 1 for testing
nano train_baseline.py
# Change: num_train_epochs = 1 (around line 31)

# Run training
bash VERIFY_AND_START.sh
```

4. Check first few log lines:
```bash
tail -f logs/training_baseline*.log
```

Look for:
```
{'loss': [NON-ZERO NUMBER], 'grad_norm': [NON-ZERO NUMBER], 'learning_rate': [NON-ZERO NUMBER]}
```

If loss is still 0.0 or NaN, try next option.

---

## Step 6: If Nothing Works - Minimal Config

As last resort, use minimal stable config:

Edit [config/config.py](config/config.py):
```python
MAX_SEQ_LENGTH = 512  # Very conservative

TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 15,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 5,
    "fp16": True,
    "bf16": False,
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",
}
```

Edit [train_baseline.py:311](train_baseline.py#L311):
```python
# Disable gradient checkpointing
# model.gradient_checkpointing_enable()
```

This MUST work because it's the most basic stable configuration.

---

## Expected Successful Output

After fix is applied, you should see:

```
Step    Loss         Grad Norm    LR           Epoch
10      8.9234       2.1456       0.000198     0.06
20      8.7123       1.8923       0.000196     0.13
30      8.5234       2.3421       0.000194     0.19
```

**Key indicators**:
- ✅ Loss > 0 and gradually decreasing
- ✅ Grad norm > 0 (typically 0.5 - 5.0)
- ✅ Learning rate > 0 (starts at 2e-4 after warmup)

---

## Quick Reference

### Test BF16:
```bash
bash DEBUG_NAN_LOSS.sh
```

### Clear cache:
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```

### Check config:
```bash
grep -A 3 "fp16\|bf16" config/config.py
```

### Kill training:
```bash
pkill -f train_baseline.py
```

### Start training:
```bash
conda activate baseline_final
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
bash VERIFY_AND_START.sh
```

---

## Summary of Solutions (Choose One)

| Option | Precision | Speed | Memory | Stability | Recommended For |
|--------|-----------|-------|--------|-----------|-----------------|
| **1. Disable gradient_checkpointing** | FP16 | Fast | Medium | High | **RECOMMENDED** |
| **2. Use FP32** | FP32 | Slow | High | Very High | If Option 1 fails |
| **3. Keep BF16** | BF16 | Fast | Medium | High | If BF16 test passes |

**My recommendation**: Start with **Option 1** (disable gradient checkpointing). It's the best balance of speed, memory, and stability.
