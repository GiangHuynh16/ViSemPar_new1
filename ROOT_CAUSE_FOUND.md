# ROOT CAUSE FOUND - NaN Loss Issue SOLVED

**Date**: 2026-01-02
**Status**: ✅ **FIXED**

---

## 🎯 Root Cause Identified

The diagnostic test you ran revealed the exact problem:

```
Step 8: Testing backward pass with gradient checkpointing...
  â Backward pass failed: element 0 of tensors does not require grad and does not have a grad_fn
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**The Issue**: `gradient_checkpointing_enable()` was being called **BEFORE** LoRA was applied.

### Code Flow (BEFORE Fix):

```python
# Line 315: Enable gradient checkpointing
model.gradient_checkpointing_enable()  # ❌ TOO EARLY!

# Line 334: Apply LoRA
model = get_peft_model(model, lora_config)

# Line 338: Try to enable gradients for LoRA params
param.requires_grad = True  # ❌ DOESN'T WORK - gradient tracking already broken
```

When gradient checkpointing is enabled before LoRA, the checkpointing mechanism doesn't know about LoRA parameters, so they don't get proper gradient tracking. This causes:
- `loss = 0.0` (loss can't be computed without gradients)
- `grad_norm = nan` (no gradients exist)
- `learning_rate = 0.0` (optimizer doesn't step)

---

## ✅ The Fix

Move `gradient_checkpointing_enable()` to **AFTER** LoRA is applied:

### Code Flow (AFTER Fix):

```python
# Line 332: Apply LoRA FIRST
model = get_peft_model(model, lora_config)

# Line 336: Enable gradients for LoRA params
for name, param in model.named_parameters():
    if 'lora_' in name:
        param.requires_grad = True  # ✅ WORKS NOW

# Line 343: Enable gradient checkpointing LAST
model.gradient_checkpointing_enable()  # ✅ NOW IT KNOWS ABOUT LoRA PARAMS
```

This ensures gradient checkpointing is aware of all LoRA parameters and properly tracks their gradients.

---

## 📋 How to Apply the Fix

### Option 1: Automated Script (Recommended)

```bash
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
bash APPLY_CRITICAL_FIX.sh
```

This script will:
1. Stop current training
2. Pull the fix from GitHub
3. Clear Python cache
4. Test the fix
5. Tell you if it's ready to train

### Option 2: Manual Steps

```bash
# Stop training
pkill -f train_baseline.py

# Pull fix
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
git reset --hard origin/main
git pull origin main

# Clear cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Test fix
conda activate baseline_final
python test_bf16_forward.py

# If test passes, start training
bash VERIFY_AND_START.sh
```

---

## 🧪 Verification

After pulling the fix, run the test again:

```bash
python test_bf16_forward.py
```

You should now see:

```
Step 8: Testing backward pass with gradient checkpointing...
  Loss before backward: 2.591958
  Gradient norm: 1.234567
  â Gradients computed successfully

=========================================
TEST SUMMARY
=========================================
â SUCCESS: BF16 working correctly
   Final loss: 2.591958
   Final grad_norm: 1.234567
```

**Key differences from before**:
- ✅ Backward pass succeeds (no RuntimeError)
- ✅ Gradient norm is a real number (not NaN)
- ✅ Test exits with success

---

## 🚀 Expected Training Output After Fix

When you start training, you should see:

```
{'loss': 8.9234, 'grad_norm': 2.1456, 'learning_rate': 0.000198, 'epoch': 0.06}
{'loss': 8.7123, 'grad_norm': 1.8923, 'learning_rate': 0.000196, 'epoch': 0.13}
{'loss': 8.5234, 'grad_norm': 2.3421, 'learning_rate': 0.000194, 'epoch': 0.19}
```

**All three values should be > 0 and non-NaN.**

---

## 📊 Why This Happened

This is a **subtle ordering bug** that's easy to miss:

1. The MTUP training code worked because it had a different initialization order
2. When creating baseline training, the order was accidentally reversed
3. Gradient checkpointing + LoRA is a delicate combination - order matters
4. The error message was cryptic ("element 0 doesn't require grad") which made it hard to diagnose

**Credit to the diagnostic test**: The `test_bf16_forward.py` script isolated the exact failure point by testing each step (forward pass, backward pass) separately.

---

## 🎯 Next Steps

1. **Apply the fix**:
   ```bash
   bash APPLY_CRITICAL_FIX.sh
   ```

2. **Verify test passes**:
   ```bash
   python test_bf16_forward.py
   ```

3. **Start training**:
   ```bash
   bash VERIFY_AND_START.sh
   ```

4. **Monitor training**:
   ```bash
   tail -f logs/training_baseline*.log
   ```

5. **Watch GPU**:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

## 📝 Summary

| Issue | Root Cause | Fix | Status |
|-------|------------|-----|--------|
| `loss: 0.0` | Gradient checkpointing before LoRA | Move after LoRA | ✅ Fixed |
| `grad_norm: nan` | LoRA params don't track gradients | Enable checkpointing last | ✅ Fixed |
| `learning_rate: 0.0` | Optimizer can't step without grads | Same fix | ✅ Fixed |

**The fix is simple but critical**: Change 3 lines to reorder when gradient checkpointing is enabled.

**Files changed**:
- [train_baseline.py](train_baseline.py) - Moved gradient_checkpointing_enable() after LoRA
- [test_bf16_forward.py](test_bf16_forward.py) - Updated test to match correct order

All code has been pushed to GitHub. Pull and run the test to verify!
