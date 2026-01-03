# Vietnamese AMR Baseline 7B - Bug Fixes Documentation

## 📌 Quick Start

### Tình huống hiện tại

Model vừa train xong nhưng kết quả **THẢM HỌA**:
- Old model: 82.7% valid AMRs ✅
- New model: 5.3% valid AMRs ❌

### Nguyên nhân

**3 bugs nghiêm trọng** trong code đã được identify và fix:

1. **Instruction masking sai** - Tokenization mismatch
2. **Parenthesis check sai** - Đếm trong string gốc thay vì accumulated
3. **Overfitting** - Loss 0.0011 quá thấp

### Giải pháp

✅ **Bugs đã được fix**, sẵn sàng retrain!

---

## 📚 Documentation Files

### Cho người dùng (Vietnamese)

1. **[CRITICAL_ANALYSIS_AND_FIXES.md](CRITICAL_ANALYSIS_AND_FIXES.md)** ⭐ BẮT ĐẦU TẠI ĐÂY
   - Phân tích chi tiết vấn đề
   - Giải thích bugs bằng tiếng Việt
   - Kế hoạch hành động
   - So sánh old vs new model

### Cho developers (Technical)

2. **[BUGS_IDENTIFIED.md](BUGS_IDENTIFIED.md)**
   - Technical details về bugs
   - Code examples và fix
   - Root cause analysis
   - Impact assessment

### Scripts

3. **[VALIDATE_BEFORE_RETRAIN.sh](VALIDATE_BEFORE_RETRAIN.sh)**
   - Validate training data
   - Test tokenization fix
   - Quick training test (10 samples)
   - Run before full retrain

4. **[diagnose_tokenization.py](diagnose_tokenization.py)**
   - Test instruction masking
   - Verify tokenization consistency
   - Compare separate vs combined tokenization

---

## 🔧 Files Modified

### Core fixes

1. **[train_baseline_fixed.py](train_baseline_fixed.py)**
   - ✅ Fixed instruction masking (lines 227-270)
   - ✅ Use `encode(..., add_special_tokens=False)`
   - ✅ Correct prompt position calculation

2. **[predict_baseline_fixed.py](predict_baseline_fixed.py)**
   - ✅ Fixed balance check (lines 142-147)
   - ✅ Check accumulated text
   - ✅ Error handling for missing samples (lines 207-238)

### Configuration (unchanged)

- [config/config_fixed.py](config/config_fixed.py) - Prompt template và configs
- [TRAIN_BASELINE_FIXED.sh](TRAIN_BASELINE_FIXED.sh) - Training script
- [TEST_FIXED_MODEL.sh](TEST_FIXED_MODEL.sh) - Testing script

---

## 🚀 How to Retrain

### Option A: Validate first (RECOMMENDED)

```bash
# 1. SSH to server
ssh islabworker2@islab-server2

# 2. Go to project directory
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# 3. Pull latest fixes
git pull

# 4. Run validation
conda activate baseline_final
bash VALIDATE_BEFORE_RETRAIN.sh

# 5. If validation passes, retrain
bash TRAIN_BASELINE_FIXED.sh
```

### Option B: Retrain immediately

```bash
# 1. SSH to server
ssh islabworker2@islab-server2

# 2. Go to project directory
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# 3. Pull latest fixes
git pull

# 4. Retrain
conda activate baseline_final
bash TRAIN_BASELINE_FIXED.sh
```

**Training time:** ~4-5 hours on A6000 48GB

---

## 📊 Expected Results

### Before fixes (OLD - BROKEN)

```
Total AMRs: 150
Valid AMRs: 8 (5.3%)
Invalid AMRs: 137 (91.3%)

Issues:
- Unmatched parentheses: 137/150
- Duplicate nodes: Many
- Missing samples: 2
```

### After fixes (EXPECTED)

```
Total AMRs: 150
Valid AMRs: ~120-135 (80-90%)  ← Target
Invalid AMRs: ~15-30 (10-20%)

Improvements:
- Correct instruction masking
- No explanations after AMR
- All 150 samples generated
- Proper Penman format
```

**Note:** Cần test early checkpoints (200, 400, 600) để tránh overfitting.

---

## 🐛 Bugs Summary

| Bug | Location | Status | Impact |
|-----|----------|--------|--------|
| #1: Instruction masking | [train_baseline_fixed.py:227-270](train_baseline_fixed.py#L227-L270) | ✅ Fixed | Critical - Model học sai |
| #2: Balance check | [predict_baseline_fixed.py:142-147](predict_baseline_fixed.py#L142-L147) | ✅ Fixed | Critical - Output broken |
| #3: Overfitting | Training config | ⏳ Needs testing | High - Poor generalization |
| #4: Data quality | Training data | ⏳ Needs validation | Medium - Garbage in, garbage out |
| #5: Missing samples | [predict_baseline_fixed.py:207-238](predict_baseline_fixed.py#L207-L238) | ✅ Fixed | Low - Error handling added |

---

## 📝 Technical Details

### Bug #1: Instruction Masking

**Problem:**
```python
# WRONG: Tokenize separately
prompt_encoding = tokenizer(prompt, ...)
prompt_length = len(prompt_encoding['input_ids'][0])
labels[:prompt_length] = -100  # Mismatch!
```

**Fix:**
```python
# CORRECT: Encode without special tokens
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
amr_ids = tokenizer.encode(amr, add_special_tokens=False)
full_ids = prompt_ids + amr_ids + eos_ids
labels[:len(prompt_ids)] = -100  # Exact position!
```

### Bug #2: Balance Check

**Problem:**
```python
# WRONG: Check in original string
for line in lines:
    amr_lines.append(line)
    if amr.count('(') == amr.count(')'):  # Always checking full string!
        found_amr_end = True
```

**Fix:**
```python
# CORRECT: Check accumulated
for line in lines:
    amr_lines.append(line)
    accumulated = '\n'.join(amr_lines)
    if accumulated.count('(') == accumulated.count(')'):  # Check what we've seen!
        found_amr_end = True
```

---

## ⚠️ Important Notes

### During Training

1. **Monitor loss:**
   - Target: 0.05 - 0.15 (healthy range)
   - If < 0.05: overfitting, use early checkpoint
   - If > 0.20: undertrained, train more

2. **Save checkpoints:**
   - Every 200 steps
   - Keep checkpoint-200, 400, 600, 800 for testing

3. **Validation:**
   - Check predictions during training
   - Compare with ground truth
   - Validate format regularly

### After Training

1. **Test multiple checkpoints:**
   ```bash
   python predict_baseline_fixed.py --model outputs/.../checkpoint-200 ...
   python predict_baseline_fixed.py --model outputs/.../checkpoint-400 ...
   python predict_baseline_fixed.py --model outputs/.../checkpoint-600 ...
   ```

2. **Compare results:**
   - Use validation script for each checkpoint
   - Choose checkpoint with best valid AMR %
   - Final checkpoint không phải lúc nào cũng tốt nhất!

3. **Calculate SMATCH:**
   - Only after confirming high valid AMR %
   - Invalid AMRs → SMATCH không có ý nghĩa

---

## 📞 Next Steps

1. **Read:** [CRITICAL_ANALYSIS_AND_FIXES.md](CRITICAL_ANALYSIS_AND_FIXES.md)
2. **Decide:** Option A (validate first) vs Option B (retrain now)
3. **Execute:** Follow steps above
4. **Monitor:** Track training loss and checkpoints
5. **Test:** Multiple checkpoints after training
6. **Evaluate:** Compare with ground truth

---

## 🎯 Success Criteria

✅ **Training successful if:**
- Valid AMRs: > 75% (target: 80-90%)
- Proper Penman format
- Balanced parentheses
- No duplicate nodes
- All 150 samples generated
- SMATCH > old model (if old model SMATCH known)

❌ **Training failed if:**
- Valid AMRs: < 70%
- Loss < 0.01 (overfitting)
- Missing samples
- Unbalanced parentheses > 20%

---

## 📚 Additional Resources

- [config/config_fixed.py](config/config_fixed.py) - Configuration details
- [validate_vietnamese_output.py](validate_vietnamese_output.py) - Validation tool
- [TRAINING_FIXES_SUMMARY.md](TRAINING_FIXES_SUMMARY.md) - Original fix documentation
- [THESIS_CHAPTER_MTUP.md](THESIS_CHAPTER_MTUP.md) - MTUP results for comparison

---

**Last updated:** 2026-01-03

**Status:** ✅ Fixes ready, ⏳ Pending retrain

**Contact:** Check with user before retraining
