# 🔧 Post-Processing Implementation Guide

## ✅ COMPLETED

Post-processing pipeline has been added to `evaluate_mtup_model.py`

## 📊 Complete Pipeline

```
Input Sentence
      ↓
[PREPROCESSING] ← Prepare training data
      ↓
[TRAINING] ← Learn 2 tasks (structure + variables)
      ↓
[INFERENCE] ← Generate raw AMR
      ↓
[POST-PROCESSING] ← **NEW!** Repair errors
      ↓
[EVALUATION] ← SMATCH scoring
```

## 🔧 Post-Processing Steps

### 1. Remove Prompt Leakage
```python
# Remove: "2 bước", "Hướng dẫn:", "AMR hoàn chỉnh:", etc.
# Fix: (2 bước) → removed
```

### 2. Extract Valid Structure
```python
# Find first '(' and start from there
# Ensures we have valid AMR start
```

### 3. Balance Parentheses (Stack-Based)
```python
# Fix: (a / and :op1(...) :op2(...))))
# To:  (a / and :op1(...) :op2(...))
#
# Algorithm: Track opening/closing with stack
# Skip extra ')', add missing ')'
```

### 4. Rename Duplicate Variables
```python
# Fix: (n / nhớ :agent (n / tôi))
# To:  (n / nhớ :agent (n2 / tôi))
#
# Algorithm: Track seen variables, rename duplicates
```

### 5. Validate PENMAN Format
```python
# Ensure starts with '(' and ends with ')'
# Clean up whitespace
```

## 📈 Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total errors** | 49/150 (33%) | ~20-25/150 (13-17%) | **-50% errors** |
| **Success rate** | 67% (101/150) | ~83-87% (125-130/150) | **+16-20%** |
| **F1 Score** | 0.48 | **0.54-0.56** | **+12-17%** |

### Error Reduction Breakdown:

| Error Type | Count Before | Expected After | Fix Method |
|------------|--------------|----------------|------------|
| Unmatched parens | 30 | ~5 | Stack balancing |
| Duplicate vars | 10 | ~0 | Variable renaming |
| Prompt leak | 4 | ~0 | Text removal |
| Node not found | 5 | ~5 | (Still need work) |
| Other | 0 | ~10-15 | (New edge cases) |
| **TOTAL** | **49** | **~20-25** | **-50%** |

## 🚀 How to Test

### On Server:

```bash
# 1. Pull latest code
cd ~/ViSemPar_new1
git pull origin main

# 2. Run evaluation with post-processing
bash RUN_FULL_EVALUATION_TMUX.sh

# 3. Or quick test (10 samples)
python3 evaluate_mtup_model.py \
  --checkpoint outputs/checkpoints_mtup/mtup_full_training_final \
  --test-file data/public_test_ground_truth.txt \
  --max-samples 10
```

### Expected Output:

```
================================================================================
EVALUATION RESULTS
================================================================================

Processed: ~127-130/150 examples  ← Up from 101
Errors:    ~20-23                 ← Down from 49

================================================================================
SMATCH SCORES
================================================================================
  Precision: ~0.55-0.57  ← Up from 0.50
  Recall:    ~0.52-0.55  ← Up from 0.47
  F1:        0.54-0.56   ← Up from 0.48 ✅
================================================================================
```

## 🔍 Based on Literature

### ViAMR (VLSP 2025 - Paper 31)
- Constraint-aware inference stack
- **String-level repair steps** ← We implemented this!
- Handles Reentrancy (variable sharing)

### IBM Transition AMR Parser
- Parenthesis balancing
- Post-processing improves F-scores significantly

### Common AMR Issues
- "Neural network output does not guarantee PENMAN notation"
- "Fixing bugs in postprocessing leads to better F-scores"

## 📁 Code Changes

### File: `evaluate_mtup_model.py`

**Added**:
```python
def post_process_amr(amr_string: str) -> str:
    """
    6-step repair pipeline:
    1. Remove prompt leakage
    2. Extract valid structure
    3. Balance parentheses (stack-based)
    4. Rename duplicate variables
    5. Validate format
    6. Clean whitespace
    """
    # ... implementation ...
```

**Modified**:
```python
def generate_mtup_prediction(...):
    # ... existing code ...

    # POST-PROCESSING: Apply repair pipeline
    final_amr = post_process_amr(final_amr)  # ← NEW!

    return final_amr
```

## 🎯 Next Steps

### After This Test:

1. **If F1 = 0.54-0.56**: ✅ Success! Post-processing works
2. **If F1 = 0.50-0.53**: 🟡 Partial success, analyze remaining errors
3. **If F1 < 0.50**: ❌ Post-processing needs adjustment

### Future Improvements:

1. **Advanced validation**:
   - Check node references (fix "Node not found" errors)
   - Validate relation labels
   - Use penman library for parsing

2. **Train longer**:
   - Current: 1-2 epochs
   - Try: 3-5 epochs
   - Expected: +3-5% F1

3. **Better template**:
   - Try v5_cot (Chain-of-Thought)
   - Add explicit parenthesis balancing instructions

## 🐛 Troubleshooting

### If errors increase:

Check if post-processing is breaking valid AMRs:
```python
# Add debug flag to compare before/after
if debug:
    print("Before:", raw_amr)
    print("After:", post_process_amr(raw_amr))
```

### If F1 doesn't improve:

1. Check log for new error types
2. Analyze which fixes are working
3. May need to adjust post-processing logic

## 📚 References

- [ViAMR: Fine-tuning LLMs for Vietnamese AMR](https://aclanthology.org/2025.vlsp-1.31.pdf)
- [IBM Transition AMR Parser](https://github.com/IBM/transition-amr-parser)
- [PENMAN Notation](https://penman.readthedocs.io/en/latest/notation.html)

---

**Status**: ✅ Ready to test
**Commit**: `7a500d5`
**Expected F1**: **0.54-0.56** (from 0.48)
