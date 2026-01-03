# Critical Bugs Identified in "Fixed" Model

## Summary

The model produces **91.3% invalid AMRs** (137/150) vs **17.3%** in the old "buggy" model (26/150).

This is a **catastrophic regression** caused by multiple bugs in both training and prediction.

---

## Bug #1: Instruction Masking Tokenization Mismatch (CRITICAL)

**Location:** [train_baseline_fixed.py:243-253](train_baseline_fixed.py#L243-L253)

**Problem:**
The current implementation tokenizes the prompt separately from the full text:

```python
# Tokenize full text
full_text = prompt + amr + tokenizer.eos_token
encoding = self.tokenizer(full_text, ...)
input_ids = encoding['input_ids'].squeeze()
labels = input_ids.clone()

# Tokenize prompt SEPARATELY
prompt_encoding = self.tokenizer(prompt, ...)
prompt_length = len(prompt_encoding['input_ids'][0])

# Mask using separate tokenization
labels[:prompt_length] = -100  # WRONG!
```

**Why it's broken:**
Tokenizers are **context-dependent**. Tokenizing `"A"` then `"B"` separately can produce different token sequences than tokenizing `"AB"` together:

- `tokenizer("Hello World")` might produce: `[Hello, World]`
- `tokenizer("Hello")` + `tokenizer(" World")` might produce: `[Hello]` + `[ĠWorld]` (where Ġ is a space token)

This means the `prompt_length` calculated from separate tokenization **does NOT correspond** to where the prompt actually ends in the full text.

**Impact:**
- The model is trained on parts of the instruction (should be masked)
- The model is NOT trained on parts of the AMR (should be trained)
- This causes completely broken output structure

**Fix:**
Use `add_special_tokens=False` and encode without context, or find the exact boundary in the combined encoding:

```python
# Correct approach: encode without special tokens
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
amr_ids = tokenizer.encode(amr, add_special_tokens=False)
eos_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

# Build full sequence
full_ids = prompt_ids + amr_ids + eos_ids

# Now we know exactly where prompt ends
prompt_end = len(prompt_ids)
labels = full_ids.copy()
labels[:prompt_end] = -100
```

---

## Bug #2: Parenthesis Balance Check in Prediction (CRITICAL)

**Location:** [predict_baseline_fixed.py:137-146](predict_baseline_fixed.py#L137-L146)

**Problem:**
```python
for line in lines:
    if not found_amr_end:
        amr_lines.append(line)
        if ')' in line:
            # BUG: Counting in ORIGINAL amr string!
            open_count = amr.count('(')  # <-- Should be accumulated amr_lines
            close_count = amr.count(')')
            if open_count == close_count:
                found_amr_end = True
```

**Why it's broken:**
The code checks if parentheses are balanced in the **original full AMR string**, not in the **accumulated lines so far**.

This means it will NEVER detect the end correctly because it's always checking the entire output.

**Impact:**
- The "stop when balanced" logic doesn't work
- Model output includes explanations and garbage
- Contributes to malformed AMRs

**Fix:**
```python
for line in lines:
    if not found_amr_end:
        amr_lines.append(line)
        if ')' in line:
            # Check ACCUMULATED text, not original
            accumulated = '\n'.join(amr_lines)
            open_count = accumulated.count('(')
            close_count = accumulated.count(')')
            if open_count == close_count and open_count > 0:
                found_amr_end = True
```

---

## Bug #3: Overfitting / Training Loss Too Low

**Location:** Training hyperparameters

**Problem:**
Final training loss: **0.0011** (extremely low)

**Why it's broken:**
- Loss this low suggests severe overfitting
- Model memorized training examples instead of learning patterns
- Overfitted models fail to generalize to new inputs

**Impact:**
- Model produces structurally invalid AMRs on test data
- Cannot generalize Penman format rules
- Breaks on unseen sentences

**Fix:**
- Use early checkpoint (e.g., checkpoint-400 instead of checkpoint-1600)
- Reduce training epochs
- Increase learning rate decay
- Add more regularization (weight decay, dropout)

---

## Bug #4: Potential Data Contamination

**Location:** Data loading

**Problem:**
Need to verify that training data doesn't contain malformed AMRs with:
- Unbalanced parentheses
- Duplicate node names
- Incorrect Penman format

**Impact:**
If training data has errors, model learns to produce errors.

**Fix:**
Validate all training examples before training:
```bash
python validate_vietnamese_output.py --file data/train_amr_1.txt
python validate_vietnamese_output.py --file data/train_amr_2.txt
```

---

## Bug #5: Missing Samples (148 vs 150)

**Location:** Prediction script

**Problem:**
Only 148 AMRs generated instead of 150.

**Possible causes:**
- Prediction script crashed silently on 2 sentences
- Loop termination bug
- Encoding issues with certain Vietnamese characters

**Fix:**
Add error handling and logging:
```python
for i, sentence in enumerate(sentences, 1):
    try:
        print(f"[{i}/{len(sentences)}] Processing...")
        amr = predict_amr(...)
        predictions.append(amr)
    except Exception as e:
        print(f"ERROR on sentence {i}: {e}")
        predictions.append("(e / error)")  # Placeholder
```

---

## Comparison: Old vs New Model

| Metric | Old "Buggy" | New "Fixed" | Change |
|--------|-------------|-------------|--------|
| Valid AMRs | 124/150 (82.7%) | 8/150 (5.3%) | **-77.4%** ❌ |
| Invalid AMRs | 26/150 (17.3%) | 137/150 (91.3%) | **+74.0%** ❌ |
| Unmatched parens | 26 | 137 | **+111** ❌ |
| Missing samples | 0 | 2 | **+2** ❌ |
| Training loss | Unknown | 0.0011 | Overfitting ⚠️ |

**Conclusion:** Every metric got WORSE with the "fixes".

---

## Root Cause Analysis

The "fixes" we applied addressed real issues (missing EOS, no instruction masking, unclear prompt), but the **implementation had critical bugs**:

1. ✅ **Intent was correct**: Mask instructions, add EOS, clarify prompt
2. ❌ **Implementation was broken**: Tokenization mismatch breaks masking
3. ❌ **Validation was insufficient**: Didn't test training examples
4. ❌ **Overfitting not detected**: Loss 0.0011 should have raised red flags

---

## Recommended Action Plan

### Option A: Fix and Retrain (Recommended)
1. Fix instruction masking bug (Bug #1)
2. Fix prediction balance check (Bug #2)
3. Validate training data (Bug #4)
4. Retrain with proper monitoring
5. Use early checkpoint to avoid overfitting (Bug #3)

### Option B: Use Old Model
The old "buggy" model (82.7% valid) is **significantly better** than the new one (5.3% valid). Consider using it until fixes are validated.

### Option C: Investigate Before Retraining
1. Run diagnosis script on server
2. Validate training data quality
3. Test fix with small sample (10 examples)
4. Confirm instruction masking works correctly
5. Then retrain

---

## Files to Update

1. `train_baseline_fixed.py` - Fix Bug #1 (instruction masking)
2. `predict_baseline_fixed.py` - Fix Bug #2 (balance check), Bug #5 (error handling)
3. `config_fixed.py` - Adjust hyperparameters to reduce overfitting
4. `diagnose_tokenization.py` - Run on server to confirm Bug #1

---

## Status

**Current state:** Model is completely broken and unusable.

**Priority:** CRITICAL - Need immediate fix or rollback to old model.

**Next step:** User decision on Option A, B, or C.
