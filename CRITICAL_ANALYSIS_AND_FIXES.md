# 🚨 PHÂN TÍCH VẤN ĐỀ VÀ GIẢI PHÁP

## Tóm tắt

Model "fixed" vừa train ra kết quả **THẢM HỌA**:
- **Old model (buggy):** 124/150 valid AMRs (82.7%)
- **New model (fixed):** 8/150 valid AMRs (5.3%)
- **Regression:** -77.4% ❌

Nguyên nhân: **3 bugs nghiêm trọng** trong code training và prediction.

---

## 🐛 Bug #1: Instruction Masking HOÀN TOÀN SAI (CRITICAL)

### Vấn đề

File: [train_baseline_fixed.py:243-253](train_baseline_fixed.py#L243-L253)

Code cũ (SAI):
```python
# Tokenize full text
full_text = prompt + amr + tokenizer.eos_token
encoding = self.tokenizer(full_text, ...)
input_ids = encoding['input_ids'].squeeze()
labels = input_ids.clone()

# Tokenize prompt RIÊNG BIỆT
prompt_encoding = self.tokenizer(prompt, ...)
prompt_length = len(prompt_encoding['input_ids'][0])

# Mask dùng độ dài từ tokenization riêng biệt
labels[:prompt_length] = -100  # SAI HOÀN TOÀN!
```

### Tại sao sai?

Tokenizer **phụ thuộc context**!

Ví dụ:
- Tokenize `"Hello World"` → `[Hello, World]`
- Tokenize `"Hello"` + `"World"` riêng → `[Hello]` + `[World]` (khác nhau!)

Do đó `prompt_length` tính từ tokenization riêng **KHÔNG phải** là vị trí kết thúc prompt trong full text!

### Hậu quả

- Model học cả instruction (đáng lẽ phải mask)
- Model KHÔNG học một phần AMR (đáng lẽ phải train)
- Output hoàn toàn broken

### ✅ Fix đã apply

File: [train_baseline_fixed.py:227-270](train_baseline_fixed.py#L227-L270)

```python
# Encode từng phần KHÔNG có special tokens để tránh mismatch
prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
amr_ids = self.tokenizer.encode(amr, add_special_tokens=False)
eos_ids = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)

# Ghép lại
full_ids = prompt_ids + amr_ids + eos_ids

# Bây giờ biết CHÍNH XÁC prompt kết thúc ở đâu
prompt_end = len(prompt_ids)
labels = input_ids.copy()
labels[:prompt_end] = -100  # ĐÚNG!
```

---

## 🐛 Bug #2: Check Parenthesis Balance SAI (CRITICAL)

### Vấn đề

File: [predict_baseline_fixed.py:137-146](predict_baseline_fixed.py#L137-L146) (đã fix)

Code cũ (SAI):
```python
for line in lines:
    if not found_amr_end:
        amr_lines.append(line)
        if ')' in line:
            # SAI: Đếm trong string GỐC, không phải accumulated!
            open_count = amr.count('(')  # <-- SAI!
            close_count = amr.count(')')
            if open_count == close_count:
                found_amr_end = True
```

### Tại sao sai?

Code check balance trong **toàn bộ AMR gốc**, không phải trong **phần đã tích lũy**.

→ Logic "dừng khi balanced" **KHÔNG BAO GIỜ HOẠT ĐỘNG**

### Hậu quả

- Model output bao gồm explanation và garbage
- AMRs bị malformed
- 91.3% invalid rate

### ✅ Fix đã apply

```python
for line in lines:
    if not found_amr_end:
        amr_lines.append(line)
        if ')' in line:
            # ĐÚNG: Check trong accumulated text
            accumulated = '\n'.join(amr_lines)
            open_count = accumulated.count('(')
            close_count = accumulated.count(')')
            if open_count == close_count and open_count > 0:
                found_amr_end = True
```

---

## 🐛 Bug #3: Overfitting - Loss Quá Thấp

### Vấn đề

Training loss cuối: **0.0011** (cực kỳ thấp!)

### Tại sao là vấn đề?

Loss thấp đến vậy = **overfitting nghiêm trọng**:
- Model **thuộc lòng** training examples
- Model KHÔNG học được pattern
- Fail hoàn toàn trên test data

### Hậu quả

- 91.3% invalid AMRs
- Không generalize được Penman format rules
- Bị broken trên unseen sentences

### 💡 Giải pháp

1. **Sử dụng early checkpoint** thay vì checkpoint cuối:
   - Checkpoint-400 thay vì checkpoint-1600
   - Loss cao hơn = generalize tốt hơn

2. **Hoặc retrain với:**
   - Fewer epochs (hiện tại: 3 epochs × 545 steps = 1635 total)
   - Higher learning rate decay
   - More weight decay/dropout

---

## 📊 So sánh Old vs New

| Metric | Old "Buggy" | New "Fixed" (Bug) | New "Fixed" (Real Fix) |
|--------|-------------|-------------------|------------------------|
| Valid AMRs | 124/150 (82.7%) | 8/150 (5.3%) | **Chưa test** |
| Invalid AMRs | 26/150 (17.3%) | 137/150 (91.3%) | **Chưa test** |
| Unmatched parens | 26 | 137 | **Chưa test** |
| Missing samples | 0 | 2 | **Fixed với error handling** |
| Training loss | Unknown | 0.0011 (overfitted) | **Chưa train** |

---

## 🎯 Kế hoạch hành động

### Option A: Retrain với fixes mới (KHUYẾN NGHỊ)

1. ✅ **Fixed Bug #1** - Instruction masking corrected
2. ✅ **Fixed Bug #2** - Balance check corrected
3. ✅ **Fixed Bug #5** - Added error handling
4. ⏳ **Validate training data** - Đảm bảo không có lỗi trong data
5. ⏳ **Retrain model** - Với code đã fix
6. ⏳ **Test early checkpoint** - Tránh overfitting

**Ước tính thời gian:** ~4-5 giờ training

**Rủi ro:** Có thể vẫn cần điều chỉnh hyperparameters

### Option B: Dùng model cũ tạm thời

Model cũ (82.7% valid) **TỐT HƠN NHIỀU** so với model mới (5.3% valid).

→ Có thể dùng tạm trong khi fix và retrain.

### Option C: Validate trước khi retrain

1. Chạy diagnostic script trên server
2. Validate training data quality
3. Test fix với sample nhỏ (10 examples)
4. Confirm instruction masking works
5. Sau đó mới retrain full

**Ước tính thời gian:** ~1-2 giờ validation + 4-5 giờ training

**Ưu điểm:** An toàn hơn, đảm bảo fix đúng trước khi train

---

## 📝 Files đã update

### 1. train_baseline_fixed.py
- ✅ Fixed instruction masking (Bug #1)
- ✅ Sử dụng `encode(..., add_special_tokens=False)`
- ✅ Tính chính xác vị trí kết thúc prompt

### 2. predict_baseline_fixed.py
- ✅ Fixed balance check (Bug #2)
- ✅ Check accumulated text, không phải original
- ✅ Added error handling cho missing samples (Bug #5)

### 3. diagnose_tokenization.py (NEW)
- ✅ Script để test tokenization mismatch
- ⏳ Cần chạy trên server để confirm fix

### 4. BUGS_IDENTIFIED.md (NEW)
- ✅ Chi tiết kỹ thuật về các bugs
- ✅ Phân tích root cause
- ✅ So sánh old vs new

---

## 🚀 Next Steps

### Ngay lập tức

```bash
# 1. Push code lên server
git add train_baseline_fixed.py predict_baseline_fixed.py diagnose_tokenization.py BUGS_IDENTIFIED.md CRITICAL_ANALYSIS_AND_FIXES.md
git commit -m "Fix critical bugs in instruction masking and prediction"
git push

# 2. SSH vào server
ssh islabworker2@islab-server2

# 3. Pull updates
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
git pull

# 4. Validate training data
conda activate baseline_final
python validate_vietnamese_output.py --file data/train_amr_1.txt
python validate_vietnamese_output.py --file data/train_amr_2.txt

# 5. Test tokenization fix
python diagnose_tokenization.py

# 6. Nếu validate pass → Retrain
bash TRAIN_BASELINE_FIXED.sh
```

### Trong quá trình training

**Giám sát:**
- Check training loss không xuống quá thấp (<0.01)
- Nếu loss < 0.05, consider stopping early
- Save checkpoints mỗi 200 steps

**Dự phòng:**
- Nếu model vẫn bad, thử checkpoint-200, checkpoint-400
- So sánh validation predictions với ground truth
- Có thể cần điều chỉnh learning rate hoặc epochs

---

## 💡 Bài học rút ra

1. **Tokenization phụ thuộc context** - Không thể tokenize riêng rồi tính độ dài
2. **Validation quan trọng** - Phải test thoroughly trước khi train
3. **Loss thấp ≠ model tốt** - Overfitting nguy hiểm
4. **Error handling cần thiết** - Phải catch errors để debug

---

## ❓ Questions?

Bạn muốn:

**A.** Retrain ngay với fixes mới (4-5 giờ)

**B.** Validate kỹ trước (1-2 giờ) rồi mới train (4-5 giờ)

**C.** Dùng model cũ tạm, research thêm trước khi retrain

**D.** Khác (đề xuất của bạn)

---

## 📌 Status

**Bugs identified:** 5/5 ✅

**Bugs fixed:** 3/5 ✅
- ✅ Bug #1 (instruction masking)
- ✅ Bug #2 (balance check)
- ⏳ Bug #3 (overfitting) - cần test checkpoint
- ⏳ Bug #4 (data quality) - cần validate
- ✅ Bug #5 (missing samples)

**Code updated:** ✅

**Ready to retrain:** ⏳ (pending validation)

**Current model:** ❌ BROKEN (5.3% valid)

**Old model:** ⚠️ Available fallback (82.7% valid)
