# 🔧 Tóm tắt: Các lỗi training và cách fix

## 🚨 3 lỗi nghiêm trọng phát hiện

### 1. ❌ Thiếu EOS Token

**Vấn đề:**
```python
# Code cũ (SAI):
text = PROMPT_TEMPLATE.format(sentence=sentence) + amr
```

Model không biết khi nào dừng → Sinh thêm text sau AMR

**Evidence từ output:**
- Model sinh ra giải thích sau AMR
- Câu văn tiếp tục sau khi AMR đã kết thúc
- Không có điểm dừng rõ ràng

**Fix:**
```python
# Code mới (ĐÚNG):
full_text = prompt + amr + tokenizer.eos_token
```

---

### 2. ❌ Không có Instruction Masking

**Vấn đề:**
```python
# Code cũ (SAI):
labels = input_ids.clone()
labels[labels == tokenizer.pad_token_id] = -100
```

Model học cả prompt "Bạn là chuyên gia..." → Lãng phí!

**Tại sao sai:**
- Model phải học cả instruction (không cần thiết)
- Chậm hội tụ
- Loss cao ở phần không cần học

**Fix:**
```python
# Code mới (ĐÚNG):
# 1. Tách prompt và AMR
prompt = example['prompt']
amr = example['amr']

# 2. Tính prompt length
prompt_encoding = tokenizer(prompt, ...)
prompt_length = len(prompt_encoding['input_ids'][0])

# 3. Mask instruction part
labels = input_ids.clone()
labels[:prompt_length] = -100  # ← CRITICAL FIX
labels[labels == tokenizer.pad_token_id] = -100
```

**Giải thích:**
- `-100` = ignore trong loss calculation
- Chỉ train trên AMR output (phần sau prompt)
- Model học nhanh hơn, hiệu quả hơn

---

### 3. ❌ Prompt không rõ ràng về Penman format

**Prompt cũ (THIẾU):**
```
Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt.
Hãy chuyển đổi câu sau sang định dạng AMR.

Quy tắc:
- Sử dụng khái niệm tiếng Việt có dấu gạch dưới
- Gán biến cho mỗi khái niệm
- ... (mơ hồ)

Câu: {sentence}
AMR:
```

**Vấn đề:**
- Không nói rõ "định dạng Penman"
- Không cấm giải thích
- Không nhấn mạnh cấu trúc cây

**Prompt mới (RÕ RÀNG):**
```
Bạn là chuyên gia ngôn ngữ học máy tính, chuyên về phân tích ngữ nghĩa tiếng Việt.
Hãy chuyển đổi câu văn sau sang định dạng AMR theo đúng **chuẩn Penman**.

Các quy tắc bắt buộc:
1. Sử dụng định dạng Penman: (biến / khái niệm :quan-hệ (biến2 / khái niệm2))
2. Khái niệm tiếng Việt đa âm tiết phải dùng dấu gạch dưới (ví dụ: c / chính_phủ)
3. Sử dụng các quan hệ chuẩn: :ARG0, :ARG1, :ARG2, :time, :location, ...
4. Đảm bảo cấu trúc cây với ngoặc đơn hoàn toàn cân bằng
5. Mỗi khái niệm chỉ được gán một biến duy nhất
6. KHÔNG thêm giải thích, chỉ trả về cấu trúc AMR thuần túy  ← CRITICAL

Câu tiếng Việt: {sentence}

AMR (Penman):
```

**Điểm khác biệt:**
- ✅ Nói rõ "chuẩn Penman"
- ✅ Ví dụ cụ thể format
- ✅ Cấm giải thích (quy tắc #6)
- ✅ Nhấn mạnh balanced parentheses

---

## 📊 Kết quả của các lỗi này

### Model cũ (có 3 lỗi):
- ❌ 26/150 AMR không hợp lệ (17.3% lỗi)
- ❌ Unmatched parentheses
- ❌ Duplicate node names
- ❌ Sinh giải thích sau AMR
- ❌ Không thể tính SMATCH

### Model mới (sau fix):
- ✅ Dự kiến 150/150 AMR hợp lệ
- ✅ Balanced parentheses
- ✅ Không duplicate nodes
- ✅ Dừng đúng tại EOS
- ✅ Tính được SMATCH

---

## 🔧 Chi tiết implementation

### File cần sửa:

1. **config/config_fixed.py** ← Prompt mới
2. **train_baseline_fixed.py** ← Training script mới với masking
3. **predict_baseline_fixed.py** ← Prediction script

### Dataset class mới:

```python
class BaselineDatasetFixed(Dataset):
    def __getitem__(self, idx):
        example = self.examples[idx]

        prompt = example['prompt']
        amr = example['amr']

        # FIX 1: Add EOS token
        full_text = prompt + amr + self.tokenizer.eos_token

        # Tokenize
        encoding = self.tokenizer(full_text, ...)
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()

        # FIX 2: Instruction masking
        prompt_encoding = self.tokenizer(prompt, ...)
        prompt_length = len(prompt_encoding['input_ids'][0])
        labels[:prompt_length] = -100  # Mask instruction

        # Mask padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
```

### Generation config:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.9,
    repetition_penalty=1.15,
    eos_token_id=tokenizer.eos_token_id,  # ← CRITICAL
    pad_token_id=tokenizer.pad_token_id,
)

# Remove EOS and explanations
generated = tokenizer.decode(outputs[0])
amr = generated[len(prompt):].strip()
if tokenizer.eos_token in amr:
    amr = amr.split(tokenizer.eos_token)[0].strip()
```

---

## 📋 Checklist để verify fixes

Sau khi train xong, check:

- [ ] **EOS Token:**
  - [ ] Training data có thêm `tokenizer.eos_token` chưa?
  - [ ] Generation config có `eos_token_id` chưa?
  - [ ] Output có dừng đúng không?

- [ ] **Instruction Masking:**
  - [ ] Labels có mask prompt (`labels[:prompt_length] = -100`)?
  - [ ] Training loss chỉ tính trên AMR?
  - [ ] Check sample batch xem labels đúng chưa

- [ ] **Prompt:**
  - [ ] Có từ "Penman" trong prompt?
  - [ ] Có cấm giải thích (quy tắc #6)?
  - [ ] Có ví dụ cụ thể format?

- [ ] **Output Quality:**
  - [ ] Tất cả AMR có balanced parentheses?
  - [ ] Không có duplicate nodes?
  - [ ] Không có giải thích sau AMR?
  - [ ] SMATCH tính được?

---

## 🎯 Commands để chạy

```bash
# 1. Cleanup
chmod +x CLEANUP_AND_ORGANIZE.sh
./CLEANUP_AND_ORGANIZE.sh

# 2. Train với fixes
conda activate baseline_final
python train_baseline_fixed.py --epochs 15 --show-sample

# 3. Archive model
TIMESTAMP=$(ls -t outputs/ | grep baseline_fixed | head -1)
mv outputs/$TIMESTAMP/final models_archive/baseline_7b_fixed/

# 4. Test
python predict_baseline_fixed.py \
  --model models_archive/baseline_7b_fixed/final \
  --test-file data/public_test.txt \
  --output evaluation_results/baseline_7b_fixed/predictions.txt

# 5. Evaluate
python -m smatch -f \
  evaluation_results/baseline_7b_fixed/predictions.txt \
  data/public_test_ground_truth.txt \
  --significant 4
```

---

## 💡 Tại sao những lỗi này quan trọng?

### 1. EOS Token:
- Không có → Model không biết dừng
- Có → Model dừng đúng lúc
- **Impact:** Từ "sinh vô tận" → "sinh đúng độ dài"

### 2. Instruction Masking:
- Không có → Model học cả prompt
- Có → Model chỉ học AMR
- **Impact:** Từ "chậm, lãng phí" → "nhanh, hiệu quả"

### 3. Clear Prompt:
- Không rõ → Model đoán format
- Rõ ràng → Model biết chính xác
- **Impact:** Từ "17% lỗi" → "0% lỗi" (dự kiến)

---

## 📚 References

### Files created:
1. `config/config_fixed.py` - Fixed config with new prompt
2. `train_baseline_fixed.py` - Fixed training script
3. `predict_baseline_fixed.py` - Fixed prediction script
4. `CLEANUP_AND_ORGANIZE.sh` - Cleanup script
5. `RETRAIN_INSTRUCTIONS.md` - Detailed instructions
6. `TRAINING_FIXES_SUMMARY.md` - This file

### Next steps:
1. Read `RETRAIN_INSTRUCTIONS.md`
2. Run cleanup
3. Train with `train_baseline_fixed.py`
4. Evaluate and compare

---

**Author:** Claude Code
**Date:** 2026-01-03
**Version:** 1.0
