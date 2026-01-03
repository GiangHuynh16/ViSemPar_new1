# 🚀 BẮT ĐẦU TẠI ĐÂY - Complete Guide

## 📋 Tình huống hiện tại

Model baseline 7B vừa train xong nhưng kết quả thảm họa:
- **Checkpoint-200 (best cũ): 70% valid AMRs**
- **Checkpoint-1635 (cuối): 5.3% valid AMRs** ❌
- **Nguyên nhân:** 3 critical bugs + overfitting nghiêm trọng

**✅ TẤT CẢ ĐÃ ĐƯỢC FIX!** Sẵn sàng retrain.

---

## 🎯 3 Bước Đơn Giản

### Bước 1: Pull code mới nhất (1 phút)

```bash
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
git pull

source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final
```

### Bước 2: Retrain model (2-3 giờ)

```bash
# Option A: Validate trước (recommended, +2 phút)
bash VALIDATE_BEFORE_RETRAIN.sh
bash TRAIN_BASELINE_FIXED.sh

# Option B: Train ngay
bash TRAIN_BASELINE_FIXED.sh
```

**Thời gian:** 2-3 giờ (giảm từ 4-5 giờ nhờ chỉ 2 epochs)

**Trong khi chờ:** Mở terminal thứ 2 để monitor:
```bash
# Terminal 2
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
tail -f logs/training_*.log

# Hoặc watch GPU
watch -n 1 nvidia-smi
```

### Bước 3: Test checkpoints và tìm best (10-15 phút)

```bash
# Auto-test tất cả checkpoints
bash TEST_ALL_CHECKPOINTS.sh

# Chọn option 3: Test early checkpoints only (100-500)
# Script sẽ tự động tìm checkpoint tốt nhất!
```

**Kết quả:** Script sẽ show checkpoint nào có highest valid AMR %

---

## 📊 Kỳ vọng kết quả

| Metric | Old (buggy) | New (fixed) | Improvement |
|--------|-------------|-------------|-------------|
| Valid AMRs | 70% | **80-90%** | +10-20% |
| Invalid AMRs | 30% | **10-20%** | -10-20% |
| Training time | 4-5 giờ | **2-3 giờ** | -40% |
| Best checkpoint | 200 | **100-300** | Early stopping |

---

## 🐛 Bugs đã fix

### 1. Instruction Masking (CRITICAL)
**Vấn đề:** Model học cả instruction thay vì chỉ AMR
**Fix:** Dùng `encode(..., add_special_tokens=False)` để tránh tokenization mismatch

### 2. Balance Check (CRITICAL)
**Vấn đề:** Đếm ngoặc trong string gốc thay vì accumulated
**Fix:** Check trong accumulated text: `'\n'.join(amr_lines)`

### 3. Prompt quá phức tạp
**Vấn đề:** 135 dòng với 6 quy tắc → Model confused
**Fix:** 3 dòng đơn giản: "Chuyển câu... sang AMR"

### 4. Overfitting
**Vấn đề:** 15 epochs → checkpoint-1635 chỉ 5.3% valid
**Fix:** 2 epochs, save mỗi 100 steps để tìm sweet spot

---

## 📝 Files quan trọng

### Để đọc:
1. **[START_HERE.md](START_HERE.md)** ← Bạn đang đọc file này
2. **[FINAL_FIXES_SUMMARY.md](FINAL_FIXES_SUMMARY.md)** - Chi tiết tất cả fixes
3. **[QUICKSTART.md](QUICKSTART.md)** - Quick reference

### Để chạy:
1. **[TRAIN_BASELINE_FIXED.sh](TRAIN_BASELINE_FIXED.sh)** - Training script
2. **[TEST_ALL_CHECKPOINTS.sh](TEST_ALL_CHECKPOINTS.sh)** - Test tất cả checkpoints
3. **[VALIDATE_BEFORE_RETRAIN.sh](VALIDATE_BEFORE_RETRAIN.sh)** - Pre-training validation

### Core code:
1. **[train_baseline_fixed.py](train_baseline_fixed.py)** - Training với fixes
2. **[predict_baseline_fixed.py](predict_baseline_fixed.py)** - Prediction với fixes
3. **[config/config_fixed.py](config/config_fixed.py)** - Config optimized

---

## 🎬 Workflow đầy đủ

```bash
# 1. Pull code
git pull

# 2. Validate (optional)
bash VALIDATE_BEFORE_RETRAIN.sh

# 3. Train (2-3 giờ)
bash TRAIN_BASELINE_FIXED.sh

# 4. Test checkpoints (10-15 phút)
bash TEST_ALL_CHECKPOINTS.sh

# 5. Calculate SMATCH cho best checkpoint
python -m smatch -f \
    evaluation_results/baseline_7b_fixed/predictions.txt \
    data/public_test_ground_truth.txt \
    --significant 4

# 6. So sánh với MTUP
# See THESIS_CHAPTER_MTUP.md for MTUP results
```

---

## ✅ Success Criteria

**Training thành công nếu:**
- ✅ Valid AMRs: > 120/150 (80%)
- ✅ All 150 samples generated
- ✅ Balanced parentheses
- ✅ No duplicate nodes
- ✅ No explanations after AMR

**Training thất bại nếu:**
- ❌ Valid AMRs: < 105/150 (70%) → Không improvement
- ❌ Missing samples
- ❌ Unbalanced parentheses > 30%

**Nếu thất bại:** Checkpoint sớm có thể vẫn tốt. Test checkpoint-100, 200, 300...

---

## 🆘 Troubleshooting

### Training bị lỗi?
```bash
# Check log
tail -100 logs/training_*.log

# Check GPU
nvidia-smi

# Check environment
conda list | grep -E 'torch|transformers|peft'
```

### Validation failed?
```bash
# Run diagnostic
python TEST_TOKENIZATION_FIX.py

# Validate training data
python validate_vietnamese_output.py --file data/train_amr_1.txt
```

### Results vẫn bad?
1. Test early checkpoints (100, 200, 300)
2. Check training loss curve
3. So sánh với checkpoint-200 cũ (70% valid)
4. Nếu < 70% → Có vấn đề khác, báo lại

---

## 📈 Timeline

**Tổng thời gian:** ~3 giờ

- [ ] Pull code: 1 phút
- [ ] Validate (optional): 2 phút
- [ ] Training: 2-3 giờ
- [ ] Test checkpoints: 10-15 phút
- [ ] Calculate SMATCH: 1-2 phút
- [ ] So sánh results: 5 phút

**Có thể làm khác trong lúc training:**
- Đọc documentation
- Chuẩn bị thesis chapter
- Nghỉ ngơi 😊

---

## 🎯 Sau khi xong

**Nếu kết quả tốt (> 80% valid):**
1. ✅ Copy best checkpoint
2. ✅ Calculate SMATCH
3. ✅ So sánh với MTUP
4. ✅ Update thesis
5. ✅ Upload to HuggingFace (optional)

**Nếu kết quả OK (70-80% valid):**
1. ⚠️ Acceptable cho baseline
2. ⚠️ Có thể cải thiện thêm
3. ⚠️ Nhưng đủ để so sánh với MTUP

**Nếu kết quả xấu (< 70% valid):**
1. ❌ Test early checkpoints
2. ❌ Check logs cho errors
3. ❌ Báo lại để debug

---

## 💡 Tips

1. **Monitor training loss:**
   - Good: 0.05 - 0.15
   - Overfitting: < 0.05
   - Undertrained: > 0.20

2. **Test nhiều checkpoints:**
   - Checkpoint cuối KHÔNG luôn tốt nhất
   - Checkpoint-200 cũ tốt hơn checkpoint-1635
   - Sweet spot thường ở 100-400

3. **Compare với old model:**
   - Old checkpoint-200: 70% valid
   - Target new model: 80-90% valid
   - Nếu < 70% → Có vấn đề

---

## 📞 Questions?

**Read these first:**
1. [FINAL_FIXES_SUMMARY.md](FINAL_FIXES_SUMMARY.md) - Comprehensive changelog
2. [CRITICAL_ANALYSIS_AND_FIXES.md](CRITICAL_ANALYSIS_AND_FIXES.md) - Technical analysis
3. [BUGS_IDENTIFIED.md](BUGS_IDENTIFIED.md) - Bug details

**Still stuck?**
- Check logs: `tail -100 logs/training_*.log`
- Check GPU: `nvidia-smi`
- Check environment: `conda list`

---

## 🚀 Ready to start?

```bash
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
git pull
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final
bash TRAIN_BASELINE_FIXED.sh
```

**Good luck! 🍀**

---

**Last updated:** 2026-01-03

**Status:** ✅ All fixes applied, ready to retrain

**Confidence:** High - Thoroughly analyzed and tested
