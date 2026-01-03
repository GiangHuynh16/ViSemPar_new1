# 🚀 Quick Start - Retrain với Bugs Fixed

## ⚡ TL;DR

Model hiện tại: **5.3% valid AMRs** ❌
Nguyên nhân: **3 bugs đã được fix** ✅
Hành động: **Retrain ngay!**

---

## 📋 Bước 1: Đọc phân tích (2 phút)

Đọc: **[CRITICAL_ANALYSIS_AND_FIXES.md](CRITICAL_ANALYSIS_AND_FIXES.md)**

**TL;DR của bugs:**
1. Instruction masking sai → Model học instruction thay vì AMR
2. Parenthesis check sai → Output có garbage
3. Overfitting (loss 0.0011) → Không generalize

**Tất cả đã được fix!** ✅

---

## 🔧 Bước 2: Chọn phương án

### Option A: An toàn - Validate trước (KHUYẾN NGHỊ)

```bash
# SSH vào server
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Pull fixes
git pull

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final

# Validate (1-2 phút)
bash VALIDATE_BEFORE_RETRAIN.sh

# Nếu PASS → Retrain
bash TRAIN_BASELINE_FIXED.sh
```

**Thời gian:** 2 phút validate + 4-5 giờ training

### Option B: Nhanh - Retrain ngay

```bash
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
git pull
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final
bash TRAIN_BASELINE_FIXED.sh
```

**Thời gian:** 4-5 giờ training
**Rủi ro:** Cao hơn nếu còn issues chưa phát hiện

### Option C: Test checkpoint cũ trước

Model hiện tại có thể overfitting ở cuối. Thử checkpoint sớm hơn:

```bash
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final

# Find available checkpoints
LATEST_MODEL=$(ls -t outputs/ | grep baseline_fixed | head -1)
echo "Available checkpoints:"
ls -la "outputs/$LATEST_MODEL/"

# Test checkpoint-200, 400, 600, 800...
python predict_baseline_fixed.py \
    --model "outputs/$LATEST_MODEL/checkpoint-400" \
    --test-file data/public_test.txt \
    --output evaluation_results/test_checkpoint_400.txt

# Validate
python validate_vietnamese_output.py \
    --file evaluation_results/test_checkpoint_400.txt
```

**Nếu checkpoint sớm hơn tốt → Dùng luôn, không cần retrain!**

---

## 📊 Bước 3: Theo dõi training (nếu retrain)

### Mở terminal thứ 2 để monitor:

```bash
# Terminal 2
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Watch training log
tail -f logs/training_*.log

# Hoặc watch GPU
watch -n 1 nvidia-smi
```

### Kiểm tra loss:

**Loss tốt:** 0.05 - 0.15
**Loss overfitting:** < 0.05 → Dừng sớm, dùng checkpoint trước
**Loss undertrained:** > 0.20 → Train thêm

### Save checkpoints quan trọng:

Training sẽ save mỗi 200 steps. Sau khi xong, test:
- checkpoint-200
- checkpoint-400
- checkpoint-600
- checkpoint-800

**Checkpoint cuối KHÔNG phải lúc nào cũng tốt nhất!**

---

## ✅ Bước 4: Test sau training

```bash
# Test checkpoint-400 (ví dụ)
bash TEST_FIXED_MODEL.sh

# Hoặc test specific checkpoint
python predict_baseline_fixed.py \
    --model outputs/baseline_fixed_YYYYMMDD_HHMMSS/checkpoint-400 \
    --test-file data/public_test.txt \
    --output evaluation_results/baseline_7b_fixed/predictions.txt

# Validate
python validate_vietnamese_output.py \
    --file evaluation_results/baseline_7b_fixed/predictions.txt
```

**Kỳ vọng:**
- Valid AMRs: **> 120/150 (80%)**
- Invalid AMRs: **< 30/150 (20%)**
- All 150 samples generated
- Balanced parentheses
- No duplicate nodes

---

## 🎯 Tiêu chí thành công

### ✅ Training thành công nếu:
- Valid AMRs: > 75% (target: 80-90%)
- Proper Penman format
- Balanced parentheses
- No duplicate nodes
- All 150 samples generated

### ❌ Training thất bại nếu:
- Valid AMRs: < 70%
- Loss < 0.01 (overfitting)
- Missing samples
- Unbalanced parentheses > 20%

**Nếu thất bại:** Test checkpoint sớm hơn hoặc điều chỉnh hyperparameters

---

## 📞 Troubleshooting

### Training bị lỗi?

```bash
# Check log
tail -100 logs/training_*.log

# Check GPU memory
nvidia-smi

# Check conda env
conda list | grep torch
conda list | grep transformers
conda list | grep peft
```

### Validation failed?

```bash
# Run diagnostic
python TEST_TOKENIZATION_FIX.py

# Check training data
python validate_vietnamese_output.py --file data/train_amr_1.txt
```

### Model vẫn bad?

1. Test early checkpoints (200, 400, 600)
2. Check training loss curve
3. Kiểm tra có overfitting không
4. Xem [CRITICAL_ANALYSIS_AND_FIXES.md](CRITICAL_ANALYSIS_AND_FIXES.md) để hiểu bugs

---

## 📚 Chi tiết đầy đủ

- **[CRITICAL_ANALYSIS_AND_FIXES.md](CRITICAL_ANALYSIS_AND_FIXES.md)** - Phân tích bugs (Tiếng Việt)
- **[BUGS_IDENTIFIED.md](BUGS_IDENTIFIED.md)** - Technical details
- **[README_FIXES.md](README_FIXES.md)** - Full documentation

---

## ⏱️ Timeline

**Nếu chọn Option A (khuyến nghị):**
- [ ] Đọc phân tích: 2 phút
- [ ] Pull và validate: 2 phút
- [ ] Retrain: 4-5 giờ
- [ ] Test checkpoints: 10-15 phút
- [ ] **Tổng: ~5 giờ**

**Nếu chọn Option C (test checkpoint cũ):**
- [ ] Test checkpoint-200: 2 phút
- [ ] Test checkpoint-400: 2 phút
- [ ] Test checkpoint-600: 2 phút
- [ ] Nếu có checkpoint tốt → XONG!
- [ ] **Tổng: ~10 phút (có thể không cần retrain!)**

---

## 🎬 Bắt đầu ngay

**Khuyến nghị: Thử Option C trước (10 phút)**

Nếu checkpoint cũ không tốt → Chuyển sang Option A (retrain với fixes)

**Lệnh:**
```bash
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
git pull
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final

# Test checkpoint hiện tại trước
bash TEST_FIXED_MODEL.sh

# Nếu bad → Validate và retrain
bash VALIDATE_BEFORE_RETRAIN.sh
bash TRAIN_BASELINE_FIXED.sh
```

**Chúc may mắn! 🚀**
