# 📚 Index - Retrain Baseline 7B Documentation

## 🎯 Bắt đầu từ đây

### Option 1: Automated (Recommended)
```bash
bash QUICK_START_RETRAIN.sh
```
**Script này sẽ:**
- ✅ Cleanup files cũ tự động
- ✅ Train model với tất cả fixes
- ✅ Archive model
- ✅ Test và evaluate

### Option 2: Manual Step-by-Step
Đọc [RETRAIN_INSTRUCTIONS.md](RETRAIN_INSTRUCTIONS.md) để làm từng bước.

---

## 📄 Các file documentation

### 1. **TRAINING_FIXES_SUMMARY.md** ⭐ ĐỌC ĐẦU TIÊN
**Mục đích:** Hiểu rõ 3 lỗi nghiêm trọng và cách fix

**Nội dung:**
- ❌ Lỗi 1: Thiếu EOS token
- ❌ Lỗi 2: Không có instruction masking
- ❌ Lỗi 3: Prompt không rõ Penman format
- ✅ Code fix chi tiết cho từng lỗi
- 📊 So sánh kết quả old vs new

**Ai nên đọc:** Tất cả mọi người trước khi retrain

**Đọc file:** [TRAINING_FIXES_SUMMARY.md](TRAINING_FIXES_SUMMARY.md)

---

### 2. **RETRAIN_INSTRUCTIONS.md** 📋 HƯỚNG DẪN CHI TIẾT
**Mục đích:** Step-by-step guide để retrain

**Nội dung:**
- Bước 1: Cleanup files cũ
- Bước 2: Activate conda
- Bước 3: Train với fixes
- Bước 4: Archive model
- Bước 5: Test và evaluate
- Bước 6: Compare old vs new
- 🚨 Troubleshooting

**Ai nên đọc:** Người sẽ thực hiện retrain

**Đọc file:** [RETRAIN_INSTRUCTIONS.md](RETRAIN_INSTRUCTIONS.md)

---

### 3. **QUICK_START_RETRAIN.sh** 🚀 AUTOMATED SCRIPT
**Mục đích:** Tự động hóa toàn bộ quy trình

**Chức năng:**
- ✅ Cleanup tự động
- ✅ Train với fixes
- ✅ Archive model
- ✅ Test tự động
- ✅ Summary kết quả

**Cách dùng:**
```bash
chmod +x QUICK_START_RETRAIN.sh
bash QUICK_START_RETRAIN.sh
```

**Ai nên dùng:** Người muốn chạy nhanh, ít tương tác

**File:** [QUICK_START_RETRAIN.sh](QUICK_START_RETRAIN.sh)

---

### 4. **CLEANUP_AND_ORGANIZE.sh** 🧹 CLEANUP SCRIPT
**Mục đích:** Dọn dẹp và tổ chức files

**Chức năng:**
- Archive model cũ (buggy) → `models_archive/baseline_7b_old/`
- Archive results cũ → `evaluation_results/baseline_7b_old/`
- Tạo README cho từng directory
- Show disk usage
- Hướng dẫn next steps

**Cách dùng:**
```bash
chmod +x CLEANUP_AND_ORGANIZE.sh
./CLEANUP_AND_ORGANIZE.sh
```

**Ai nên dùng:** Chạy một lần trước khi retrain

**File:** [CLEANUP_AND_ORGANIZE.sh](CLEANUP_AND_ORGANIZE.sh)

---

## 🔧 Các file code mới (FIXED)

### 1. **config/config_fixed.py**
**Thay đổi:**
- ✅ Prompt mới với chuẩn Penman rõ ràng
- ✅ Thêm quy tắc "KHÔNG thêm giải thích"
- ✅ Ví dụ cụ thể format

**Import:**
```python
from config_fixed import PROMPT_TEMPLATE, TRAINING_CONFIG
```

### 2. **train_baseline_fixed.py**
**Thay đổi:**
- ✅ Add EOS token: `full_text = prompt + amr + tokenizer.eos_token`
- ✅ Instruction masking: `labels[:prompt_length] = -100`
- ✅ Dataset class mới: `BaselineDatasetFixed`

**Chạy:**
```bash
python train_baseline_fixed.py --epochs 15 --show-sample
```

### 3. **predict_baseline_fixed.py**
**Thay đổi:**
- ✅ Dùng `eos_token_id` trong generation
- ✅ Remove explanations sau AMR
- ✅ Auto calculate SMATCH

**Chạy:**
```bash
python predict_baseline_fixed.py \
  --model models_archive/baseline_7b_fixed \
  --test-file data/public_test.txt \
  --output evaluation_results/baseline_7b_fixed/predictions.txt
```

---

## 📊 Cấu trúc thư mục sau khi hoàn tất

```
ViSemPar_new1/
│
├── 📚 DOCUMENTATION (ĐỌC ĐẦU TIÊN)
│   ├── INDEX_RETRAIN_DOCS.md              ← Bạn đang đọc file này
│   ├── TRAINING_FIXES_SUMMARY.md          ⭐ Đọc đầu tiên
│   ├── RETRAIN_INSTRUCTIONS.md            📋 Step-by-step guide
│   ├── QUICK_START_RETRAIN.sh             🚀 Automated script
│   └── CLEANUP_AND_ORGANIZE.sh            🧹 Cleanup script
│
├── 🔧 CODE (FIXED VERSION)
│   ├── config/
│   │   ├── config.py                      ❌ Old (buggy)
│   │   └── config_fixed.py                ✅ New (use this)
│   ├── train_baseline.py                  ❌ Old (buggy)
│   ├── train_baseline_fixed.py            ✅ New (use this)
│   └── predict_baseline_fixed.py          ✅ New prediction
│
├── 💾 MODELS
│   └── models_archive/
│       ├── baseline_7b_old/               ⚠️ Archived (buggy)
│       │   ├── checkpoint-1545/
│       │   └── README.md
│       ├── baseline_7b_fixed/             ✅ New (after training)
│       │   ├── adapter_model.safetensors
│       │   ├── adapter_config.json
│       │   └── README.md
│       └── README.md
│
├── 📊 RESULTS
│   └── evaluation_results/
│       ├── baseline_7b_old/               Old results (17.3% errors)
│       │   ├── predictions_formatted.txt
│       │   └── README.md
│       └── baseline_7b_fixed/             ✅ New results
│           ├── predictions.txt
│           └── smatch_score.txt
│
└── 🗑️ TEMP
    └── temp_files/
        └── *.b64                          Base64 encoded files
```

---

## 🎯 Quick Reference

### Tôi muốn...

#### ...hiểu vấn đề là gì?
→ Đọc [TRAINING_FIXES_SUMMARY.md](TRAINING_FIXES_SUMMARY.md)

#### ...retrain ngay, tự động hết
→ Chạy `bash QUICK_START_RETRAIN.sh`

#### ...retrain từng bước thủ công
→ Đọc [RETRAIN_INSTRUCTIONS.md](RETRAIN_INSTRUCTIONS.md)

#### ...cleanup files cũ trước
→ Chạy `./CLEANUP_AND_ORGANIZE.sh`

#### ...xem code fix chi tiết
→ Xem:
- [config/config_fixed.py](config/config_fixed.py) - Prompt mới
- [train_baseline_fixed.py](train_baseline_fixed.py) - Training với masking
- [predict_baseline_fixed.py](predict_baseline_fixed.py) - Prediction

#### ...so sánh old vs new
→ Sau khi retrain xong:
```bash
echo "OLD:"
cat evaluation_results/baseline_7b_old/README.md

echo "NEW:"
wc -l evaluation_results/baseline_7b_fixed/predictions.txt
python -m smatch -f \
  evaluation_results/baseline_7b_fixed/predictions.txt \
  data/public_test_ground_truth.txt
```

---

## ⏱️ Timeline dự kiến

| Bước | Thời gian | Tự động? |
|------|-----------|----------|
| Đọc docs | 10-15 phút | ❌ Manual |
| Cleanup | 1 phút | ✅ Script |
| Training | 2-3 giờ | ✅ Script |
| Archive | 1 phút | ✅ Script |
| Prediction | 10-15 phút | ✅ Script |
| Evaluation | 5-10 phút | ✅ Script |
| **Tổng cộng** | **~3-4 giờ** | **Mostly automated** |

---

## 🚨 Troubleshooting

### Training failed?
1. Check [RETRAIN_INSTRUCTIONS.md](RETRAIN_INSTRUCTIONS.md) → Troubleshooting section
2. Check logs: `outputs/baseline_fixed_*/logs/`
3. Verify fixes applied correctly (checklist in docs)

### Vẫn có invalid AMRs?
1. Check EOS token được thêm chưa
2. Check instruction masking đúng chưa
3. Check generation config có `eos_token_id`

### Model vẫn generate giải thích?
1. Check prompt có "KHÔNG thêm giải thích" chưa
2. Check repetition_penalty = 1.15
3. Check temperature = 0.1

---

## 📞 Support

Nếu có vấn đề, check theo thứ tự:
1. [TRAINING_FIXES_SUMMARY.md](TRAINING_FIXES_SUMMARY.md) - Hiểu vấn đề
2. [RETRAIN_INSTRUCTIONS.md](RETRAIN_INSTRUCTIONS.md) - Troubleshooting
3. Training logs - `outputs/baseline_fixed_*/logs/`
4. Model README - `models_archive/baseline_7b_fixed/README.md`

---

## ✅ Checklist hoàn thành

Sau khi retrain xong, check:

- [ ] Đã đọc TRAINING_FIXES_SUMMARY.md
- [ ] Đã chạy CLEANUP_AND_ORGANIZE.sh
- [ ] Training hoàn tất không lỗi
- [ ] Model archived vào models_archive/baseline_7b_fixed/
- [ ] Predictions được generate thành công
- [ ] 150/150 AMRs hợp lệ (0% errors)
- [ ] SMATCH score tính được
- [ ] SMATCH F1 > old version
- [ ] So sánh với MTUP 7B

---

**Created:** 2026-01-03
**Author:** Claude Code
**Version:** 1.0
**Purpose:** Complete documentation index for baseline 7B retrain
