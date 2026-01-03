# 🔧 Retrain Baseline 7B - FIXED VERSION

## ⚠️ Vấn đề với model cũ

Model baseline 7B đầu tiên có **3 lỗi nghiêm trọng**:

1. **❌ Thiếu EOS token**
   - Model không biết khi nào dừng generate
   - → Sinh ra text vô tận, thêm giải thích sau AMR

2. **❌ Không mask instruction**
   - Model học cả phần prompt "Bạn là chuyên gia..."
   - → Lãng phí capacity, chậm hội tụ

3. **❌ Prompt không rõ format Penman**
   - Model không hiểu cấu trúc AMR chuẩn
   - → 26/150 AMR không hợp lệ (17.3% lỗi)

## ✅ Các fix đã áp dụng

### 1. Thêm EOS Token
```python
# OLD (SAI):
text = PROMPT_TEMPLATE.format(sentence=sentence) + amr

# NEW (ĐÚNG):
full_text = prompt + amr + tokenizer.eos_token
```

### 2. Instruction Masking
```python
# Chỉ train trên AMR output, không train trên instruction
prompt_length = len(prompt_encoding['input_ids'][0])
labels[:prompt_length] = -100  # Mask instruction part
```

### 3. Prompt rõ ràng về Penman
```
Các quy tắc bắt buộc:
1. Sử dụng định dạng Penman: (biến / khái niệm :quan-hệ ...)
2. Khái niệm tiếng Việt đa âm tiết phải dùng dấu gạch dưới
3. Sử dụng các quan hệ chuẩn: :ARG0, :ARG1, :ARG2, ...
4. Đảm bảo cấu trúc cây với ngoặc đơn cân bằng
5. Mỗi khái niệm chỉ được gán một biến duy nhất
6. KHÔNG thêm giải thích, chỉ trả về AMR thuần túy
```

## 📋 Bước thực hiện

### Bước 1: Cleanup files cũ

```bash
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Run cleanup script
chmod +x CLEANUP_AND_ORGANIZE.sh
./CLEANUP_AND_ORGANIZE.sh
```

**Kết quả:**
- Model cũ → `models_archive/baseline_7b_old/` (archived)
- Results cũ → `evaluation_results/baseline_7b_old/`
- Temp files → `temp_files/`

### Bước 2: Activate conda environment

```bash
conda activate baseline_final
```

### Bước 3: Train model với fixes

```bash
python train_baseline_fixed.py --epochs 15 --show-sample
```

**Training sẽ:**
- ✅ Sử dụng `config_fixed.py` với prompt mới
- ✅ Thêm EOS token vào mỗi example
- ✅ Mask instruction (chỉ train trên AMR)
- ✅ Lưu checkpoint vào `outputs/baseline_fixed_YYYYMMDD_HHMMSS/`

**Expected output:**
```
╔══════════════════════════════════════════════════════════════╗
║     VIETNAMESE AMR PARSER - BASELINE TRAINING FIXED         ║
║     ✅ EOS Token | ✅ Instruction Masking | ✅ Penman      ║
╚══════════════════════════════════════════════════════════════╝

APPLYING FIXES:
  1. Adding EOS token to each example
  2. Preparing for instruction masking
  3. Using clear Penman format prompt

SAMPLE FIXED EXAMPLE:
PROMPT (will be masked):
Bạn là chuyên gia ngôn ngữ học máy tính...
...
AMR (will be trained):
(t / tuyên_bố :ARG0 ...)
```

### Bước 4: Archive model sau khi train xong

```bash
# Sau khi training hoàn tất
TIMESTAMP=$(ls -t outputs/ | grep baseline_fixed | head -1)
mv outputs/$TIMESTAMP/final models_archive/baseline_7b_fixed/

# Tạo README
cat > models_archive/baseline_7b_fixed/README.md << 'EOF'
# Baseline 7B Model - FIXED VERSION ✅

**Training Date:** $(date +%Y-%m-%d)
**Status:** ✅ All fixes applied

## Fixes:
1. ✅ EOS token added
2. ✅ Instruction masking enabled
3. ✅ Clear Penman format prompt

## Expected Results:
- Should generate valid Penman AMR
- Should stop at EOS token
- Should NOT generate explanations
- Error rate << 17.3% (old version)

## Training:
- Epochs: 15
- Batch size: 1 x 16 (gradient accumulation)
- Learning rate: 2e-4
- LoRA rank: 64
- Dataset: train_amr_1.txt + train_amr_2.txt
EOF
```

### Bước 5: Test và evaluate

```bash
# Generate predictions
python predict_baseline_fixed.py \
  --model models_archive/baseline_7b_fixed/final \
  --test-file data/public_test.txt \
  --output evaluation_results/baseline_7b_fixed/predictions.txt

# Check quality
python analyze_amr_quality.py \
  --file evaluation_results/baseline_7b_fixed/predictions.txt
```

**Expected:**
- ✅ 150/150 valid AMRs (0% error rate)
- ✅ All AMRs stop at proper endpoint
- ✅ No explanations after AMR
- ✅ SMATCH score calculable

### Bước 6: Compare với model cũ

```bash
echo "OLD MODEL (buggy):"
cat evaluation_results/baseline_7b_old/README.md

echo ""
echo "NEW MODEL (fixed):"
wc -l evaluation_results/baseline_7b_fixed/predictions.txt
python -m smatch -f \
  evaluation_results/baseline_7b_fixed/predictions.txt \
  data/public_test_ground_truth.txt \
  --significant 4
```

## 📊 Expected Improvements

| Metric | Old (Buggy) | New (Fixed) | Improvement |
|--------|-------------|-------------|-------------|
| Valid AMRs | 124/150 (82.7%) | 150/150 (100%) | +17.3% |
| Parse errors | 26 (17.3%) | 0 (0%) | -100% |
| SMATCH | Not calculable | XX.X% | Measurable! |
| Generates explanations | Yes ❌ | No ✅ | Fixed |
| Stops at EOS | No ❌ | Yes ✅ | Fixed |

## 🎯 Success Criteria

Model được coi là thành công khi:
- [ ] 100% AMRs hợp lệ (balanced parentheses, no duplicates)
- [ ] Model dừng đúng tại EOS token
- [ ] Không generate giải thích sau AMR
- [ ] SMATCH score được tính thành công
- [ ] SMATCH F1 > 0.0 (tối thiểu)

## 🚨 Troubleshooting

### Nếu vẫn còn invalid AMRs:
1. Check xem có thêm EOS token chưa:
   ```python
   # In create_baseline_dataset:
   full_text = prompt + amr + self.tokenizer.eos_token
   ```

2. Check instruction masking:
   ```python
   # Labels should have -100 for instruction part
   labels[:prompt_length] = -100
   ```

3. Check generation config:
   ```python
   # Must have eos_token_id
   outputs = model.generate(
       ...,
       eos_token_id=tokenizer.eos_token_id,
   )
   ```

### Nếu model vẫn generate giải thích:
1. Check prompt có rõ "KHÔNG thêm giải thích" chưa
2. Check repetition_penalty (nên là 1.15)
3. Check temperature (nên là 0.1 - deterministic)

## 📁 File Structure sau khi hoàn tất

```
ViSemPar_new1/
├── models_archive/
│   ├── baseline_7b_old/          ⚠️  Archived (buggy)
│   │   ├── checkpoint-1545/
│   │   └── README.md
│   ├── baseline_7b_fixed/        ✅ New (fixed)
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── README.md
│   └── README.md
├── evaluation_results/
│   ├── baseline_7b_old/          Old results
│   │   ├── predictions_formatted.txt
│   │   └── README.md
│   └── baseline_7b_fixed/        ✅ New results
│       ├── predictions.txt
│       └── smatch_score.txt
├── config/
│   ├── config.py                 Old config
│   └── config_fixed.py           ✅ New config
├── train_baseline.py             Old training script
├── train_baseline_fixed.py       ✅ New training script
└── predict_baseline_fixed.py     ✅ New prediction script
```

## ⏱️ Timeline

- Cleanup: ~1 phút
- Training: ~2-3 giờ (15 epochs, 7B model)
- Prediction: ~10-15 phút (150 sentences)
- Evaluation: ~5-10 phút

**Tổng cộng: ~3-4 giờ**

## 📝 Notes

- **CRITICAL:** Phải dùng `train_baseline_fixed.py`, KHÔNG dùng `train_baseline.py`
- **CRITICAL:** Phải import từ `config_fixed`, KHÔNG import từ `config`
- Backup model cũ trước khi cleanup (đã làm qua script)
- Monitor training loss - nên giảm đều, không NaN
- Check GPU memory - nên stable ~40-45GB

---

**Prepared by:** Claude Code
**Date:** 2026-01-03
**Version:** 1.0 - Fixed Baseline Training
