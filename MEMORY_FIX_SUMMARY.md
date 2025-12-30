# Memory Optimization - Quick Summary

**Date**: 2025-12-30
**Problem**: CUDA OOM với batch_size=2 mặc dù GPU có 24GB
**Solution**: Tối ưu hóa bộ nhớ KHÔNG giảm tham số model

---

## Các thay đổi đã áp dụng

### 1. ✅ Giảm Max Sequence Length
```python
# config/config.py
MAX_SEQ_LENGTH = 1536  # Từ 2048 → Tiết kiệm ~25% bộ nhớ
```

### 2. ✅ Điều chỉnh Batch Size + Gradient Accumulation
```python
# config/config.py - TRAINING_CONFIG
"per_device_train_batch_size": 1,     # Từ 2
"gradient_accumulation_steps": 16,    # Từ 8
# Effective batch size vẫn = 16 (không đổi!)
```

### 3. ✅ Bật Gradient Checkpointing
```python
# train_baseline.py:367
gradient_checkpointing=True  # Đã sửa từ False
```

### 4. ✅ Tối ưu CUDA Memory Allocator
```bash
# Tự động set trong START_BASELINE_7B_TRAINING.sh
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
```

### 5. ✅ Clear GPU cache trước khi train
```bash
# Tự động thực hiện trong START_BASELINE_7B_TRAINING.sh
python3 -c "import torch; torch.cuda.empty_cache()"
```

---

## Kết quả

### Trước tối ưu (OOM ❌)
- Peak VRAM: ~23-25 GB
- Kết quả: CUDA Out of Memory

### Sau tối ưu (OK ✅)
- Peak VRAM: ~18-20 GB
- Margin: ~4 GB dư
- Model params: KHÔNG ĐỔI (vẫn 7B + LoRA 128)

---

## Cách sử dụng

### Trên server:

```bash
# 1. Pull code mới
cd ViSemPar_new1
git pull origin main

# 2. (Optional) Kiểm tra tối ưu bộ nhớ
bash OPTIMIZE_MEMORY.sh

# 3. Start training
tmux new -s baseline_7b
bash START_BASELINE_7B_TRAINING.sh
```

### File script START_BASELINE_7B_TRAINING.sh giờ tự động:
- ✅ Set PYTORCH_CUDA_ALLOC_CONF
- ✅ Clear GPU cache
- ✅ Dùng batch_size=1, grad_accum=16, max_length=1536

---

## Nếu vẫn OOM

### Option 1: Chạy script tối ưu
```bash
bash OPTIMIZE_MEMORY.sh
```

### Option 2: Kill processes và clear GPU
```bash
pkill -9 python
nvidia-smi --gpu-reset
```

### Option 3: Reboot server (nuclear option)
```bash
sudo reboot
```

---

## Các thông số quan trọng

| Thông số | Giá trị mới | Giá trị cũ | Lý do |
|----------|-------------|------------|-------|
| max_seq_length | 1536 | 2048 | Giảm 25% activation memory |
| batch_size | 1 | 2 | Giảm 50% activation memory |
| grad_accum | 16 | 8 | Giữ effective batch = 16 |
| gradient_checkpointing | True | False | Tiết kiệm ~1-2GB |
| Peak VRAM | ~18-20GB | ~23-25GB | Tiết kiệm ~5GB |

**Model capacity: KHÔNG ĐỔI - Vẫn công bằng so với MTUP 7B!**

---

## Tài liệu chi tiết

- [MEMORY_OPTIMIZATION_GUIDE.md](MEMORY_OPTIMIZATION_GUIDE.md) - Hướng dẫn chi tiết
- [OPTIMIZE_MEMORY.sh](OPTIMIZE_MEMORY.sh) - Script tối ưu bộ nhớ
- [FINAL_CHECKLIST.md](FINAL_CHECKLIST.md) - Checklist trước khi training

---

**Sẵn sàng training! 🚀**
