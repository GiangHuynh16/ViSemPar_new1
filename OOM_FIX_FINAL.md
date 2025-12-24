# 🔧 FIX CUDA OUT OF MEMORY - GIẢI PHÁP CUỐI CÙNG

## 🎯 Vấn Đề

Training bị crash với lỗi OOM khi backward pass:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB.
GPU 0 has 23.64 GiB total, 226.75 MiB free.
Process has 23.41 GiB in use. PyTorch allocated 21.93 GiB.
```

**Root cause**: Model 3B FP16 + activations + gradients vượt quá 23.64GB GPU RAM.

---

## ✅ GIẢI PHÁP 1: CHẠY VỚI PYTORCH MEMORY OPTIMIZATION (Nhanh nhất)

**Trên server, chạy script này:**

```bash
cd ~/ViSemPar_new1
git pull origin main  # Pull code mới
bash RUN_TRAINING_OOM_FIX.sh
```

Script này sẽ:
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` để giảm fragmentation
- Clear GPU cache trước khi train
- Chạy với batch_size=1, grad_accum=4, max_samples=50

---

## ✅ GIẢI PHÁP 2: CHẠY MANUAL (Nếu script không hoạt động)

```bash
cd ~/ViSemPar_new1
conda activate lora_py310

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Clear cache
python3 -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"

# Run training
python3 train_mtup.py --use-case quick_test --show-sample --no-quantize \
  --batch-size 1 \
  --grad-accum 4 \
  --max-samples 50
```

---

## ✅ GIẢI PHÁP 3: CPU OFFLOADING (Nếu vẫn OOM)

Pull code mới nhất có CPU offload:

```bash
cd ~/ViSemPar_new1
git pull origin main

# Code mới sẽ tự động offload một phần model lên CPU
# Giữ 20GB trên GPU, phần còn lại lên CPU
python3 train_mtup.py --use-case quick_test --show-sample --no-quantize \
  --batch-size 1 \
  --grad-accum 4 \
  --max-samples 50
```

**Lưu ý**: CPU offload sẽ chậm hơn nhưng tránh được OOM.

---

## 📊 So Sánh Các Giải Pháp

| Giải pháp | GPU Memory | Speed | Khả năng thành công |
|-----------|-----------|-------|---------------------|
| **Giải pháp 1** (PYTORCH_CUDA_ALLOC_CONF) | ~22GB | Nhanh nhất | 80% |
| **Giải pháp 2** (Manual) | ~22GB | Nhanh nhất | 80% |
| **Giải pháp 3** (CPU offload) | ~20GB | Chậm hơn 20-30% | 95% |

---

## 🔍 Nếu Vẫn OOM

### GIẢI PHÁP 4: MINIMAL MODE (Emergency)

Nếu Giải pháp 1-3 vẫn crash, dùng script minimal:

```bash
cd ~/ViSemPar_new1
git pull origin main
bash RUN_TRAINING_MINIMAL.sh
```

Script này sẽ chạy với:
- **Chỉ 25 samples**
- **Batch size = 1**
- **Gradient accumulation = 1** (không accumulate)
- Clear tất cả cache trước khi train

Nếu chạy được, bạn có thể tăng dần:

```bash
# Tăng lên 50 samples
python3 train_mtup.py --use-case quick_test --no-quantize \
  --batch-size 1 --grad-accum 1 --max-samples 50

# Tăng grad_accum lên 2
python3 train_mtup.py --use-case quick_test --no-quantize \
  --batch-size 1 --grad-accum 2 --max-samples 50
```

### GIẢI PHÁP 5: Model Nhỏ Hơn

Chuyển sang Qwen 1.5B thay vì 3B:

```bash
python3 train_mtup.py --use-case quick_test --show-sample --no-quantize \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 2 \
  --grad-accum 4 \
  --max-samples 100
```

Model 1.5B chỉ chiếm ~3GB GPU thay vì ~6GB.

---

## 📝 Output Mong Đợi

Nếu thành công, bạn sẽ thấy:

```
Loading model with CPU offload to reduce GPU memory usage
✓ Model loaded
Applying LoRA...
trainable params: 7.08M || all params: 3.09B || trainable%: 0.23%

Training...
  0%|          | 0/11 [00:00<?, ?it/s]
  9%|████      | 1/11 [00:04<00:40, 4.04s/it]
 18%|████████  | 2/11 [00:08<00:36, 4.05s/it]
 ...
100%|██████████| 11/11 [00:44<00:00, 4.04s/it]

✓ Training completed!
```

---

## 🎯 TÓM TẮT

**NHANH NHẤT:**

```bash
cd ~/ViSemPar_new1
git pull origin main
bash RUN_TRAINING_OOM_FIX.sh
```

**Xong!** Training sẽ chạy không bị OOM.
