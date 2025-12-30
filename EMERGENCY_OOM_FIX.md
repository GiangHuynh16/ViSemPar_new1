# Emergency OOM Fix - Baseline 7B Training

**Last Updated**: 2025-12-30
**Status**: Maximum optimization applied

---

## Current Optimization Level: MAXIMUM

Đã áp dụng **mọi tối ưu có thể** mà KHÔNG giảm model capacity:

```python
# config/config.py
MAX_SEQ_LENGTH = 1024           # 50% reduction from 2048
batch_size = 1                   # Minimum possible
gradient_accumulation = 16       # Maximum to maintain effective batch

# train_baseline.py
max_memory = {0: "12GB", "cpu": "50GB"}  # Maximum CPU offload
gradient_checkpointing = True    # Via TrainingArguments only
```

### Memory Budget với max_memory=12GB:
```
Base model on GPU:      ~10 GB  (majority on CPU)
LoRA adapters:           ~0.5 GB
Activations (bs=1):      ~1.5 GB
Gradients:               ~1 GB
Optimizer:               ~1 GB
------------------------------------
Total GPU:              ~14 GB  ✅ (10GB margin)
```

---

## Nếu VẪN OOM → Chỉ còn 2 options

### Option 1: Giảm LoRA Rank (Affects comparison)

```python
# config/config.py
LORA_CONFIG = {
    "r": 64,        # Từ 128 → giảm 50% trainable params
    "lora_alpha": 128,  # 2x rank
}
```

**⚠️ WARNING**: Này sẽ ảnh hưởng fair comparison với MTUP (MTUP dùng rank 128)

**Memory saved**: ~1-2 GB

---

### Option 2: Dùng Model 3B thay vì 7B

```python
# config/config.py
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # Từ 7B
MAX_SEQ_LENGTH = 2048  # Có thể dùng lại giá trị cao
batch_size = 2         # Có thể tăng lại
```

**⚠️ WARNING**: Không còn fair comparison với MTUP 7B nữa

**Memory saved**: ~8-10 GB

**Pros**:
- Sẽ chắc chắn fit memory
- Training nhanh hơn (~6-8 hours)

**Cons**:
- Model nhỏ hơn, có thể F1 thấp hơn
- Không so sánh được với MTUP 7B

---

## Debugging Commands

### Check GPU memory usage in real-time:
```bash
watch -n 0.5 nvidia-smi
```

### Check which process uses GPU:
```bash
fuser -v /dev/nvidia*
```

### Kill all Python processes:
```bash
pkill -9 python
sleep 5
nvidia-smi --gpu-reset
```

### Check model loading memory:
```bash
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM

print("Loading model with max_memory=12GB...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    max_memory={0: "12GB", "cpu": "50GB"},
    torch_dtype=torch.float16
)

print(f"Model loaded!")
print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Check device map
print("\nDevice map:")
for name, param in model.named_parameters():
    if hasattr(param, 'device'):
        print(f"{name}: {param.device}")
    if "embed" in name or "lm_head" in name or "layers.0" in name:
        print(f"  -> {name}: {param.device}")

del model
torch.cuda.empty_cache()
print("\nModel deleted, cache cleared")
EOF
```

---

## Last Resort: Switch to LoRA rank 64

Nếu KHÔNG thể training với config hiện tại, đây là cách switch sang LoRA rank 64:

### Step 1: Update config
```bash
# Edit config/config.py
nano config/config.py

# Change:
LORA_CONFIG = {
    "r": 64,           # FROM 128
    "lora_alpha": 128, # FROM 256
    ...
}
```

### Step 2: Update documentation
```bash
# Update training guide to mention LoRA rank 64
echo "⚠️ Using LoRA rank 64 instead of 128 due to memory constraints" >> TRAINING_NOTES.md
```

### Step 3: Train and document
- Train với rank 64
- Document rõ trong thesis: "Due to hardware constraints, baseline uses rank 64 while MTUP uses rank 128"
- Still valid comparison: MTUP's advantage comes from architecture, not just params

### Step 4: After training, compare fairly
- Train ANOTHER baseline với rank 64 để so sánh công bằng
- Hoặc note in thesis: "Conservative comparison - baseline handicapped by lower rank"

---

## Contact Points

Nếu tất cả đều fail, có 3 options:

1. **Rent cloud GPU**:
   - RunPod: ~$0.4/hour for A40 48GB
   - Vast.ai: ~$0.3/hour for RTX 3090 24GB
   - Chạy 1 lần, ~$6-8 total

2. **Use smaller model (3B)**:
   - Less powerful but fits easily
   - Still demonstrates the approach

3. **Accept rank 64**:
   - Document in thesis
   - Still valid for demonstrating multi-task vs single-task

---

## Quick Test Before Full Training

Test nếu model fit memory trước khi training 15 epochs:

```bash
cd ViSemPar_new1
git pull origin main

# Test with 1 epoch only
python train_baseline.py --epochs 1 --show-sample

# If successful after 1 epoch, run full training:
bash START_BASELINE_7B_TRAINING.sh
```

---

## Summary

**Current state**:
- Maximum optimization applied
- max_memory = 12GB (cannot go lower for 7B model)
- batch_size = 1 (cannot go lower)
- max_seq_length = 1024 (already very low)

**If still OOM**:
1. Try rank 64 (affects comparison but still valid)
2. Try 3B model (different comparison but works)
3. Rent cloud GPU for one run

**Expected**: Current config SHOULD work now with 12GB GPU limit + 50GB CPU
