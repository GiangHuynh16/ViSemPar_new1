# Hướng dẫn Evaluate và Push Model lên Hugging Face

## Bước 1: Kiểm tra Training đã hoàn thành

Sau khi training xong (15 epochs, ~10-12 giờ), kiểm tra:

```bash
# Xem log cuối cùng
tail -100 logs/training_baseline*.log

# Kiểm tra checkpoint cuối
ls -lh outputs/checkpoints/
```

Tìm dòng: `***** train completed *****` hoặc tương tự để confirm training xong.

## Bước 2: Evaluate Model

### 2.1. Chạy evaluation trên test set

```bash
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
conda activate baseline_final

# Evaluate trên public test
python train_baseline.py --eval-only --checkpoint outputs/checkpoints/checkpoint-XXXX

# Hoặc dùng script evaluate riêng (nếu có)
python evaluate.py --model-path outputs/checkpoints/checkpoint-XXXX \
                   --test-file data/public_test.txt \
                   --output predictions_public.txt
```

### 2.2. Tính SMATCH score

```bash
# So sánh với ground truth
python calculate_smatch.py \
    --predictions predictions_public.txt \
    --gold data/public_test_ground_truth.txt

# Kết quả sẽ hiển thị:
# Precision: X.XX
# Recall: X.XX
# F1 (SMATCH): X.XX
```

### 2.3. Kiểm tra một vài ví dụ

```bash
# Xem 10 predictions đầu tiên
head -20 predictions_public.txt

# So sánh với ground truth
paste <(head -20 predictions_public.txt) <(head -20 data/public_test_ground_truth.txt) | column -t
```

## Bước 3: So sánh với MTUP 7B

```bash
echo "=== BASELINE 7B vs MTUP 7B Comparison ==="
echo ""
echo "BASELINE 7B (Single-task):"
echo "  F1 Score: [Điền score của bạn]"
echo ""
echo "MTUP 7B (Multi-task):"
echo "  F1 Score: [Điền score MTUP]"
echo ""
echo "Difference: [Tính hiệu số]"
```

## Bước 4: Chuẩn bị Model để Push

### 4.1. Tạo model card (README.md)

```bash
# Tạo README cho model
cat > model_card.md << 'EOF'
---
language: vi
license: apache-2.0
tags:
- vietnamese
- amr
- semantic-parsing
- qwen2.5
datasets:
- vlsp2024-amr
metrics:
- smatch
model-index:
- name: vietnamese-amr-baseline-7b
  results:
  - task:
      type: semantic-parsing
      name: AMR Parsing
    dataset:
      type: vlsp2024-amr
      name: VLSP 2024 Vietnamese AMR
    metrics:
    - type: smatch
      value: [ĐIỀN F1 SCORE CỦA BẠN]
      name: SMATCH F1
---

# Vietnamese AMR Baseline 7B

Baseline model for Vietnamese Abstract Meaning Representation (AMR) parsing, trained on VLSP 2024 dataset.

## Model Details

- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Training Approach**: Single-task (baseline) with LoRA
- **Language**: Vietnamese
- **Task**: AMR Semantic Parsing

## Training Configuration

```yaml
Model: Qwen 2.5 7B Instruct
LoRA Rank: 64
Max Sequence Length: 256
Batch Size: 1 (with gradient accumulation 16)
Epochs: 15
Learning Rate: 2e-4
Optimizer: AdamW
Precision: BF16
Gradient Checkpointing: Enabled
```

## Performance

| Metric | Score |
|--------|-------|
| SMATCH F1 | [ĐIỀN SCORE] |
| Precision | [ĐIỀN SCORE] |
| Recall | [ĐIỀN SCORE] |

## Comparison with MTUP

This baseline model is trained for comparison with Multi-Task Unified Pre-training (MTUP) approach.

| Model | Approach | F1 Score |
|-------|----------|----------|
| Baseline 7B | Single-task | [ĐIỀN SCORE] |
| MTUP 7B | Multi-task | [ĐIỀN SCORE] |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/vietnamese-amr-baseline-7b")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Inference
prompt = """Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt. Hãy chuyển đổi câu sau sang định dạng AMR.

Câu tiếng Việt: Chủ tịch nước đã phát biểu tại hội nghị.

AMR:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
amr = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(amr)
```

## Training Details

See [training logs](logs/) and [configuration](config/config.py) for full details.

## Citation

```bibtex
@misc{vietnamese-amr-baseline-7b,
  author = {[TÊN CỦA BẠN]},
  title = {Vietnamese AMR Baseline 7B},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/[YOUR_USERNAME]/vietnamese-amr-baseline-7b}
}
```
EOF
```

### 4.2. Merge LoRA weights (optional - để model dễ dùng hơn)

```python
# merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    "outputs/checkpoints/checkpoint-XXXX"  # Best checkpoint
)

print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained("merged_model")

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.save_pretrained("merged_model")

print("Done!")
```

Chạy:
```bash
python merge_lora.py
```

## Bước 5: Push lên Hugging Face

### 5.1. Login vào Hugging Face

```bash
# Install huggingface-cli nếu chưa có
pip install huggingface_hub

# Login (cần token từ https://huggingface.co/settings/tokens)
huggingface-cli login
# Paste token khi được hỏi
```

### 5.2. Tạo repository

```bash
# Tạo repo mới
huggingface-cli repo create vietnamese-amr-baseline-7b --type model

# Hoặc dùng Python
python << 'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(
    repo_id="vietnamese-amr-baseline-7b",
    repo_type="model",
    private=False  # public repo
)
EOF
```

### 5.3. Upload model

**Option 1: Upload LoRA adapter (nhẹ hơn, ~200MB)**

```bash
cd outputs/checkpoints/checkpoint-XXXX  # Best checkpoint

# Copy model card
cp /path/to/model_card.md README.md

# Upload
huggingface-cli upload YOUR_USERNAME/vietnamese-amr-baseline-7b . . --repo-type model
```

**Option 2: Upload merged model (đầy đủ, ~14GB)**

```bash
cd merged_model

# Copy model card
cp /path/to/model_card.md README.md

# Upload
huggingface-cli upload YOUR_USERNAME/vietnamese-amr-baseline-7b . . --repo-type model
```

**Option 3: Upload bằng Python (recommended - có progress bar)**

```python
# upload_to_hf.py
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "YOUR_USERNAME/vietnamese-amr-baseline-7b"

# Option 1: Upload LoRA adapter
checkpoint_dir = "outputs/checkpoints/checkpoint-XXXX"

# Option 2: Upload merged model
# checkpoint_dir = "merged_model"

print(f"Uploading {checkpoint_dir} to {repo_id}...")

api.upload_folder(
    folder_path=checkpoint_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload Vietnamese AMR Baseline 7B model"
)

# Upload README
api.upload_file(
    path_or_fileobj="model_card.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model"
)

print(f"✓ Model uploaded to https://huggingface.co/{repo_id}")
```

Chạy:
```bash
python upload_to_hf.py
```

## Bước 6: Verify Model trên Hugging Face

1. Mở https://huggingface.co/YOUR_USERNAME/vietnamese-amr-baseline-7b
2. Kiểm tra:
   - ✓ Model card hiển thị đúng
   - ✓ Files đã upload đầy đủ
   - ✓ Metrics hiển thị
3. Test model bằng Inference API (nếu có)

## Bước 7: Tạo Summary Report

```bash
cat > TRAINING_REPORT.md << 'EOF'
# Vietnamese AMR Baseline 7B - Training Report

## Training Summary

- **Start Time**: [ĐIỀN THỜI GIAN BẮT ĐẦU]
- **End Time**: [ĐIỀN THỜI GIAN KẾT THÚC]
- **Total Duration**: ~XX hours
- **GPU**: NVIDIA RTX A6000 (48GB)
- **Final Loss**: [ĐIỀN LOSS CUỐI]

## Evaluation Results

### Public Test Set

| Metric | Score |
|--------|-------|
| SMATCH F1 | XX.XX |
| Precision | XX.XX |
| Recall | XX.XX |

### Comparison with MTUP

| Model | F1 Score | Difference |
|-------|----------|------------|
| MTUP 7B | XX.XX | - |
| Baseline 7B | XX.XX | ±X.XX |

## Model Location

- **Hugging Face**: https://huggingface.co/YOUR_USERNAME/vietnamese-amr-baseline-7b
- **Local Checkpoint**: outputs/checkpoints/checkpoint-XXXX

## Key Findings

[ĐIỀN NHẬN XÉT CỦA BẠN]
- Baseline so với MTUP: ...
- Training stability: ...
- Best practices: ...

## Next Steps

- [ ] Test on private test set
- [ ] Compare with other baselines
- [ ] Analyze error cases
- [ ] Write paper/report
EOF
```

## Tóm tắt Commands

```bash
# 1. Evaluate
python train_baseline.py --eval-only --checkpoint outputs/checkpoints/checkpoint-XXXX

# 2. Calculate SMATCH
python calculate_smatch.py --predictions predictions.txt --gold ground_truth.txt

# 3. Login Hugging Face
huggingface-cli login

# 4. Upload model
python upload_to_hf.py

# 5. Verify
firefox https://huggingface.co/YOUR_USERNAME/vietnamese-amr-baseline-7b
```

## Lưu ý quan trọng

1. **Chọn best checkpoint**: Dựa vào validation loss, không phải checkpoint cuối cùng
2. **Test trước khi upload**: Chạy inference vài ví dụ để đảm bảo model hoạt động
3. **Ghi rõ config**: Document tất cả hyperparameters để reproduce được
4. **So sánh fair**: Đảm bảo MTUP và Baseline dùng cùng test set và metric
5. **Backup**: Lưu checkpoint tốt nhất ở nhiều nơi

Chúc bạn thành công! 🚀
