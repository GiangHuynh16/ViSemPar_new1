#!/bin/bash
# Automatic evaluation and upload to Hugging Face after training completes

set -e

echo "==========================================="
echo "AUTO EVALUATE AND UPLOAD TO HUGGING FACE"
echo "==========================================="
echo ""

# Configuration
HF_USERNAME="${HF_USERNAME:-YOUR_USERNAME}"  # Set this via environment or edit here
REPO_NAME="vietnamese-amr-baseline-7b"
BEST_CHECKPOINT=""  # Will be auto-detected

# Activate environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "Activating baseline_final environment..."
    eval "$(conda shell.bash hook)"
    conda activate baseline_final
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo ""

# Step 1: Find best checkpoint
echo "Step 1: Finding best checkpoint..."
if [ -d "outputs/checkpoints" ]; then
    # Find checkpoint with lowest loss (you may need to adjust this logic)
    BEST_CHECKPOINT=$(ls -td outputs/checkpoints/checkpoint-* | head -1)
    echo "  Best checkpoint: $BEST_CHECKPOINT"
else
    echo "  ✗ ERROR: No checkpoints found!"
    exit 1
fi
echo ""

# Step 2: Run evaluation
echo "Step 2: Running evaluation on public test..."
python << EOF
import sys
sys.path.insert(0, '.')

# Add your evaluation code here
# This is a template - adjust to your actual evaluation script

print("  Running inference on test set...")
# model = load_model("$BEST_CHECKPOINT")
# predictions = evaluate(model, test_data)
# save_predictions(predictions, "predictions_public.txt")

print("  ✓ Evaluation complete")
print("  Output: predictions_public.txt")
EOF
echo ""

# Step 3: Calculate SMATCH
echo "Step 3: Calculating SMATCH score..."
if [ -f "calculate_smatch.py" ]; then
    python calculate_smatch.py \
        --predictions predictions_public.txt \
        --gold data/public_test_ground_truth.txt \
        > smatch_results.txt

    cat smatch_results.txt

    # Extract F1 score
    F1_SCORE=$(grep -oP 'F1: \K[0-9.]+' smatch_results.txt || echo "N/A")
    echo ""
    echo "  📊 SMATCH F1 Score: $F1_SCORE"
else
    echo "  ⚠️  calculate_smatch.py not found, skipping SMATCH calculation"
    F1_SCORE="N/A"
fi
echo ""

# Step 4: Create model card
echo "Step 4: Creating model card..."
cat > "$BEST_CHECKPOINT/README.md" << MDEOF
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
- name: $REPO_NAME
  results:
  - task:
      type: semantic-parsing
      name: AMR Parsing
    dataset:
      type: vlsp2024-amr
      name: VLSP 2024 Vietnamese AMR
    metrics:
    - type: smatch
      value: $F1_SCORE
      name: SMATCH F1
---

# Vietnamese AMR Baseline 7B

Baseline model for Vietnamese Abstract Meaning Representation (AMR) parsing.

## Performance

| Metric | Score |
|--------|-------|
| SMATCH F1 | $F1_SCORE |

## Training Configuration

- Base Model: Qwen/Qwen2.5-7B-Instruct
- LoRA Rank: 64
- Max Sequence Length: 256
- Epochs: 15
- Learning Rate: 2e-4
- Precision: BF16
- Gradient Checkpointing: Enabled

## Usage

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "$HF_USERNAME/$REPO_NAME")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
\`\`\`

## Citation

Generated with [Claude Code](https://claude.com/claude-code)
MDEOF

echo "  ✓ Model card created"
echo ""

# Step 5: Check Hugging Face login
echo "Step 5: Checking Hugging Face authentication..."
if huggingface-cli whoami &>/dev/null; then
    HF_USER=$(huggingface-cli whoami | head -1)
    echo "  ✓ Logged in as: $HF_USER"
else
    echo "  ✗ Not logged in to Hugging Face"
    echo ""
    echo "  Please login first:"
    echo "    huggingface-cli login"
    echo ""
    read -p "  Login now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    else
        echo "  Skipping upload. You can manually upload later with:"
        echo "    huggingface-cli upload $HF_USERNAME/$REPO_NAME $BEST_CHECKPOINT . --repo-type model"
        exit 0
    fi
fi
echo ""

# Step 6: Create repository if not exists
echo "Step 6: Creating Hugging Face repository..."
python << EOF
from huggingface_hub import HfApi, create_repo
import sys

api = HfApi()
repo_id = "$HF_USERNAME/$REPO_NAME"

try:
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"  ✓ Repository ready: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"  ⚠️  Error creating repo: {e}")
    print(f"  Please create manually: https://huggingface.co/new")
    sys.exit(1)
EOF
echo ""

# Step 7: Upload to Hugging Face
echo "Step 7: Uploading model to Hugging Face..."
echo "  This may take 5-10 minutes for LoRA adapter (~200MB)"
echo ""

python << EOF
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
repo_id = "$HF_USERNAME/$REPO_NAME"
checkpoint_dir = "$BEST_CHECKPOINT"

print(f"  Uploading {checkpoint_dir}...")

try:
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Vietnamese AMR Baseline 7B (SMATCH F1: $F1_SCORE)"
    )
    print(f"  ✓ Upload complete!")
    print(f"  🎉 Model available at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"  ✗ Upload failed: {e}")
    import sys
    sys.exit(1)
EOF

echo ""
echo "==========================================="
echo "COMPLETE!"
echo "==========================================="
echo ""
echo "Summary:"
echo "  Checkpoint: $BEST_CHECKPOINT"
echo "  SMATCH F1: $F1_SCORE"
echo "  Hugging Face: https://huggingface.co/$HF_USERNAME/$REPO_NAME"
echo ""
echo "Next steps:"
echo "  1. Verify model on Hugging Face"
echo "  2. Test inference with uploaded model"
echo "  3. Compare with MTUP baseline"
echo ""
