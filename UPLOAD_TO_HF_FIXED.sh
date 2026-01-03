#!/bin/bash
# Upload baseline 7B fixed model to HuggingFace

echo "=========================================="
echo "UPLOAD BASELINE 7B FIXED TO HUGGINGFACE"
echo "=========================================="
echo ""

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final

# Find latest checkpoint
LATEST_MODEL=$(ls -t outputs/ | grep baseline_fixed | head -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "❌ No trained model found!"
    exit 1
fi

echo "Found model: outputs/$LATEST_MODEL"
echo ""

# Find best checkpoint (highest number)
BEST_CHECKPOINT=$(ls -t "outputs/$LATEST_MODEL" | grep checkpoint | head -1)

if [ -z "$BEST_CHECKPOINT" ]; then
    echo "❌ No checkpoint found in outputs/$LATEST_MODEL"
    exit 1
fi

CHECKPOINT_PATH="outputs/$LATEST_MODEL/$BEST_CHECKPOINT"

echo "Best checkpoint: $CHECKPOINT_PATH"
echo ""

# Step 1: Add tokenizer to checkpoint
echo "=========================================="
echo "STEP 1: Add tokenizer to checkpoint"
echo "=========================================="
echo ""

python add_tokenizer_to_checkpoint.py --checkpoint "$CHECKPOINT_PATH"

if [ $? -ne 0 ]; then
    echo "❌ Failed to add tokenizer"
    exit 1
fi

echo ""

# Step 2: Login to HuggingFace (if needed)
echo "=========================================="
echo "STEP 2: HuggingFace login"
echo "=========================================="
echo ""

huggingface-cli whoami 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Not logged in. Please login:"
    huggingface-cli login
fi

echo ""

# Step 3: Upload
echo "=========================================="
echo "STEP 3: Upload to HuggingFace"
echo "=========================================="
echo ""

# Ask for repo name
read -p "Enter HuggingFace repo name (e.g., your-username/vietnamese-amr-baseline-7b): " REPO_NAME

if [ -z "$REPO_NAME" ]; then
    echo "❌ Repo name required!"
    exit 1
fi

echo ""
echo "Uploading to: $REPO_NAME"
echo "From: $CHECKPOINT_PATH"
echo ""

# Upload
huggingface-cli upload "$REPO_NAME" "$CHECKPOINT_PATH" --repo-type model

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ UPLOAD COMPLETE"
    echo "=========================================="
    echo ""
    echo "Model URL: https://huggingface.co/$REPO_NAME"
    echo ""
    echo "To use the model:"
    echo ""
    echo "from transformers import AutoTokenizer, AutoModelForCausalLM"
    echo "from peft import PeftModel"
    echo ""
    echo "# Load base model"
    echo "base_model = AutoModelForCausalLM.from_pretrained("
    echo "    \"Qwen/Qwen2.5-7B-Instruct\","
    echo "    torch_dtype=torch.bfloat16,"
    echo "    device_map=\"auto\""
    echo ")"
    echo ""
    echo "# Load LoRA adapter"
    echo "model = PeftModel.from_pretrained(base_model, \"$REPO_NAME\")"
    echo ""
    echo "# Load tokenizer"
    echo "tokenizer = AutoTokenizer.from_pretrained(\"$REPO_NAME\")"
    echo ""
else
    echo ""
    echo "❌ Upload failed!"
    exit 1
fi
