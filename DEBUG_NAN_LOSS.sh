#!/bin/bash
# Debug NaN loss issue on server
# Run this to identify why loss is still 0.0 and grad_norm is NaN

set -e

echo "==========================================="
echo "DEBUG NaN LOSS ISSUE"
echo "==========================================="
echo ""

# Activate environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "Activating baseline_final environment..."
    eval "$(conda shell.bash hook)"
    conda activate baseline_final
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo ""

# Navigate to project
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Check current config
echo "Step 1: Checking current config..."
echo ""
grep -A 5 "fp16\|bf16" config/config.py
echo ""

# Clear Python cache to ensure new code is used
echo "Step 2: Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  ✓ Cache cleared"
echo ""

# Run BF16 test
echo "Step 3: Running BF16 forward pass test..."
echo ""
python test_bf16_forward.py
TEST_EXIT=$?
echo ""

if [ $TEST_EXIT -eq 0 ]; then
    echo "==========================================="
    echo "✅ BF16 TEST PASSED"
    echo "==========================================="
    echo ""
    echo "BF16 is working correctly. The issue may be:"
    echo "1. Config not being loaded properly by Trainer"
    echo "2. Data collator or dataset issue"
    echo "3. TrainingArguments override somewhere"
    echo ""
    echo "Next: Check if TrainingArguments is using BF16"
    echo ""
else
    echo "==========================================="
    echo "❌ BF16 TEST FAILED"
    echo "==========================================="
    echo ""
    echo "BF16 is NOT working on this system. Solutions:"
    echo ""
    echo "Option 1: Disable gradient checkpointing"
    echo "  Edit config/config.py and add:"
    echo "  'gradient_checkpointing': False,"
    echo ""
    echo "Option 2: Use FP32 (slower but stable)"
    echo "  Edit config/config.py:"
    echo "  'fp16': False,"
    echo "  'bf16': False,"
    echo ""
    echo "Option 3: Reduce sequence length further"
    echo "  MAX_SEQ_LENGTH = 1024"
    echo "  batch_size = 1"
    echo ""
fi

echo ""
echo "==========================================="
echo "ADDITIONAL DIAGNOSTIC INFO"
echo "==========================================="
echo ""

# Check PyTorch and CUDA versions
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check BF16 support
if hasattr(torch.cuda, 'is_bf16_supported'):
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"BF16 supported by GPU: {bf16_supported}")
else:
    print("BF16 support check not available in this PyTorch version")
    print("Recommended: PyTorch >= 1.10 for BF16 support")

# Check installed package versions
try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except:
    pass

try:
    import peft
    print(f"PEFT: {peft.__version__}")
except:
    pass
EOF

echo ""
echo "==========================================="
