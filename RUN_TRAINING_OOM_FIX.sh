#!/bin/bash
################################################################################
# FIX OOM - RUN TRAINING WITH MEMORY OPTIMIZATION
# Copy-paste vào server và chạy
################################################################################

echo "========================================================================"
echo "🚀 RUNNING TRAINING WITH OOM FIX"
echo "========================================================================"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lora_py310

# Set memory optimization flags (use new PYTORCH_ALLOC_CONF)
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# Clear GPU cache
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ GPU cache cleared")
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
EOF

echo ""
echo "Memory optimization settings:"
echo "  - PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128"
echo "  - CPU offloading enabled (20GB GPU + 30GB CPU)"
echo "  - Batch size: 1"
echo "  - Gradient accumulation: 2 (reduced from 4)"
echo "  - Max samples: 50"
echo ""

# Run training with minimal memory settings
python3 train_mtup.py --use-case quick_test --show-sample --no-quantize \
  --batch-size 1 \
  --grad-accum 2 \
  --max-samples 50

echo ""
echo "========================================================================"
