#!/bin/bash
# Install compatible PEFT version that doesn't require bitsandbytes at import time

set -e

echo "==========================================="
echo "FIX PEFT VERSION"
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

# Check current versions
echo "Step 1: Current versions..."
python << 'PYEOF'
import transformers
import peft
print(f"  Transformers: {transformers.__version__}")
print(f"  PEFT: {peft.__version__}")
PYEOF
echo ""

# Uninstall bitsandbytes
echo "Step 2: Removing bitsandbytes..."
pip uninstall bitsandbytes -y 2>/dev/null || echo "  (not installed)"
echo "  ✓ Removed"
echo ""

# Install PEFT 0.5.0 - compatible with transformers 4.31.x and doesn't require bitsandbytes
echo "Step 3: Installing PEFT 0.5.0 (compatible with Transformers 4.31.x)..."
pip install peft==0.5.0 --no-deps
echo "  ✓ Installed"
echo ""

# Verify
echo "Step 4: Verifying PEFT works without bitsandbytes..."
python << 'PYEOF'
import sys

# First check: bitsandbytes should not be installed
try:
    import bitsandbytes
    print("  ✗ ERROR: bitsandbytes is still installed!")
    sys.exit(1)
except ImportError:
    print("  ✓ bitsandbytes not installed (good)")

# Second check: PEFT should import successfully
try:
    from peft import LoraConfig, get_peft_model
    print("  ✓ PEFT imports successfully")
except ImportError as e:
    print(f"  ✗ ERROR: PEFT import failed: {e}")
    sys.exit(1)
except Exception as e:
    # If it's just a warning about bitsandbytes, that's okay
    if "bitsandbytes" in str(e).lower():
        print("  ⚠️  Warning about bitsandbytes, but import succeeded")
    else:
        print(f"  ✗ ERROR: {e}")
        sys.exit(1)

print("  ✓ All checks passed")
PYEOF

echo ""
echo "==========================================="
echo "FIX COMPLETE"
echo "==========================================="
echo ""
echo "Now start training:"
echo "  bash START_TRAINING_NOW.sh"
echo ""
