#!/bin/bash
# Quick fix: Downgrade to PEFT 0.4.0 which works with Transformers 4.31
# and doesn't have bitsandbytes hard dependency

set -e

echo "==========================================="
echo "QUICK FIX: INSTALL COMPATIBLE PEFT"
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
try:
    import transformers
    print(f"  Transformers: {transformers.__version__}")
except:
    pass

try:
    import peft
    print(f"  PEFT: {peft.__version__}")
except:
    print("  PEFT: not installed")

try:
    import bitsandbytes
    print(f"  bitsandbytes: {bitsandbytes.__version__}")
except:
    print("  bitsandbytes: not installed")
PYEOF
echo ""

# Uninstall everything
echo "Step 2: Removing PEFT and bitsandbytes..."
pip uninstall peft bitsandbytes -y 2>/dev/null || echo "  (not installed)"
echo "  ✓ Removed"
echo ""

# Install PEFT 0.4.0 (compatible with Transformers 4.31.x, no bitsandbytes hard dep)
echo "Step 3: Installing PEFT 0.4.0..."
pip install peft==0.4.0
echo "  ✓ Installed"
echo ""

# Verify
echo "Step 4: Verifying installation..."
python << 'PYEOF'
import sys

# Check bitsandbytes is NOT installed
try:
    import bitsandbytes
    print("  ✗ ERROR: bitsandbytes is still installed!")
    sys.exit(1)
except ImportError:
    print("  ✓ bitsandbytes not installed (good)")

# Check PEFT imports successfully WITHOUT bitsandbytes
try:
    from peft import LoraConfig, get_peft_model
    import peft
    print(f"  ✓ PEFT {peft.__version__} imports successfully")
    print("  ✓ All checks passed!")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "==========================================="
echo "QUICK FIX COMPLETE"
echo "==========================================="
echo ""
echo "PEFT 0.4.0 installed (compatible with your Transformers version)"
echo "bitsandbytes removed (not needed)"
echo ""
echo "Now start training:"
echo "  bash START_TRAINING_NOW.sh"
echo ""
