#!/bin/bash
# Patch PEFT to make bitsandbytes import optional (lazy import)

set -e

echo "==========================================="
echo "PATCH PEFT FOR OPTIONAL BITSANDBYTES"
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

# Find PEFT installation
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
PEFT_LORA_MODEL="$SITE_PACKAGES/peft/tuners/lora/model.py"

echo "Step 1: Locating PEFT installation..."
echo "  PEFT LoRA model: $PEFT_LORA_MODEL"

if [ ! -f "$PEFT_LORA_MODEL" ]; then
    echo "  ✗ ERROR: PEFT file not found!"
    exit 1
fi
echo "  ✓ Found"
echo ""

# Backup original file
echo "Step 2: Backing up original file..."
cp "$PEFT_LORA_MODEL" "$PEFT_LORA_MODEL.backup"
echo "  ✓ Backed up to $PEFT_LORA_MODEL.backup"
echo ""

# Check if already patched
if grep -q "# PATCHED: lazy import bitsandbytes" "$PEFT_LORA_MODEL"; then
    echo "  ⚠️  File already patched, skipping..."
else
    # Patch the file - replace "import bitsandbytes as bnb" with lazy import
    echo "Step 3: Patching PEFT to use lazy bitsandbytes import..."
    
    python << 'PYEOF'
import re

peft_file = """SITE_PACKAGES""" + "/peft/tuners/lora/model.py"
peft_file = peft_file.replace("SITE_PACKAGES", """$SITE_PACKAGES""")

with open(peft_file, 'r') as f:
    content = f.read()

# Replace the problematic import with a lazy import
old_import = "import bitsandbytes as bnb"
new_import = """# PATCHED: lazy import bitsandbytes
bnb = None
try:
    import bitsandbytes as bnb
except ImportError:
    pass  # bitsandbytes not available, that's okay if not using quantization"""

if old_import in content:
    content = content.replace(old_import, new_import)
    with open(peft_file, 'w') as f:
        f.write(content)
    print("  ✓ Patched successfully")
else:
    print("  ⚠️  Import pattern not found, file may be different version")
PYEOF

fi
echo ""

# Uninstall bitsandbytes
echo "Step 4: Removing bitsandbytes..."
pip uninstall bitsandbytes -y 2>/dev/null || echo "  (not installed)"
echo "  ✓ Removed"
echo ""

# Verify
echo "Step 5: Verifying PEFT works without bitsandbytes..."
python << 'PYEOF'
import sys

try:
    from peft import LoraConfig, get_peft_model
    print("  ✓ PEFT imports successfully")
    print("  ✓ All checks passed")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "==========================================="
echo "PATCH COMPLETE"
echo "==========================================="
echo ""
echo "PEFT has been patched to work without bitsandbytes."
echo "Original file backed up to: $PEFT_LORA_MODEL.backup"
echo ""
echo "Now start training:"
echo "  bash START_TRAINING_NOW.sh"
echo ""
