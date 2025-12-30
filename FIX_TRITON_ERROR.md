# Fix: ModuleNotFoundError: No module named 'triton.ops'

**Error**:
```
ModuleNotFoundError: No module named 'triton.ops'
```

**Root Cause**: Version conflict between `triton` and `bitsandbytes`. Even though we don't use quantization, PEFT tries to import bitsandbytes during LoRA initialization.

---

## Solution 1: Downgrade bitsandbytes (Recommended)

```bash
# Activate environment
conda activate lora_py310

# Uninstall current bitsandbytes
pip uninstall bitsandbytes -y

# Install compatible version
pip install bitsandbytes==0.41.1

# Verify
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

---

## Solution 2: Install Compatible Triton

```bash
# Activate environment
conda activate lora_py310

# Uninstall current triton
pip uninstall triton -y

# Install compatible version
pip install triton==2.0.0

# Verify
python -c "import triton; print(triton.__version__)"
```

---

## Solution 3: Fresh Environment with Correct Versions

If above solutions don't work, create fresh environment:

```bash
# Create new environment
conda create -n baseline_7b_fixed python=3.10 -y
conda activate baseline_7b_fixed

# Install PyTorch first
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install compatible versions
pip install transformers==4.36.0
pip install peft==0.6.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.1
pip install triton==2.0.0

# Install other packages
pip install datasets==2.14.5
pip install smatch==1.0.4
pip install penman==1.3.0
pip install tqdm

# Verify installation
python -c "
import torch
import transformers
import peft
import bitsandbytes
import triton
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'Bitsandbytes: {bitsandbytes.__version__}')
print(f'Triton: {triton.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

---

## Solution 4: Use Environment Without Bitsandbytes

Modify PEFT to avoid bitsandbytes import:

```bash
# Activate environment
conda activate lora_py310

# Uninstall bitsandbytes completely
pip uninstall bitsandbytes -y

# Install PEFT from source with modification (optional)
# Or just ensure we don't trigger bitsandbytes import
```

Then verify that `USE_4BIT_QUANTIZATION = False` in config.

---

## Quick Fix Commands (Run on Server)

```bash
# SSH to server
ssh islabworker2@islab-server2

# Navigate to project
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Activate environment
conda activate lora_py310

# Fix 1: Downgrade bitsandbytes
pip uninstall bitsandbytes -y
pip install bitsandbytes==0.41.1

# Fix 2: Install compatible triton
pip uninstall triton -y
pip install triton==2.0.0

# Verify
python -c "import bitsandbytes; import triton; print('OK')"

# Try training again
bash START_BASELINE_7B_TRAINING.sh
```

---

## Expected Package Versions (Working Configuration)

```
Python: 3.10
PyTorch: 2.1.0+cu118
Transformers: 4.36.0
PEFT: 0.6.0
Bitsandbytes: 0.41.1
Triton: 2.0.0
Accelerate: 0.25.0
```

---

## After Fix

Run training:

```bash
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
conda activate lora_py310
bash VERIFY_AND_START.sh
```
