#!/bin/bash
# Check available CUDA libraries on the server

echo "==========================================="
echo "CHECK CUDA LIBRARIES"
echo "==========================================="
echo ""

echo "Step 1: Checking CUDA installations..."
for cuda_dir in /usr/local/cuda-11.8 /usr/local/cuda /usr/lib/x86_64-linux-gnu; do
    if [ -d "$cuda_dir" ]; then
        echo "  Found: $cuda_dir"
        if [ -d "$cuda_dir/lib64" ]; then
            echo "    → Has lib64 directory"
        fi
    fi
done
echo ""

echo "Step 2: Looking for libcusparse.so.11..."
find /usr/local /usr/lib -name "libcusparse.so*" 2>/dev/null | head -20
echo ""

echo "Step 3: Looking for libcudart.so..."
find /usr/local /usr/lib -name "libcudart.so*" 2>/dev/null | head -20
echo ""

echo "Step 4: Current LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH"
echo ""

echo "Step 5: Testing bitsandbytes import with different paths..."
echo ""

# Test 1: /usr/local/cuda/lib64
echo "Test 1: Using /usr/local/cuda/lib64"
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH python3 -c "import bitsandbytes; print('✓ Success')" 2>&1 | tail -5
echo ""

# Test 2: /usr/local/cuda-11.8/lib64
echo "Test 2: Using /usr/local/cuda-11.8/lib64"
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH python3 -c "import bitsandbytes; print('✓ Success')" 2>&1 | tail -5
echo ""

# Test 3: Both + system lib
echo "Test 3: Using both CUDA paths + system lib"
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH python3 -c "import bitsandbytes; print('✓ Success')" 2>&1 | tail -5
echo ""

echo "==========================================="
echo "CHECK COMPLETE"
echo "==========================================="
