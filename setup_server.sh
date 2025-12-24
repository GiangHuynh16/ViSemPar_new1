#!/bin/bash
# Setup Script for Vietnamese AMR Parser on Server
# Run this script on the server after cloning the repository

set -e  # Exit on error

echo "======================================================================"
echo "Vietnamese AMR Parser - Server Setup"
echo "======================================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check Python version
echo -e "\n${YELLOW}Step 1: Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo -e "${RED}Error: Python 3.8+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"

# Step 2: Create directories
echo -e "\n${YELLOW}Step 2: Creating required directories...${NC}"
mkdir -p data outputs logs outputs/checkpoints outputs/checkpoints_mtup
echo -e "${GREEN}✓ Directories created${NC}"

# Step 3: Install dependencies
echo -e "\n${YELLOW}Step 3: Installing dependencies...${NC}"
echo "This may take a few minutes..."

if command -v conda &> /dev/null; then
    echo "Using conda environment..."
    # Assume conda env is already activated
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Using system Python..."
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
fi
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 4: Setup Hugging Face
echo -e "\n${YELLOW}Step 4: Setting up Hugging Face...${NC}"
pip install --upgrade huggingface_hub

echo ""
echo "Choose Hugging Face setup method:"
echo "1. CLI Login (Recommended - most secure)"
echo "2. Environment Variable (manual setup)"
echo "3. Skip (setup later)"
read -p "Enter choice [1-3]: " hf_choice

case $hf_choice in
    1)
        echo "Starting Hugging Face CLI login..."
        echo "You'll need your access token from: https://huggingface.co/settings/tokens"
        huggingface-cli login
        echo -e "${GREEN}✓ Hugging Face CLI login complete${NC}"
        ;;
    2)
        echo ""
        read -p "Enter your Hugging Face token: " hf_token
        echo "export HF_TOKEN=\"$hf_token\"" >> ~/.bashrc
        export HF_TOKEN="$hf_token"
        echo -e "${GREEN}✓ HF_TOKEN added to ~/.bashrc${NC}"
        echo -e "${YELLOW}Run 'source ~/.bashrc' to activate${NC}"
        ;;
    3)
        echo -e "${YELLOW}⚠ Skipping Hugging Face setup${NC}"
        echo "You can setup later with: huggingface-cli login"
        ;;
    *)
        echo -e "${RED}Invalid choice. Skipping Hugging Face setup.${NC}"
        ;;
esac

# Step 5: Verify installation
echo -e "\n${YELLOW}Step 5: Verifying installation...${NC}"

echo "Checking critical packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo -e "${RED}PyTorch not found${NC}"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || echo -e "${RED}Transformers not found${NC}"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

echo -e "${GREEN}✓ Installation verified${NC}"

# Step 6: Check data files
echo -e "\n${YELLOW}Step 6: Checking data files...${NC}"
if [ -f "data/train_amr_1.txt" ] && [ -f "data/train_amr_2.txt" ]; then
    echo -e "${GREEN}✓ Training data found${NC}"
else
    echo -e "${YELLOW}⚠ Training data not found in data/directory${NC}"
    echo "Please copy your training files:"
    echo "  - data/train_amr_1.txt"
    echo "  - data/train_amr_2.txt"
fi

# Step 7: Test imports
echo -e "\n${YELLOW}Step 7: Testing project modules...${NC}"
python3 -c "from src.preprocessor_mtup import MTUPAMRPreprocessor; print('✓ MTUP Preprocessor OK')" || echo -e "${RED}✗ MTUP Preprocessor failed${NC}"
python3 -c "from config.config_mtup import MODEL_NAME; print(f'✓ Config OK - Model: {MODEL_NAME}')" || echo -e "${RED}✗ Config failed${NC}"

# Summary
echo ""
echo "======================================================================"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy training data to data/ directory (if not done)"
echo ""
echo "2. Start training:"
echo "   ${YELLOW}# Quick test${NC}"
echo "   python3 train_mtup.py --use-case quick_test"
echo ""
echo "   ${YELLOW}# Fast iteration (recommended)${NC}"
echo "   python3 train_mtup.py --use-case fast_iteration"
echo ""
echo "   ${YELLOW}# With tmux (for persistent sessions)${NC}"
echo "   tmux new -s amr-training"
echo "   python3 train_mtup.py --use-case fast_iteration"
echo "   # Press Ctrl+B, then D to detach"
echo ""
echo "3. Monitor training:"
echo "   tmux attach -t amr-training"
echo "   tail -f logs/training_mtup.log"
echo ""
echo "For more info, see:"
echo "  - README.md"
echo "  - MTUP_IMPLEMENTATION.md"
echo "  - HUGGINGFACE_SETUP.md"
echo ""
echo "======================================================================"
