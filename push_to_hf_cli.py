#!/usr/bin/env python3
"""
Push trained models to HuggingFace Hub using CLI login

Prerequisites:
    huggingface-cli login  # Run this once

Usage:
    python3 push_to_hf_cli.py --model-type mtup
    python3 push_to_hf_cli.py --model-type baseline
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from datetime import datetime


def find_latest_model(model_type="mtup"):
    """
    Auto-detect latest trained model

    Args:
        model_type: "mtup" or "baseline"

    Returns:
        Path to latest model or None
    """
    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        return None

    # Search patterns - check both old and new locations
    if model_type == "mtup":
        patterns = ["checkpoints_mtup/*_final", "mtup_*"]
    else:
        patterns = ["checkpoints/*", "baseline_*"]

    # Find all matching directories
    candidates = []
    for pattern in patterns:
        candidates.extend(outputs_dir.glob(pattern))

    if not candidates:
        return None

    # Sort by modification time (newest first)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Check each candidate for required files
    for candidate in candidates:
        # Check if adapter_model.safetensors or adapter_model.bin exists
        if (candidate / "adapter_model.safetensors").exists() or (candidate / "adapter_model.bin").exists():
            return candidate

        # Check subdirectories (e.g., final/, checkpoint-XXX/)
        if candidate.is_dir():
            for subdir in sorted(candidate.iterdir(), reverse=True):
                if subdir.is_dir():
                    if (subdir / "adapter_model.safetensors").exists() or (subdir / "adapter_model.bin").exists():
                        return subdir

    return None


def push_to_hf(model_type="mtup", model_path=None):
    """
    Push model to HuggingFace Hub

    Args:
        model_type: "mtup" or "baseline"
        model_path: Optional specific path to model

    Returns:
        True if successful, False otherwise
    """

    print("=" * 80)
    print(f"🚀 PUSH {model_type.upper()} MODEL TO HUGGINGFACE HUB")
    print("=" * 80)
    print()

    # Auto-detect model path if not provided
    if model_path is None:
        print(f"🔍 Auto-detecting latest {model_type.upper()} model...")
        model_path = find_latest_model(model_type)

        if model_path is None:
            print(f"❌ ERROR: No {model_type} model found in outputs/")
            print()
            print("Expected directories:")
            if model_type == "mtup":
                print("  - outputs/mtup_full_training_*/")
                print("  - outputs/mtup_fast_iteration_*/")
                print("  - outputs/mtup_*/")
            else:
                print("  - outputs/baseline_*/")
            print()
            print("Have you trained the model?")
            print(f"  python3 train_mtup.py --use-case full_training --model qwen2.5-7b")
            return False

        print(f"✅ Found model: {model_path}")
    else:
        model_path = Path(model_path)
        print(f"📁 Using specified path: {model_path}")

    if not model_path.exists():
        print(f"❌ ERROR: Model path does not exist: {model_path}")
        return False

    # Check required files (either .bin or .safetensors)
    has_model = (model_path / "adapter_model.bin").exists() or (model_path / "adapter_model.safetensors").exists()
    has_config = (model_path / "adapter_config.json").exists()

    missing_files = []
    if not has_model:
        missing_files.append("adapter_model.bin or adapter_model.safetensors")
    if not has_config:
        missing_files.append("adapter_config.json")

    if missing_files:
        print(f"❌ ERROR: Missing required files in {model_path}:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    print()

    # Get user info from CLI login
    print("🔐 Checking HuggingFace authentication...")
    try:
        api = HfApi()  # Uses credentials from 'huggingface-cli login'
        user = api.whoami()
        username = user['name']
        print(f"✅ Logged in as: {username}")
    except Exception as e:
        print(f"❌ ERROR: Not logged in to HuggingFace")
        print(f"   Error: {e}")
        print()
        print("Please login first:")
        print("   huggingface-cli login")
        print()
        print("Then run this script again.")
        return False

    # Repository configuration
    repo_name = f"vietnamese-amr-{model_type}-qwen"
    full_repo_name = f"{username}/{repo_name}"

    print()
    print("=" * 80)
    print("UPLOAD CONFIGURATION")
    print("=" * 80)
    print(f"📁 Local path:  {model_path}")
    print(f"👤 Username:    {username}")
    print(f"📦 Repository:  {full_repo_name}")
    print(f"🔐 Visibility:  Private")
    print()

    try:
        # Create repository
        print("📦 Creating repository...")
        create_repo(
            repo_id=repo_name,
            private=True,  # Private repo
            exist_ok=True,  # Don't error if exists
            repo_type="model"
        )
        print(f"✅ Repository ready: {full_repo_name}")
        print()

        # Create model card
        print("📝 Creating model card...")

        # Build model card sections
        approach = "Two-Task Decomposition (MTUP)" if model_type == "mtup" else "Single-Task Direct Generation"
        lora_rank = 64 if model_type == "mtup" else 128
        trainable_params = "~67M (LoRA adapters)" if model_type == "mtup" else "~134M (LoRA adapters)"
        expected_f1 = "~0.49-0.53" if model_type == "mtup" else "~0.42-0.46"

        model_card = f"""---
language:
- vi
license: apache-2.0
tags:
- amr
- semantic-parsing
- vietnamese
- qwen2.5
- lora
library_name: peft
base_model: Qwen/Qwen2.5-7B-Instruct
---

# Vietnamese AMR Parser - {model_type.upper()}

LoRA adapter for Vietnamese Abstract Meaning Representation (AMR) parsing.

## Model Details

- **Base Model**: Qwen/Qwen2.5-7B-Instruct (7.62B parameters)
- **Approach**: {approach}
- **LoRA Rank**: {lora_rank}
- **Trainable Parameters**: {trainable_params}
- **Framework**: PEFT (Parameter-Efficient Fine-Tuning)
- **Language**: Vietnamese
- **Task**: AMR Parsing
- **Uploaded**: {datetime.now().strftime("%Y-%m-%d")}

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{full_repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{full_repo_name}")

# Parse Vietnamese sentence
sentence = "Tôi yêu Việt Nam"
prompt = \"\"\"### NHIỆM VỤ
Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### CÂU ĐẦU VÀO
{{sentence}}

### KẾT QUẢ

## BƯỚC 1: Cấu trúc AMR (chưa có biến)
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=512)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Performance

- **Expected F1 Score**: {expected_f1}
- **Evaluation Metric**: SMATCH
- **Test Set**: Vietnamese AMR corpus (150 examples)

## Training Details

- **Training Data**: Vietnamese AMR dataset (~2500 examples)
- **Optimizer**: AdamW 8-bit
- **Learning Rate**: 2e-4 with cosine schedule
- **Batch Size**: 16 (effective)
- **Epochs**: 10
- **Hardware**: NVIDIA GPU with 24GB VRAM

## Citation

```bibtex
@misc{{vietnamese-amr-{model_type}-2025,
  title = {{Vietnamese AMR Parser ({model_type.upper()})}},
  author = {{{username}}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{full_repo_name}}}
}}
```

## License

Apache 2.0

## Contact

For questions or issues, please open an issue on the model repository.
"""

        # Save model card
        readme_path = model_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        print("✅ Model card created")
        print()

        # Upload files
        print("📤 Uploading files to HuggingFace Hub...")
        print("   This may take 2-3 minutes depending on connection...")
        print()

        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            repo_type="model",
            ignore_patterns=[
                "checkpoint-*",
                "*.log",
                "runs/",
                "__pycache__/",
                "*.pyc",
                ".git/",
                ".gitignore"
            ]
        )

        print()
        print("=" * 80)
        print("✅ SUCCESS! MODEL UPLOADED TO HUGGINGFACE HUB")
        print("=" * 80)
        print()
        print(f"🔗 Model URL:")
        print(f"   https://huggingface.co/{full_repo_name}")
        print()
        print("📥 To use on your local machine:")
        print()
        print(f"""from peft import PeftModel
from transformers import AutoModelForCausalLM

model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct"),
    "{full_repo_name}"
)
""")
        print()
        print("💡 TIP: You can now delete the model from server to save space:")
        print(f"   rm -rf {model_path}")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        print("Common issues:")
        print("1. Network connection problem")
        print("2. Insufficient permissions")
        print("3. Repository already exists (should be OK with exist_ok=True)")
        print()
        print("If error persists, try:")
        print("1. huggingface-cli login  # Re-login")
        print("2. Check network: ping huggingface.co")
        print("3. Check model files are complete")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Push trained model to HuggingFace Hub using CLI login",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and push latest MTUP model
  python3 push_to_hf_cli.py --model-type mtup

  # Push latest Baseline model
  python3 push_to_hf_cli.py --model-type baseline

  # Push specific model path
  python3 push_to_hf_cli.py --model-type mtup --model-path outputs/mtup_full_training_20250126

Prerequisites:
  Must login to HuggingFace CLI first:
    huggingface-cli login

  Then paste your token (get from https://huggingface.co/settings/tokens)
"""
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["mtup", "baseline"],
        required=True,
        help="Which model to push: mtup or baseline"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional: Specific model path (auto-detects if not provided)"
    )

    args = parser.parse_args()

    # Push model
    success = push_to_hf(args.model_type, args.model_path)

    if success:
        print("🎉 All done! Model is now on HuggingFace Hub!")
        print()
    else:
        print("❌ Push failed. Please check the errors above.")
        print()
        exit(1)


if __name__ == "__main__":
    main()
