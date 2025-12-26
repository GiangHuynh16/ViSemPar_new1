#!/usr/bin/env python3
"""
Push trained models to HuggingFace Hub - AUTO-DETECT VERSION

Automatically finds the latest trained model and pushes it.

Usage:
1. Copy .env.example to .env
2. Edit .env and add your HF_TOKEN
3. Run: python3 push_to_hf_auto.py --model-type mtup
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
from dotenv import load_dotenv
import argparse
from datetime import datetime

# Load .env file
load_dotenv()


def find_latest_model(model_type="mtup"):
    """
    Auto-detect latest trained model

    Args:
        model_type: "mtup" or "baseline"

    Returns:
        Path to latest model or None
    """
    outputs_dir = Path("outputs")

    if model_type == "mtup":
        # Look for mtup_* directories
        pattern = "mtup_*"
    else:
        # Look for baseline directories
        pattern = "baseline_*"

    # Find all matching directories
    candidates = list(outputs_dir.glob(pattern))

    if not candidates:
        return None

    # Sort by modification time (newest first)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Return the most recent one
    latest = candidates[0]

    # Check if it has the required files
    if (latest / "adapter_model.bin").exists():
        return latest

    # If not, try subdirectories (e.g., final/)
    subdirs = [d for d in latest.iterdir() if d.is_dir()]
    for subdir in subdirs:
        if (subdir / "adapter_model.bin").exists():
            return subdir

    return None


def push_to_hf(model_type="mtup", model_path=None):
    """
    Push model to HuggingFace

    Args:
        model_type: "mtup" or "baseline"
        model_path: Override auto-detection with specific path
    """

    # Get config from .env
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    make_private = os.getenv("MAKE_PRIVATE", "true").lower() == "true"

    if model_type == "mtup":
        repo_name = os.getenv("HF_REPO_MTUP", "vietnamese-amr-mtup-7b")
    else:
        repo_name = os.getenv("HF_REPO_BASELINE", "vietnamese-amr-baseline-7b")

    # Validate token
    if not hf_token or hf_token == "hf_your_token_here":
        print("❌ ERROR: HF_TOKEN not set in .env file")
        print()
        print("Steps to fix:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a token with 'write' permission")
        print("3. Copy .env.example to .env")
        print("4. Edit .env and paste your token")
        print()
        return False

    if not hf_username or hf_username == "your_username":
        print("❌ ERROR: HF_USERNAME not set in .env file")
        print("   Please edit .env and add your HuggingFace username")
        return False

    # Auto-detect model path if not provided
    if model_path is None:
        print(f"🔍 Auto-detecting latest {model_type.upper()} model...")
        model_path = find_latest_model(model_type)

        if model_path is None:
            print(f"❌ ERROR: No {model_type} model found in outputs/")
            print()
            print("Expected patterns:")
            if model_type == "mtup":
                print("  - outputs/mtup_*/")
                print("  - outputs/mtup_full_training_*/")
            else:
                print("  - outputs/baseline_*/")
            print()
            print("Have you trained the model yet?")
            print(f"  python3 train_mtup.py --use-case full_training --model qwen2.5-7b")
            return False

        print(f"✅ Found model: {model_path}")
    else:
        model_path = Path(model_path)
        print(f"📁 Using specified path: {model_path}")

    if not model_path.exists():
        print(f"❌ ERROR: Model path does not exist: {model_path}")
        return False

    # Check required files
    required_files = ["adapter_model.bin", "adapter_config.json"]
    for file in required_files:
        if not (model_path / file).exists():
            print(f"❌ ERROR: Missing {file} in {model_path}")
            return False

    print()
    print("=" * 80)
    print(f"🚀 PUSHING {model_type.upper()} MODEL TO HUGGINGFACE HUB")
    print("=" * 80)
    print()
    print(f"📁 Local path: {model_path}")
    print(f"👤 Username:   {hf_username}")
    print(f"📦 Repo name:  {repo_name}")
    print(f"🔐 Private:    {make_private}")
    print()

    try:
        # Login with token from .env
        print("🔐 Logging in to HuggingFace...")
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Logged in successfully!")
        print()

        # Initialize API
        api = HfApi()
        full_repo_name = f"{hf_username}/{repo_name}"

        # Create repo
        print(f"📦 Creating repository: {full_repo_name}...")
        try:
            create_repo(
                repo_id=repo_name,
                private=make_private,
                exist_ok=True,
                repo_type="model",
                token=hf_token
            )
            print(f"✅ Repository ready!")
        except Exception as e:
            print(f"⚠️  Repository may already exist: {e}")

        print()

        # Create model card
        print("📝 Creating model card...")
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

LoRA adapter for Vietnamese Abstract Meaning Representation parsing.

## Model Details

- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Approach**: {"Two-Task Decomposition (MTUP)" if model_type == "mtup" else "Single-Task Direct Generation"}
- **LoRA Rank**: {64 if model_type == "mtup" else 128}
- **Framework**: PEFT
- **Trained**: {datetime.now().strftime("%Y-%m-%d")}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base_model, "{full_repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{full_repo_name}")

# Generate AMR
sentence = "Tôi yêu Việt Nam"
prompt = f\"\"\"### NHIỆM VỤ
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

Expected F1: {" ~0.49-0.53" if model_type == "mtup" else "~0.42-0.46"}

## Citation

```bibtex
@misc{{vietnamese-amr-{model_type}-2025,
  title = {{Vietnamese AMR Parser ({model_type.upper()})}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{full_repo_name}}}
}}
```

## License

Apache 2.0
"""

        readme_path = model_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        print("✅ Model card created")
        print()

        # Upload
        print("📤 Uploading files to HuggingFace Hub...")
        print("   This may take 2-3 minutes...")
        print()

        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            repo_type="model",
            token=hf_token,
            ignore_patterns=["checkpoint-*", "*.log", "runs/", "__pycache__", "*.pyc"]
        )

        print()
        print("=" * 80)
        print("✅ SUCCESS! MODEL PUSHED TO HUGGINGFACE HUB")
        print("=" * 80)
        print()
        print(f"🔗 Model URL: https://huggingface.co/{full_repo_name}")
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
        print("✅ You can now delete the model from server to save space!")
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
        print("1. Invalid HF_TOKEN → Check .env file")
        print("2. No internet → Check connection")
        print("3. Model files missing → Check model directory")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Push model to HuggingFace Hub (auto-detect version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and push latest MTUP model
  python3 push_to_hf_auto.py --model-type mtup

  # Push specific model path
  python3 push_to_hf_auto.py --model-type mtup --model-path outputs/mtup_full_training_20250126

Setup:
  1. Copy .env.example to .env
  2. Edit .env and add your HF_TOKEN from https://huggingface.co/settings/tokens
  3. Run this script!
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
        help="Override auto-detection with specific model path"
    )

    args = parser.parse_args()

    # Check .env exists
    if not Path(".env").exists():
        print("⚠️  .env file not found!")
        print()
        print("Creating .env from .env.example...")

        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ Created .env file")
            print()
            print("📝 NOW EDIT .env FILE:")
            print("   1. Open .env in editor")
            print("   2. Replace 'hf_your_token_here' with your actual token")
            print("   3. Replace 'your_username' with your HF username")
            print("   4. Run this script again")
            print()
            return
        else:
            print("❌ .env.example not found!")
            return

    # Push model
    success = push_to_hf(args.model_type, args.model_path)

    if success:
        print("🎉 All done! Model is now on HuggingFace Hub!")
    else:
        print("❌ Push failed. Please check the errors above.")


if __name__ == "__main__":
    main()
