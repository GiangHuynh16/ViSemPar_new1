#!/usr/bin/env python3
"""
Add tokenizer to existing LoRA checkpoint for HuggingFace upload
"""

import sys
import argparse
from pathlib import Path
from transformers import AutoTokenizer


def add_tokenizer_to_checkpoint(checkpoint_path: str):
    """Copy tokenizer files to checkpoint directory"""

    checkpoint_dir = Path(checkpoint_path)

    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("=" * 70)
    print("ADD TOKENIZER TO LORA CHECKPOINT")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Base model
    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    # Load tokenizer from base model
    print(f"Loading tokenizer from: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True
    )

    # Save to checkpoint directory
    print(f"Saving tokenizer to: {checkpoint_path}")
    tokenizer.save_pretrained(str(checkpoint_dir))

    print()
    print("=" * 70)
    print("✅ TOKENIZER ADDED")
    print("=" * 70)
    print()
    print("Files in checkpoint:")
    for f in sorted(checkpoint_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name:40} ({size:,} bytes)")
    print()
    print("✅ Checkpoint is now ready for HuggingFace upload!")
    print()
    print("To upload:")
    print(f"  huggingface-cli upload <your-username>/model-name {checkpoint_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Add tokenizer to LoRA checkpoint'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to LoRA checkpoint directory'
    )

    args = parser.parse_args()

    add_tokenizer_to_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
