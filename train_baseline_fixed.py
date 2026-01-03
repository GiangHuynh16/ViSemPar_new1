#!/usr/bin/env python3
"""
Vietnamese AMR Parser - Baseline Training FIXED VERSION

Critical fixes:
1. ✅ Add EOS token to training data
2. ✅ Mask instruction part (only train on AMR output)
3. ✅ Clear Penman format in prompt
4. ✅ Prevent model from generating explanations

Run: python3 train_baseline_fixed.py --epochs 15
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'config'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_hf_login():
    """Check and ensure HuggingFace login before training"""
    try:
        from hf_auth import ensure_hf_login

        logger.info("\n" + "=" * 70)
        logger.info("CHECKING HUGGINGFACE LOGIN")
        logger.info("=" * 70)

        ensure_hf_login(require_write=False)

        logger.info("=" * 70 + "\n")
        return True

    except Exception as e:
        logger.warning(f"⚠️  HuggingFace login check failed: {e}")
        logger.warning("Continuing without HuggingFace login...")
        return False


def print_banner():
    """Print baseline training banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     VIETNAMESE AMR PARSER - BASELINE TRAINING FIXED         ║
    ║     ✅ EOS Token | ✅ Instruction Masking | ✅ Penman      ║
    ║                                                              ║
    ║  🎯 Standard Approach: Sentence → AMR                       ║
    ║  📊 For comparison with MTUP                                ║
    ║                                                              ║
    ║              VLSP 2025 - AMR Parsing                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"⚡ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 70)


def check_environment():
    """Check directories and data files"""
    from config_fixed import DATA_DIR, OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR

    logger.info("\n" + "=" * 70)
    logger.info("ENVIRONMENT CHECK")
    logger.info("=" * 70)

    # Check and create directories
    for dir_name, dir_path in [
        ("Data", DATA_DIR),
        ("Output", OUTPUT_DIR),
        ("Logs", LOG_DIR),
        ("Checkpoints", CHECKPOINT_DIR)
    ]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created {dir_name} directory: {dir_path}")
        else:
            logger.info(f"✓ {dir_name} directory exists: {dir_path}")

    # Check data files
    from config_fixed import DATA_CONFIG
    data_files = DATA_CONFIG['train_files']
    missing_files = []

    for file in data_files:
        file_path = DATA_DIR / file
        if not file_path.exists():
            missing_files.append(file)
            logger.warning(f"⚠️  Missing data file: {file}")
        else:
            logger.info(f"✓ Found data file: {file}")

    if missing_files:
        logger.error(f"\n❌ Missing {len(missing_files)} data file(s)")
        logger.error("Please ensure training data is in the data/ directory")
        sys.exit(1)

    logger.info("=" * 70)


def load_training_data(args):
    """Load and preprocess training data with FIXED format"""
    from config_fixed import DATA_DIR, DATA_CONFIG, PROMPT_TEMPLATE
    from data_loader import AMRDataLoader

    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOADING TRAINING DATA (FIXED FORMAT)")
    logger.info("=" * 70)

    # Load data
    train_files = DATA_CONFIG['train_files']
    all_examples = []

    # Create data loader
    loader = AMRDataLoader(DATA_DIR)

    for file in train_files:
        file_path = DATA_DIR / file
        examples = loader.parse_amr_file(file_path)
        all_examples.extend(examples)
        logger.info(f"✓ Loaded {len(examples)} examples from {file}")

    logger.info(f"\n✓ Total examples loaded: {len(all_examples)}")

    # Limit if max_samples specified
    if args.max_samples and args.max_samples < len(all_examples):
        all_examples = all_examples[:args.max_samples]
        logger.info(f"✓ Limited to {len(all_examples)} examples (--max-samples)")

    # Convert to baseline format with FIXES
    baseline_examples = []

    logger.info("\n" + "=" * 70)
    logger.info("APPLYING FIXES:")
    logger.info("  1. Adding EOS token to each example")
    logger.info("  2. Preparing for instruction masking")
    logger.info("  3. Using clear Penman format prompt")
    logger.info("=" * 70)

    for example in all_examples:
        sentence = example.get('sentence', '')
        amr = example.get('amr', '')

        if not sentence or not amr:
            continue

        # Create prompt (instruction part)
        prompt = PROMPT_TEMPLATE.format(sentence=sentence)

        # FIX 1: Add EOS token after AMR (will be added in dataset creation)
        # This tells model when to stop generating

        baseline_examples.append({
            'prompt': prompt,      # Instruction part (will be masked)
            'amr': amr,           # Output part (will be trained)
            'sentence': sentence,
        })

    logger.info(f"✓ Processed {len(baseline_examples)} baseline examples")

    # Split validation if requested
    val_examples = None
    if args.val_split > 0:
        split_idx = int(len(baseline_examples) * (1 - args.val_split))
        val_examples = baseline_examples[split_idx:]
        baseline_examples = baseline_examples[:split_idx]
        logger.info(f"\n✓ Train/Val split: {len(baseline_examples)}/{len(val_examples)}")

    # Show sample
    if baseline_examples and args.show_sample:
        logger.info("\n" + "=" * 70)
        logger.info("SAMPLE FIXED EXAMPLE")
        logger.info("=" * 70)
        sample = baseline_examples[0]
        logger.info("PROMPT (will be masked):")
        logger.info(sample['prompt'][:400] + "...")
        logger.info("\nAMR (will be trained):")
        logger.info(sample['amr'][:200] + "...")
        logger.info("=" * 70)

    return baseline_examples, val_examples


def create_baseline_dataset(examples: List[Dict], tokenizer, max_length: int):
    """Create PyTorch dataset with INSTRUCTION MASKING"""
    from torch.utils.data import Dataset

    class BaselineDatasetFixed(Dataset):
        def __init__(self, examples, tokenizer, max_length):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            example = self.examples[idx]

            # FIX 2: Add EOS token
            prompt = example['prompt']
            amr = example['amr']

            # Complete text with EOS token
            full_text = prompt + amr + self.tokenizer.eos_token

            # Tokenize full text
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = input_ids.clone()

            # FIX 3: INSTRUCTION MASKING
            # Only train on AMR output, not on instruction
            # Tokenize prompt separately to find where it ends
            prompt_encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            prompt_length = len(prompt_encoding['input_ids'][0])

            # Mask instruction part (set to -100)
            labels[:prompt_length] = -100

            # Mask padding tokens
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

    return BaselineDatasetFixed(examples, tokenizer, max_length)


def setup_model_and_tokenizer(args):
    """Setup model with LoRA for efficient fine-tuning"""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import LoraConfig, get_peft_model
    from config_fixed import MODEL_NAME, LORA_CONFIG

    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: LOADING MODEL AND TOKENIZER")
    logger.info("=" * 70)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"✓ Tokenizer loaded")
    logger.info(f"  Vocab size: {len(tokenizer)}")
    logger.info(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    logger.info(f"  PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # Load base model
    logger.info(f"\nLoading base model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"✓ Base model loaded")
    logger.info(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")

    # Apply LoRA
    logger.info("\nApplying LoRA configuration...")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        **LORA_CONFIG
    )

    model = get_peft_model(model, lora_config)

    # CRITICAL: Set model to training mode
    model.train()

    # IMPORTANT: DISABLE gradient checkpointing for LoRA compatibility
    # Gradient checkpointing causes "None of the inputs have requires_grad" error with LoRA
    # 7B model with batch_size=1 should fit in 48GB without checkpointing
    logger.info("⚠️  Gradient checkpointing DISABLED for LoRA compatibility")

    # Count trainable parameters (compatible with all peft versions)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"✓ LoRA applied")
    logger.info(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"  Total params: {total_params:,}")

    logger.info("=" * 70)

    return model, tokenizer


def train_model(model, tokenizer, train_dataset, val_dataset, args):
    """Train the model using Hugging Face Trainer"""
    from transformers import Trainer, TrainingArguments
    from config_fixed import TRAINING_CONFIG, OUTPUT_DIR

    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: TRAINING SETUP")
    logger.info("=" * 70)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"baseline_fixed_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Override config with args
    config = TRAINING_CONFIG.copy()
    if args.epochs:
        config['num_train_epochs'] = args.epochs
    if args.batch_size:
        config['per_device_train_batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=100 if val_dataset else None,
        **config
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION:")
    logger.info("=" * 70)
    logger.info(f"  Epochs: {config['num_train_epochs']}")
    logger.info(f"  Batch size: {config['per_device_train_batch_size']}")
    logger.info(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Warmup steps: {config['warmup_steps']}")
    logger.info(f"  Weight decay: {config['weight_decay']}")
    logger.info(f"  Max grad norm: {config['max_grad_norm']}")
    logger.info(f"  FP16: {config['fp16']}")
    logger.info(f"  BF16: {config['bf16']}")
    logger.info(f"  Optimizer: {config['optim']}")
    logger.info(f"  LR scheduler: {config['lr_scheduler_type']}")
    logger.info("=" * 70)

    # Train
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)

    train_result = trainer.train()

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info(f"  Final loss: {train_result.training_loss:.4f}")
    logger.info(f"  Training time: {train_result.metrics['train_runtime']:.2f}s")
    logger.info("=" * 70)

    # Save final model
    logger.info("\nSaving final model...")
    final_model_path = output_dir / "final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    logger.info(f"✓ Model saved to {final_model_path}")

    return trainer, output_dir


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train Vietnamese AMR Baseline Model - FIXED')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size per device')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--max-samples', type=int, help='Limit training samples for testing')
    parser.add_argument('--val-split', type=float, default=0.05, help='Validation split ratio')
    parser.add_argument('--show-sample', action='store_true', help='Show sample training example')
    parser.add_argument('--no-hf-login', action='store_true', help='Skip HuggingFace login check')

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Check HuggingFace login
    if not args.no_hf_login:
        check_hf_login()

    # Check environment
    check_environment()

    # Load data
    train_examples, val_examples = load_training_data(args)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)

    # Create datasets
    from config_fixed import MAX_SEQ_LENGTH

    logger.info("\n" + "=" * 70)
    logger.info("CREATING DATASETS WITH FIXES")
    logger.info("=" * 70)

    train_dataset = create_baseline_dataset(train_examples, tokenizer, MAX_SEQ_LENGTH)
    logger.info(f"✓ Train dataset: {len(train_dataset)} examples")

    val_dataset = None
    if val_examples:
        val_dataset = create_baseline_dataset(val_examples, tokenizer, MAX_SEQ_LENGTH)
        logger.info(f"✓ Val dataset: {len(val_dataset)} examples")

    logger.info("=" * 70)

    # Train
    trainer, output_dir = train_model(model, tokenizer, train_dataset, val_dataset, args)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETE - FIXES APPLIED:")
    logger.info("=" * 70)
    logger.info("  ✅ EOS token added to all training examples")
    logger.info("  ✅ Instruction masking enabled (only train on AMR)")
    logger.info("  ✅ Clear Penman format in prompt")
    logger.info("  ✅ Model should stop generating at EOS token")
    logger.info("=" * 70)
    logger.info(f"\n📁 Model saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Test model predictions on public_test.txt")
    logger.info("  2. Calculate SMATCH score")
    logger.info("  3. Compare with MTUP 7B results")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
