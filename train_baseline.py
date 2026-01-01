#!/usr/bin/env python3
"""
Vietnamese AMR Parser - Baseline Training (Single-Task)

Standard single-task learning approach:
- Direct sentence → AMR mapping
- No multi-task decomposition
- For fair comparison with MTUP

Run: python3 train_baseline.py --epochs 15
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
    ║     VIETNAMESE AMR PARSER - BASELINE TRAINING               ║
    ║     Single-Task Direct Mapping                              ║
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
    from config import DATA_DIR, OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR

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
    from config import DATA_CONFIG
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
    """Load and preprocess training data for baseline (single-task)"""
    from config import DATA_DIR, DATA_CONFIG, PROMPT_TEMPLATE
    from data_loader import AMRDataLoader

    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOADING TRAINING DATA (BASELINE FORMAT)")
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

    # Convert to baseline format (simple sentence → AMR mapping)
    baseline_examples = []
    for example in all_examples:
        sentence = example.get('sentence', '')
        amr = example.get('amr', '')

        if not sentence or not amr:
            continue

        # Create baseline training example using prompt template
        text = PROMPT_TEMPLATE.format(sentence=sentence) + amr

        baseline_examples.append({
            'text': text,
            'sentence': sentence,
            'amr': amr
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
        logger.info("SAMPLE BASELINE EXAMPLE")
        logger.info("=" * 70)
        logger.info(baseline_examples[0]['text'][:600] + "...")
        logger.info("=" * 70)

    return baseline_examples, val_examples


def create_baseline_dataset(examples: List[Dict], tokenizer, max_length: int):
    """Create PyTorch dataset from baseline examples"""
    from torch.utils.data import Dataset

    class BaselineDataset(Dataset):
        def __init__(self, examples, tokenizer, max_length):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            example = self.examples[idx]

            # Tokenize
            encoding = self.tokenizer(
                example['text'],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = input_ids.clone()

            # CRITICAL: Mask padding tokens in labels to prevent learning on padding
            # Without this, model trains on padding tokens → loss = 0, grad_norm = NaN
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

    return BaselineDataset(examples, tokenizer, max_length)


def setup_model_and_tokenizer(args):
    """Setup model with LoRA for efficient fine-tuning"""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from config import MODEL_NAME, LORA_CONFIG, USE_4BIT_QUANTIZATION

    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: LOADING MODEL AND TOKENIZER")
    logger.info("=" * 70)

    logger.info(f"Model: {MODEL_NAME}")

    # Quantization disabled (same as MTUP 7B)
    use_quantization = USE_4BIT_QUANTIZATION and not args.no_quantize
    if args.no_quantize:
        logger.warning("⚠️  Quantization DISABLED by --no-quantize flag")
    logger.info(f"Using 4-bit quantization: {use_quantization}")

    # Load tokenizer
    logger.info("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Set pad_token = eos_token")

    logger.info(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Setup quantization config
    quantization_config = None
    if use_quantization and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        logger.info("✓ 4-bit quantization config created")

    # Load model
    logger.info("\nLoading model...")

    # CRITICAL FIX: device_map + gradient_checkpointing causes "expected device meta but got cuda:0" error
    # Solution: Load model DIRECTLY on GPU without device_map, use gradient checkpointing instead
    #
    # Memory strategy:
    # - Load full model in FP16: ~14GB
    # - LoRA adapters: ~0.5GB
    # - Gradient checkpointing: saves activation memory
    # - batch_size=1, seq=512: minimal activation footprint
    # Total: Should fit in 24GB VRAM

    if not use_quantization and torch.cuda.is_available():
        logger.info("⚠️  Loading model DIRECTLY on GPU (no device_map, no CPU offload)")
        logger.info("   Memory strategy: FP16 + gradient checkpointing + batch_size=1")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=None,
            device_map=None,  # DISABLED - load to GPU directly
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        # Move to GPU explicitly
        model = model.to("cuda:0")
        logger.info(f"✓ Model loaded on GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config if use_quantization else None,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        logger.info(f"✓ Model loaded")

    # NOTE: gradient_checkpointing_enable() must be called AFTER LoRA is applied
    # Otherwise LoRA parameters won't have proper gradient tracking

    # Prepare model for LoRA training
    # For non-quantized models, we need to:
    # 1. Cast layer norms to fp32 for stability
    # 2. Freeze base model parameters (only train LoRA)
    if not use_quantization:
        # Cast LayerNorm to fp32 for training stability
        for name, module in model.named_modules():
            if "norm" in name.lower() or isinstance(module, torch.nn.LayerNorm):
                module = module.to(torch.float32)
        logger.info("✓ Cast LayerNorm modules to fp32 for stability")

        # Freeze all base model parameters
        for param in model.parameters():
            param.requires_grad = False
        logger.info("✓ Froze base model parameters (will unfreeze LoRA params after applying LoRA)")

    # Prepare for k-bit training if quantized
    if use_quantization and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
        logger.info("✓ Model prepared for k-bit training")

    # Setup LoRA
    logger.info("\nApplying LoRA...")
    lora_config = LoraConfig(
        r=LORA_CONFIG['r'],
        lora_alpha=LORA_CONFIG['lora_alpha'],
        target_modules=LORA_CONFIG.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
        lora_dropout=LORA_CONFIG.get('lora_dropout', 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA - this will automatically set requires_grad=True for LoRA params
    model = get_peft_model(model, lora_config)
    logger.info("✓ LoRA applied")

    # Verify LoRA parameters have gradients enabled
    lora_params_count = sum(1 for name, param in model.named_parameters() if param.requires_grad and 'lora_' in name)
    if lora_params_count == 0:
        logger.warning("⚠️  No LoRA parameters have requires_grad=True! Manually enabling...")
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
    logger.info(f"✓ LoRA parameters with gradients: {lora_params_count}")

    # CRITICAL: Set model to training mode BEFORE enabling gradient checkpointing
    # This ensures parameters are properly initialized for gradient computation
    model.train()
    logger.info("✓ Model set to training mode")

    # CRITICAL: Enable gradient checkpointing AFTER LoRA is applied AND model.train()
    # This ensures LoRA parameters have proper gradient tracking
    model.gradient_checkpointing_enable()
    logger.info("✓ Gradient checkpointing enabled (reduces memory usage)")

    model.print_trainable_parameters()

    return model, tokenizer


def train_baseline_model(model, tokenizer, train_dataset, val_dataset, args):
    """Train model with baseline (single-task) approach"""
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from config import OUTPUT_DIR, CHECKPOINT_DIR, TRAINING_CONFIG

    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: TRAINING WITH BASELINE APPROACH")
    logger.info("=" * 70)

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"baseline_{timestamp}"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs or TRAINING_CONFIG.get('num_train_epochs', 15),
        per_device_train_batch_size=args.batch_size or TRAINING_CONFIG.get('per_device_train_batch_size', 2),
        per_device_eval_batch_size=args.batch_size or TRAINING_CONFIG.get('per_device_train_batch_size', 2),
        gradient_accumulation_steps=args.grad_accum or TRAINING_CONFIG.get('gradient_accumulation_steps', 8),
        learning_rate=args.lr or TRAINING_CONFIG.get('learning_rate', 2e-4),
        weight_decay=TRAINING_CONFIG.get('weight_decay', 0.01),
        warmup_steps=TRAINING_CONFIG.get('warmup_steps', 100),
        logging_steps=args.log_steps or TRAINING_CONFIG.get('logging_steps', 10),
        save_steps=args.save_steps or TRAINING_CONFIG.get('save_steps', 200),
        eval_steps=args.eval_steps or TRAINING_CONFIG.get('save_steps', 200) if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        save_total_limit=TRAINING_CONFIG.get('save_total_limit', 5),
        load_best_model_at_end=TRAINING_CONFIG.get('load_best_model_at_end', True) if val_dataset else False,
        metric_for_best_model=TRAINING_CONFIG.get('metric_for_best_model', 'loss') if val_dataset else None,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,  # Disable for non-quantized mode (SAME AS MTUP)
        optim=TRAINING_CONFIG.get('optim', 'adamw_torch'),
        lr_scheduler_type=TRAINING_CONFIG.get('lr_scheduler_type', 'cosine'),
        report_to=["tensorboard"],
        logging_dir=str(OUTPUT_DIR / "logs" / f"baseline_{timestamp}"),
    )

    logger.info("Training Configuration:")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Total steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    logger.info(f"  🔍 FP16: {training_args.fp16}")
    logger.info(f"  🔍 BF16: {training_args.bf16}")
    logger.info(f"  Gradient checkpointing: {model.is_gradient_checkpointing if hasattr(model, 'is_gradient_checkpointing') else 'enabled'}")

    # Data collator - Use default collator since we already set labels in dataset
    # DataCollatorForLanguageModeling can cause issues with pre-set labels
    from transformers import default_data_collator
    data_collator = default_data_collator

    # NOTE: Monkey-patch no longer needed since we're not using device_map
    # Model is loaded directly on GPU, so Trainer can move it normally

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset else None,
        data_collator=data_collator,
    )

    # Train
    logger.info("\n🚀 Starting training...")
    logger.info("=" * 70)

    train_result = trainer.train()

    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info(f"Training time: {train_result.metrics['train_runtime']:.2f}s")
    logger.info(f"Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

    # Save final model
    final_model_path = CHECKPOINT_DIR / "baseline_7b_final"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n💾 Saving model to: {final_model_path.absolute()}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Verify model files exist
    required_files = ["adapter_model.safetensors", "adapter_config.json", "tokenizer_config.json"]
    missing = []
    for file in required_files:
        if not (final_model_path / file).exists():
            missing.append(file)

    if missing:
        logger.warning(f"⚠️  Missing files: {', '.join(missing)}")
    else:
        logger.info("✓ All model files saved successfully")

    logger.info("\n" + "=" * 70)
    logger.info("NEXT STEPS")
    logger.info("=" * 70)
    logger.info(f"\n1. Evaluate model:")
    logger.info(f"   python evaluate_baseline_model.py \\")
    logger.info(f"     --checkpoint {final_model_path} \\")
    logger.info(f"     --test-file data/public_test_ground_truth.txt \\")
    logger.info(f"     --output results/baseline_7b_evaluation.json")
    logger.info(f"\n2. Compare with MTUP:")
    logger.info(f"   python compare_results.py \\")
    logger.info(f"     --baseline results/baseline_7b_evaluation.json \\")
    logger.info(f"     --mtup results/mtup_7b_evaluation.json")
    logger.info("\n" + "=" * 70)

    return trainer, final_model_path


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Vietnamese AMR Baseline Training")

    # Training parameters
    parser.add_argument('--epochs', type=int, help="Number of training epochs (default: 15)")
    parser.add_argument('--batch-size', type=int, help="Per-device batch size (default: 2)")
    parser.add_argument('--grad-accum', type=int, help="Gradient accumulation steps (default: 8)")
    parser.add_argument('--lr', type=float, help="Learning rate (default: 2e-4)")
    parser.add_argument('--max-samples', type=int, help="Limit training samples")
    parser.add_argument('--val-split', type=float, default=0.1, help="Validation split ratio (default: 0.1)")

    # Model parameters
    parser.add_argument('--no-quantize', action='store_true', help="Disable 4-bit quantization")
    parser.add_argument('--max-seq-length', type=int, default=2048, help="Maximum sequence length")

    # Logging
    parser.add_argument('--log-steps', type=int, help="Logging frequency")
    parser.add_argument('--save-steps', type=int, help="Checkpoint save frequency")
    parser.add_argument('--eval-steps', type=int, help="Evaluation frequency")
    parser.add_argument('--show-sample', action='store_true', help="Show sample training example")

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Check HuggingFace login (non-blocking)
    check_hf_login()

    # Check environment
    check_environment()

    # Load data
    train_examples, val_examples = load_training_data(args)

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(args)

    # Create datasets
    from config import MAX_SEQ_LENGTH
    max_length = args.max_seq_length or MAX_SEQ_LENGTH

    logger.info(f"\nCreating datasets (max_length={max_length})...")
    train_dataset = create_baseline_dataset(train_examples, tokenizer, max_length)
    val_dataset = create_baseline_dataset(val_examples, tokenizer, max_length) if val_examples else None
    logger.info(f"✓ Train dataset: {len(train_dataset)} examples")
    if val_dataset:
        logger.info(f"✓ Val dataset: {len(val_dataset)} examples")

    # Train
    trainer, model_path = train_baseline_model(model, tokenizer, train_dataset, val_dataset, args)

    logger.info("\n" + "=" * 70)
    logger.info("🎉 BASELINE TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\n✓ Model saved to: {model_path}")
    logger.info("\n")


if __name__ == "__main__":
    main()
