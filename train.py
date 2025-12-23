"""
Vietnamese AMR Parser - Unified Training Script
Combines training, validation, and inference in one file
Fixes tensor dimension mismatch and avoids automatic file creation

Run with: python train.py
Options: python train.py --help
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'config'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         VIETNAMESE AMR PARSER - UNIFIED TRAINING            ║
    ║                                                              ║
    ║              Abstract Meaning Representation                 ║
    ║                     for Vietnamese                          ║
    ║                                                              ║
    ║                      VLSP 2025                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"⚡ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 70)


def check_directories():
    """Check if required directories exist"""
    from config import DATA_DIR, OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR
    
    missing_dirs = []
    for dir_path in [DATA_DIR, OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR]:
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        logger.error("❌ Missing required directories:")
        for dir_path in missing_dirs:
            logger.error(f"   {dir_path}")
        logger.error("\n💡 Create them with:")
        logger.error("   mkdir -p data outputs logs outputs/checkpoints")
        sys.exit(1)
    
    logger.info("✓ All required directories exist")


def check_data_files():
    """Check if training data files exist"""
    from config import DATA_DIR, DATA_CONFIG
    
    missing_files = []
    for train_file in DATA_CONFIG['train_files']:
        file_path = DATA_DIR / train_file
        if not file_path.exists():
            missing_files.append(train_file)
    
    if missing_files:
        logger.error("❌ Missing training files:")
        for file_name in missing_files:
            logger.error(f"   {DATA_DIR / file_name}")
        logger.error(f"\n💡 Place training data files in: {DATA_DIR}")
        sys.exit(1)
    
    logger.info("✓ All training data files found")


def load_data():
    """Load and prepare datasets"""
    from config import DATA_DIR, DATA_CONFIG
    from data_loader import AMRDataLoader
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 70)
    
    loader = AMRDataLoader(DATA_DIR)
    
    # Load training data
    train_dataset, val_dataset = loader.load_training_data(
        train_files=DATA_CONFIG['train_files'],
        validation_split=DATA_CONFIG['validation_split'],
        max_samples=DATA_CONFIG['max_samples']
    )
    
    logger.info(f"✓ Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"✓ Validation samples: {len(val_dataset)}")
    
    # Load test data if available
    test_datasets = {}
    
    if (DATA_DIR / DATA_CONFIG['public_test_file']).exists():
        public_test = loader.load_test_data(
            DATA_CONFIG['public_test_file'],
            DATA_CONFIG.get('public_test_ground_truth')
        )
        test_datasets['public'] = public_test
        logger.info(f"✓ Public test samples: {len(public_test)}")
    
    if (DATA_DIR / DATA_CONFIG['private_test_file']).exists():
        private_test = loader.load_test_data(DATA_CONFIG['private_test_file'])
        test_datasets['private'] = private_test
        logger.info(f"✓ Private test samples: {len(private_test)}")
    
    return train_dataset, val_dataset, test_datasets, loader


def train_model(train_dataset, val_dataset, args):
    """Train the AMR parser model"""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,  # FIXED: Use Seq2Seq collator instead of LanguageModeling
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from config import (
        OUTPUT_DIR, CHECKPOINT_DIR, MAX_SEQ_LENGTH, 
        LORA_CONFIG, TRAINING_CONFIG, PROMPT_TEMPLATE,
        MODEL_SAVE_NAME, PREPROCESSING_CONFIG
    )
    from preprocessor import ImprovedAMRPreprocessor
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: LOADING MODEL")
    logger.info("=" * 70)
    
    # Model selection based on args
    if args.model == "7b":
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        logger.info(f"Using: {model_name} (7B - More stable)")
    else:
        model_name = "Qwen/Qwen2.5-14B-Instruct" 
        logger.info(f"Using: {model_name} (14B - Better performance)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info("✓ Tokenizer loaded")
    
    # Load model with quantization
    logger.info("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info("✓ Model loaded")
    
    # ============= STEP 3: SETUP LORA =============
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: CONFIGURING LORA")
    logger.info("=" * 70)
    
    lora_config = LoraConfig(
        r=LORA_CONFIG['r'],
        lora_alpha=LORA_CONFIG['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_CONFIG['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"✓ LoRA configured")
    logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")
    logger.info(f"  Trainable: {100 * trainable_params / total_params:.2f}%")
    
    # ============= STEP 4: PREPARE DATASET =============
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 4: PREPARING DATASET")
    logger.info("=" * 70)
    
    preprocessor = ImprovedAMRPreprocessor(config=PREPROCESSING_CONFIG)
    
    def format_example(example):
        """Format example for training with proper structure"""
        # Preprocess AMR
        preprocessed_amr = preprocessor.preprocess(example['amr'])
        
        # Create prompt and full text
        prompt = PROMPT_TEMPLATE.format(sentence=example['sentence'])
        full_text = f"{prompt}{preprocessed_amr}{tokenizer.eos_token}"
        
        # Tokenize
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,  # Let data collator handle padding
            return_tensors=None  # Return as list, not tensor
        )
        
        # IMPORTANT: Labels must be a list, not nested
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    logger.info("Formatting training data...")
    train_formatted = train_dataset.map(
        format_example,
        remove_columns=train_dataset.column_names,
        desc="Processing training data"
    )
    
    val_formatted = None
    if val_dataset:
        logger.info("Formatting validation data...")
        val_formatted = val_dataset.map(
            format_example,
            remove_columns=val_dataset.column_names,
            desc="Processing validation data"
        )
    
    logger.info(f"✓ Dataset prepared: {len(train_formatted)} training samples")
    
    # ============= STEP 5: TRAINING SETUP =============
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 5: TRAINING CONFIGURATION")
    logger.info("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = CHECKPOINT_DIR / f"{MODEL_SAVE_NAME}_{timestamp}"
    
    # Adjust training config based on args
    epochs = args.epochs if args.epochs else TRAINING_CONFIG['num_train_epochs']
    batch_size = args.batch_size if args.batch_size else TRAINING_CONFIG['per_device_train_batch_size']
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
        learning_rate=TRAINING_CONFIG['learning_rate'],
        warmup_steps=TRAINING_CONFIG['warmup_steps'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        max_grad_norm=TRAINING_CONFIG['max_grad_norm'],
        logging_steps=TRAINING_CONFIG['logging_steps'],
        save_steps=TRAINING_CONFIG['save_steps'],
        save_total_limit=TRAINING_CONFIG['save_total_limit'],
        fp16=True,
        optim=TRAINING_CONFIG['optim'],
        lr_scheduler_type=TRAINING_CONFIG['lr_scheduler_type'],
        seed=TRAINING_CONFIG['seed'],
        evaluation_strategy="steps" if val_formatted else "no",
        eval_steps=100 if val_formatted else None,
        load_best_model_at_end=True if val_formatted else False,
        metric_for_best_model="loss" if val_formatted else None,
        report_to="none",
        remove_unused_columns=False,  # Important for custom datasets
    )
    
    # FIXED: Use DataCollatorForSeq2Seq instead of DataCollatorForLanguageModeling
    # This properly handles variable-length sequences and padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,  # Ignore padding in loss calculation
        padding=True,  # Enable dynamic padding
        max_length=MAX_SEQ_LENGTH
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_formatted,
        eval_dataset=val_formatted,
        data_collator=data_collator,
    )
    
    logger.info(f"✓ Trainer configured")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Effective batch size: {batch_size * TRAINING_CONFIG['gradient_accumulation_steps']}")
    logger.info(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    
    # ============= STEP 6: TRAINING =============
    logger.info("")
    logger.info("=" * 70)
    logger.info("🚀 STARTING TRAINING")
    logger.info("=" * 70)
    logger.info("")
    
    train_result = trainer.train()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Final train loss: {train_result.training_loss:.4f}")
    
    # ============= STEP 7: SAVE MODEL =============
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 7: SAVING MODEL")
    logger.info("=" * 70)
    
    final_output = OUTPUT_DIR / MODEL_SAVE_NAME
    # Check if directory exists, if not, show error
    if not final_output.parent.exists():
        logger.error(f"❌ Output directory does not exist: {final_output.parent}")
        logger.error("💡 Create it with: mkdir -p outputs")
        return None, None, None
    
    logger.info(f"Saving model to: {final_output}")
    model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    
    logger.info(f"✓ Model saved to {final_output}")
    
    return model, tokenizer, preprocessor


def run_inference(model, tokenizer, test_datasets, data_loader, args):
    """Run inference on test datasets"""
    from config import OUTPUT_DIR, INFERENCE_CONFIG, PROMPT_TEMPLATE, OUTPUT_PREFIX
    from inference import AMRInference
    from postprocessor import ImprovedAMRPostprocessor
    
    if not test_datasets:
        logger.info("No test datasets to run inference on")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 8: INFERENCE")
    logger.info("=" * 70)
    
    postprocessor = ImprovedAMRPostprocessor()
    
    config = {
        'INFERENCE_CONFIG': INFERENCE_CONFIG,
        'PROMPT_TEMPLATE': PROMPT_TEMPLATE
    }
    
    inference = AMRInference(
        model=model,
        tokenizer=tokenizer,
        postprocessor=postprocessor,
        config=config
    )
    
    results = {}
    
    for test_name, test_df in test_datasets.items():
        logger.info(f"\n🔍 Running inference on {test_name} test set...")
        
        predictions = inference.generate_from_dataframe(
            df=test_df,
            batch_size=args.inference_batch_size,
            show_progress=True
        )
        
        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"{test_name}_results_{timestamp}"
        
        # Check if parent directory exists
        if not output_dir.parent.exists():
            logger.error(f"❌ Output directory does not exist: {output_dir.parent}")
            logger.error("💡 Create it with: mkdir -p outputs")
            continue
        
        saved_files = data_loader.save_predictions(
            predictions=predictions,
            output_dir=output_dir,
            prefix=f"{OUTPUT_PREFIX}_{test_name}"
        )
        
        logger.info(f"✓ {test_name.capitalize()} test predictions saved:")
        for format_name, filepath in saved_files.items():
            logger.info(f"  - {format_name}: {filepath}")
        
        results[test_name] = {
            'predictions': predictions,
            'output_dir': output_dir,
            'test_df': test_df
        }
    
    return results


def evaluate_results(results):
    """Evaluate predictions if ground truth available"""
    from config import EVAL_CONFIG
    from evaluation import AMREvaluator
    
    if not results:
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 9: EVALUATION")
    logger.info("=" * 70)
    
    evaluator = AMREvaluator(config=EVAL_CONFIG)
    
    for test_name, result_data in results.items():
        test_df = result_data['test_df']
        
        # Check if ground truth available
        if 'amr' not in test_df.columns:
            logger.info(f"\n⚠️  No ground truth for {test_name} test - skipping evaluation")
            continue
        
        logger.info(f"\n📊 Evaluating {test_name} test set...")
        
        metrics = evaluator.evaluate_and_save(
            predictions=result_data['predictions'],
            output_dir=result_data['output_dir'],
            ground_truth=test_df['amr'].tolist(),
            prefix=f"{test_name}_evaluation"
        )
        
        logger.info(f"\n{test_name.upper()} TEST RESULTS:")
        logger.info("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key:.<40} {value:.4f}")
            else:
                logger.info(f"{key:.<40} {value}")
        logger.info("=" * 50)


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description="Vietnamese AMR Parser - Unified Training Script"
    )
    
    # Training options
    parser.add_argument(
        '--model',
        type=str,
        choices=['7b', '14b'],
        default='7b',
        help='Model size: 7b (more stable) or 14b (better performance)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size per device (overrides config)'
    )
    
    # Pipeline options
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training phase (requires saved model)'
    )
    parser.add_argument(
        '--skip-inference',
        action='store_true',
        help='Skip inference on test sets'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation phase'
    )
    
    # Inference options
    parser.add_argument(
        '--inference-batch-size',
        type=int,
        default=4,
        help='Batch size for inference (default: 4)'
    )
    
    # Model loading
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to saved model (for skip-training)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check environment
    logger.info("Checking environment...")
    check_directories()
    check_data_files()
    
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available! Training will be slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    logger.info("✓ Environment check passed")
    
    # Load data
    train_dataset, val_dataset, test_datasets, data_loader = load_data()
    
    # Training
    model, tokenizer, preprocessor = None, None, None
    
    if not args.skip_training:
        model, tokenizer, preprocessor = train_model(train_dataset, val_dataset, args)
        
        if model is None:
            logger.error("Training failed - exiting")
            sys.exit(1)
    else:
        logger.info("\n⏭️  Skipping training as requested")
        if not args.model_path:
            logger.error("--model-path required when using --skip-training")
            sys.exit(1)
        
        logger.info(f"Loading model from: {args.model_path}")
        # TODO: Implement model loading
        logger.error("Model loading not yet implemented")
        sys.exit(1)
    
    # Inference
    results = None
    if not args.skip_inference and test_datasets:
        results = run_inference(model, tokenizer, test_datasets, data_loader, args)
    
    # Evaluation
    if not args.skip_evaluation and results:
        evaluate_results(results)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("🎉 PIPELINE COMPLETE!")
    logger.info("=" * 70)
    
    from config import OUTPUT_DIR, LOG_DIR
    logger.info(f"📁 All outputs saved to: {OUTPUT_DIR}")
    logger.info(f"📊 Logs saved to: {LOG_DIR}")
    logger.info("=" * 70)
    
    logger.info("\n✨ Next steps:")
    logger.info("  1. Review evaluation metrics (if available)")
    logger.info("  2. Check prediction quality in output files")
    logger.info("  3. Submit results to VLSP 2025 competition")
    logger.info("  4. Consider fine-tuning if needed")
    logger.info("")
    logger.info("🚀 Your Vietnamese AMR parser is ready!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("❌ Fatal error occurred:")
        sys.exit(1)
