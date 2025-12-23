"""
Vietnamese AMR Parser - Standard Training (No Unsloth)
Stable training without dependency issues
Run with: python train_standard.py
"""

import os
import sys
import torch
import logging
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
    ║         VIETNAMESE AMR PARSER - STANDARD TRAINING           ║
    ║                                                              ║
    ║              Abstract Meaning Representation                 ║
    ║                     for Vietnamese                          ║
    ║                                                              ║
    ║                  (Without Unsloth)                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"🚀 Training Mode: Standard HuggingFace + LoRA")
    print(f"💾 PyTorch: {torch.__version__}")
    print(f"⚡ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)


def main():
    print_banner()
    
    # Import after banner
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # Import project modules
    from config import (
        DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR,
        MAX_SEQ_LENGTH, LORA_CONFIG, TRAINING_CONFIG,
        PROMPT_TEMPLATE, DATA_CONFIG, MODEL_SAVE_NAME,
        PREPROCESSING_CONFIG
    )
    from data_loader import AMRDataLoader
    from preprocessor import ImprovedAMRPreprocessor
    
    # ============= STEP 1: LOAD DATA =============
    logger.info("=" * 70)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 70)
    
    loader = AMRDataLoader(DATA_DIR)
    train_dataset, val_dataset = loader.load_training_data(
        train_files=DATA_CONFIG['train_files'],
        validation_split=DATA_CONFIG['validation_split'],
        max_samples=DATA_CONFIG['max_samples']
    )
    
    logger.info(f"✓ Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"✓ Validation samples: {len(val_dataset)}")
    
    # ============= STEP 2: LOAD MODEL =============
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: LOADING MODEL")
    logger.info("=" * 70)
    
    # Use 7B model (more stable, fits better in memory)
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    logger.info(f"Model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info("Loading model with 4-bit quantization...")
    from transformers import BitsAndBytesConfig
    
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
        """Format example for training"""
        preprocessed_amr = preprocessor.preprocess(example['amr'])
        prompt = PROMPT_TEMPLATE.format(sentence=example['sentence'])
        text = f"{prompt}{preprocessed_amr}{tokenizer.eos_token}"
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
            return_tensors=None
        )
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
    logger.info("STEP 6: TRAINING CONFIGURATION")
    logger.info("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = CHECKPOINT_DIR / f"{MODEL_SAVE_NAME}_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=TRAINING_CONFIG['num_train_epochs'],
        per_device_train_batch_size=TRAINING_CONFIG['per_device_train_batch_size'],
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
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
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
    logger.info(f"  Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    logger.info(f"  Effective batch size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    logger.info(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    
    # ============= STEP 6: TRAINING =============
    logger.info("")
    logger.info("=" * 70)
    logger.info("🚀 STARTING TRAINING")
    logger.info("=" * 70)
    
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
    final_output.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to: {final_output}")
    model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    
    logger.info(f"✓ Model saved to {final_output}")
    
    # ============= FINAL SUMMARY =============
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 ALL STEPS COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"📁 Model: {final_output}")
    logger.info(f"📁 Checkpoints: {output_dir}")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info("✨ Next steps:")
    logger.info("  1. Use the saved model for inference")
    logger.info("  2. Run evaluation on test sets")
    logger.info("  3. Generate predictions")
    logger.info("")
    logger.info("🚀 Your Vietnamese AMR parser is ready!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error occurred:")
        sys.exit(1)
