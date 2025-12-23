"""
Main Entry Point for Vietnamese AMR Parser Training
Run with: python main.py
"""

import os
import sys
import logging
import logging.config
import argparse
from pathlib import Path
from datetime import datetime
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'config'))

# Import configuration
from config import (
    PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR,
    MODEL_NAME, MAX_SEQ_LENGTH, LORA_CONFIG, TRAINING_CONFIG,
    INFERENCE_CONFIG, DATA_CONFIG, HF_CONFIG, MODEL_SAVE_NAME,
    PROMPT_TEMPLATE, LOGGING_CONFIG, EVAL_CONFIG, PREPROCESSING_CONFIG,
    OUTPUT_PREFIX
)

# Import modules
from data_loader import AMRDataLoader
from preprocessor import ImprovedAMRPreprocessor
from postprocessor import ImprovedAMRPostprocessor
from model import AMRModelTrainer
from inference import AMRInference
from evaluation import AMREvaluator

# Setup logging
# logging.config.dictConfig(LOGGING_CONFIG)
# Using basic logging
logger = logging.getLogger(__name__)


def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         VIETNAMESE AMR PARSER - IMPROVED PIPELINE          ║
    ║                                                              ║
    ║              Abstract Meaning Representation                 ║
    ║                     for Vietnamese                          ║
    ║                                                              ║
    ║                      VLSP 2025                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"🚀 Model: {MODEL_NAME}")
    print(f"💾 Max Seq Length: {MAX_SEQ_LENGTH}")
    print(f"🔧 LoRA R={LORA_CONFIG['r']}, Alpha={LORA_CONFIG['lora_alpha']}")
    print(f"📚 Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"⚡ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)


def setup_environment():
    """Setup environment and check dependencies"""
    logger.info("Setting up environment...")
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be very slow.")
        logger.warning("This script requires GPU for reasonable training time.")
    
    # Create directories
    for dir_path in [DATA_DIR, OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check data files
    missing_files = []
    for train_file in DATA_CONFIG['train_files']:
        if not (DATA_DIR / train_file).exists():
            missing_files.append(train_file)
    
    if missing_files:
        logger.error(f"Missing training files: {missing_files}")
        logger.error(f"Please place training data in: {DATA_DIR}")
        sys.exit(1)
    
    logger.info("Environment setup complete ✓")


def load_data():
    """Load and prepare datasets"""
    logger.info("=" * 70)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 70)
    
    data_loader = AMRDataLoader(DATA_DIR)
    
    # Load training data
    train_dataset, val_dataset = data_loader.load_training_data(
        train_files=DATA_CONFIG['train_files'],
        validation_split=DATA_CONFIG['validation_split'],
        max_samples=DATA_CONFIG['max_samples']
    )
    
    logger.info(f"✓ Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"✓ Validation samples: {len(val_dataset)}")
    
    # Load test data if available
    public_test_df = None
    if (DATA_DIR / DATA_CONFIG['public_test_file']).exists():
        public_test_df = data_loader.load_test_data(
            DATA_CONFIG['public_test_file'],
            DATA_CONFIG.get('public_test_ground_truth')
        )
        logger.info(f"✓ Public test samples: {len(public_test_df)}")
    
    private_test_df = None
    if (DATA_DIR / DATA_CONFIG['private_test_file']).exists():
        private_test_df = data_loader.load_test_data(
            DATA_CONFIG['private_test_file']
        )
        logger.info(f"✓ Private test samples: {len(private_test_df)}")
    
    return train_dataset, val_dataset, public_test_df, private_test_df, data_loader


def train_model(train_dataset, val_dataset):
    """Train the model"""
    logger.info("=" * 70)
    logger.info("STEP 2: TRAINING MODEL")
    logger.info("=" * 70)
    
    # Initialize preprocessor
    preprocessor = ImprovedAMRPreprocessor(config=PREPROCESSING_CONFIG)
    
    # Initialize model trainer
    config = {
        'MODEL_NAME': MODEL_NAME,
        'MAX_SEQ_LENGTH': MAX_SEQ_LENGTH,
        'LORA_CONFIG': LORA_CONFIG,
        'TRAINING_CONFIG': TRAINING_CONFIG,
        'PROMPT_TEMPLATE': PROMPT_TEMPLATE,
    }
    
    trainer = AMRModelTrainer(config)
    
    # Load model
    model, tokenizer = trainer.load_model()
    
    # Prepare datasets
    train_formatted = trainer.prepare_dataset(
        train_dataset,
        preprocessor,
        PROMPT_TEMPLATE
    )
    
    val_formatted = None
    if val_dataset:
        val_formatted = trainer.prepare_dataset(
            val_dataset,
            preprocessor,
            PROMPT_TEMPLATE
        )
    
    # Create trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = CHECKPOINT_DIR / f"{MODEL_SAVE_NAME}_{timestamp}"
    
    trainer.create_trainer(
        train_dataset=train_formatted,
        eval_dataset=val_formatted,
        output_dir=checkpoint_dir
    )
    
    # Train
    train_result = trainer.train()
    
    # Save model
    logger.info("Saving model...")
    output_model_dir = OUTPUT_DIR / MODEL_SAVE_NAME
    trainer.save_model(output_model_dir, save_method="merged")
    
    # Push to Hub if configured
    if HF_CONFIG['push_to_hub']:
        logger.info("Pushing to Hugging Face Hub...")
        try:
            trainer.push_to_hub(
                repo_name=HF_CONFIG['repo_name'],
                private=HF_CONFIG['private']
            )
        except Exception as e:
            logger.warning(f"Failed to push to Hub: {e}")
            logger.info("You can push manually later with push_to_hub()")
    
    # Prepare for inference
    trainer.prepare_for_inference()
    
    logger.info("✓ Training complete!")
    return trainer.model, trainer.tokenizer, preprocessor


def run_inference(model, tokenizer, test_df, data_loader, test_name="test"):
    """Run inference on test set"""
    logger.info("=" * 70)
    logger.info(f"STEP 3: INFERENCE ON {test_name.upper()}")
    logger.info("=" * 70)
    
    # Initialize postprocessor
    postprocessor = ImprovedAMRPostprocessor()
    
    # Initialize inference engine
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
    
    # Generate predictions
    predictions = inference.generate_from_dataframe(
        df=test_df,
        batch_size=4,  # Adjust based on GPU memory
        show_progress=True
    )
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"{test_name}_results_{timestamp}"
    
    saved_files = data_loader.save_predictions(
        predictions=predictions,
        output_dir=output_dir,
        prefix=f"{OUTPUT_PREFIX}_{test_name}"
    )
    
    logger.info("✓ Predictions saved:")
    for format_name, filepath in saved_files.items():
        logger.info(f"  - {format_name}: {filepath}")
    
    return predictions, output_dir


def evaluate_results(predictions, ground_truth, output_dir):
    """Evaluate predictions"""
    logger.info("=" * 70)
    logger.info("STEP 4: EVALUATION")
    logger.info("=" * 70)
    
    evaluator = AMREvaluator(config=EVAL_CONFIG)
    
    # Convert ground truth if needed
    gt_amrs = None
    if ground_truth is not None:
        gt_amrs = ground_truth if isinstance(ground_truth, list) else ground_truth.tolist()
    
    # Evaluate
    metrics = evaluator.evaluate_and_save(
        predictions=predictions,
        output_dir=output_dir,
        ground_truth=gt_amrs,
        prefix="evaluation"
    )
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key:.<50} {value:.4f}")
        else:
            logger.info(f"{key:.<50} {value}")
    
    logger.info("=" * 70)
    
    return metrics


def main():
    """Main training and evaluation pipeline"""
    parser = argparse.ArgumentParser(description="Vietnamese AMR Parser Training")
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip training and only run inference (requires saved model)')
    parser.add_argument('--skip-public-test', action='store_true',
                       help='Skip public test evaluation')
    parser.add_argument('--skip-private-test', action='store_true',
                       help='Skip private test evaluation')
    parser.add_argument('--model-path', type=str,
                       help='Path to saved model (for skip-training)')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup
    setup_environment()
    
    # Load data
    train_dataset, val_dataset, public_test_df, private_test_df, data_loader = load_data()
    
    # Training
    if not args.skip_training:
        model, tokenizer, preprocessor = train_model(train_dataset, val_dataset)
    else:
        logger.info("Skipping training as requested")
        if not args.model_path:
            logger.error("--model-path required when using --skip-training")
            sys.exit(1)
        # Load saved model (implementation would go here)
        logger.error("Loading saved model not yet implemented")
        sys.exit(1)
    
    # Public test evaluation
    if public_test_df is not None and not args.skip_public_test:
        public_predictions, public_output_dir = run_inference(
            model, tokenizer, public_test_df, data_loader, test_name="public_test"
        )
        
        # Evaluate if ground truth available
        if 'amr' in public_test_df.columns:
            evaluate_results(
                predictions=public_predictions,
                ground_truth=public_test_df['amr'],
                output_dir=public_output_dir
            )
    
    # Private test inference
    if private_test_df is not None and not args.skip_private_test:
        private_predictions, private_output_dir = run_inference(
            model, tokenizer, private_test_df, data_loader, test_name="private_test"
        )
        
        logger.info("✓ Private test predictions complete (no ground truth for evaluation)")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("🎉 ALL STEPS COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"📁 All outputs saved to: {OUTPUT_DIR}")
    logger.info(f"📊 Logs saved to: {LOG_DIR}")
    logger.info("=" * 70)
    
    logger.info("\n✨ Next steps:")
    logger.info("  1. Review evaluation metrics")
    logger.info("  2. Check prediction quality")
    logger.info("  3. Submit results to VLSP 2025")
    logger.info("  4. Consider fine-tuning parameters if needed")
    
    if HF_CONFIG['push_to_hub']:
        logger.info(f"\n🤗 Model available on Hugging Face: {HF_CONFIG['repo_name']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error occurred:")
        sys.exit(1)
