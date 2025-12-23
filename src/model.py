"""
Model Training Module for Vietnamese AMR Parser
Uses Unsloth for efficient LoRA fine-tuning
"""

import torch
import logging
from typing import Dict, Optional
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer

logger = logging.getLogger(__name__)


class AMRModelTrainer:
    """
    Handles model loading, training, and saving
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model(self):
        """Load and configure model with LoRA"""
        logger.info(f"Loading model: {self.config['MODEL_NAME']}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config['MODEL_NAME'],
            max_seq_length=self.config['MAX_SEQ_LENGTH'],
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        # Configure LoRA
        logger.info("Configuring LoRA...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            **self.config['LORA_CONFIG']
        )
        
        # Log model info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
        
        return self.model, self.tokenizer
    
    def prepare_dataset(
        self,
        dataset: Dataset,
        preprocessor,
        prompt_template: str
    ) -> Dataset:
        """
        Prepare dataset for training
        Applies preprocessing and formats with prompts
        """
        logger.info(f"Preparing dataset: {len(dataset)} samples")
        
        def format_example(example):
            # Preprocess AMR
            preprocessed_amr = preprocessor.preprocess(example['amr'])
            
            # Format prompt
            prompt = prompt_template.format(sentence=example['sentence'])
            
            # Combine for training
            text = f"{prompt}{preprocessed_amr}"
            
            return {"text": text}
        
        # Map over dataset
        formatted_dataset = dataset.map(
            format_example,
            remove_columns=dataset.column_names,
            desc="Formatting examples"
        )
        
        logger.info(f"Dataset prepared: {len(formatted_dataset)} samples")
        return formatted_dataset
    
    def create_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: Path = None
    ):
        """Create Trainer with optimized settings"""
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            **self.config['TRAINING_CONFIG'],
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="loss" if eval_dataset else None,
        )
        
        # Use TRL's SFTTrainer for better memory efficiency
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config['MAX_SEQ_LENGTH'],
            packing=False,  # Don't pack for AMR to preserve structure
        )
        
        logger.info("Trainer created successfully")
        return self.trainer
    
    def train(self):
        """Run training"""
        if self.trainer is None:
            raise ValueError("Trainer not created. Call create_trainer() first.")
        
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)
        
        # Train
        train_result = self.trainer.train()
        
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Final train loss: {train_result.training_loss:.4f}")
        logger.info("=" * 80)
        
        return train_result
    
    def save_model(self, output_dir: Path, save_method: str = "merged"):
        """
        Save model in different formats
        
        Args:
            save_method: "merged" (full model) or "lora" (adapter only)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to: {output_dir}")
        
        if save_method == "merged":
            # Save merged model (base + LoRA)
            logger.info("Saving merged 16-bit model...")
            self.model.save_pretrained_merged(
                str(output_dir / "merged_16bit"),
                self.tokenizer,
                save_method="merged_16bit"
            )
            
            # Also save 4-bit for inference
            logger.info("Saving merged 4-bit model...")
            self.model.save_pretrained_merged(
                str(output_dir / "merged_4bit"),
                self.tokenizer,
                save_method="merged_4bit"
            )
            
        elif save_method == "lora":
            # Save only LoRA adapter
            logger.info("Saving LoRA adapter...")
            self.model.save_pretrained(str(output_dir / "lora_adapter"))
            self.tokenizer.save_pretrained(str(output_dir / "lora_adapter"))
        
        else:
            raise ValueError(f"Unknown save method: {save_method}")
        
        logger.info(f"Model saved successfully to {output_dir}")
    
    def push_to_hub(self, repo_name: str, private: bool = False):
        """Push model to Hugging Face Hub"""
        logger.info(f"Pushing to Hub: {repo_name}")
        
        try:
            # Push merged 16-bit model
            self.model.push_to_hub_merged(
                repo_name,
                self.tokenizer,
                save_method="merged_16bit",
                token=True,  # Will use HF_TOKEN from environment
                private=private
            )
            logger.info(f"✅ Model pushed to Hub: {repo_name}")
            
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")
            logger.info("Make sure you have HF_TOKEN set and huggingface-cli login done")
    
    def prepare_for_inference(self):
        """Switch model to inference mode"""
        logger.info("Preparing model for inference...")
        FastLanguageModel.for_inference(self.model)
        logger.info("Model ready for inference")


class EarlyStoppingCallback:
    """Custom early stopping based on SMATCH score"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            # Improvement
            self.best_score = score
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            logger.info("Early stopping triggered!")
            self.should_stop = True
            return True
        
        return False


def test_model_loading():
    """Test model loading"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.config import (
        MODEL_NAME, MAX_SEQ_LENGTH, LORA_CONFIG, TRAINING_CONFIG
    )
    
    config = {
        'MODEL_NAME': MODEL_NAME,
        'MAX_SEQ_LENGTH': MAX_SEQ_LENGTH,
        'LORA_CONFIG': LORA_CONFIG,
        'TRAINING_CONFIG': TRAINING_CONFIG,
    }
    
    trainer = AMRModelTrainer(config)
    model, tokenizer = trainer.load_model()
    
    print("Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")


if __name__ == "__main__":
    test_model_loading()

