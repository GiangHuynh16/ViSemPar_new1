"""
Vietnamese AMR Parser Configuration - FIXED VERSION
Fixes critical training issues:
1. Missing EOS token
2. No instruction masking
3. Unclear Penman format requirements
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Model Configuration - BASELINE (7B for comparison with MTUP)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 256  # Reduced to fit in memory

# Quantization - DISABLED
USE_4BIT_QUANTIZATION = False

# LoRA Configuration
LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
}

# Training Configuration
TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 2,  # REDUCED: checkpoint-200 was best (70% valid), avoid overfitting
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "warmup_steps": 50,  # Reduced warmup
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 100,  # Save more frequently to find sweet spot
    "save_total_limit": 10,  # Keep more checkpoints for testing
    "fp16": False,
    "bf16": True,
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "load_best_model_at_end": False,  # Disabled: peft version incompatibility
    "metric_for_best_model": "loss",
    "gradient_checkpointing": False,  # CRITICAL: Disable for LoRA compatibility
}

# Inference Configuration
INFERENCE_CONFIG = {
    "temperature": 0.3,  # Slightly higher for diversity
    "top_p": 0.95,  # Allow more tokens
    "top_k": 50,
    "repetition_penalty": 1.2,  # Stronger penalty to avoid loops
    "max_new_tokens": 512,
    "do_sample": True,
    "num_beams": 1,  # Greedy for consistency
}

# Data Configuration
DATA_CONFIG = {
    "train_files": ["train_amr_1.txt", "train_amr_2.txt"],
    "public_test_file": "public_test.txt",
    "public_test_ground_truth": "public_test_ground_truth.txt",
    "private_test_file": "private_test.txt",
    "validation_split": 0.05,
    "max_samples": None,
}

# Hugging Face Configuration
HF_CONFIG = {
    "repo_name": "vietnamese-amr-baseline-7b-fixed",
    "private": False,
    "push_to_hub": False,
    "hub_strategy": "every_save",
}

# Model Save Configuration
MODEL_SAVE_NAME = "vlsp_amr_baseline_7b_fixed"
OUTPUT_PREFIX = "vietnamese_amr_fixed"

# Evaluation Configuration
EVAL_CONFIG = {
    "compute_smatch": True,
    "smatch_timeout": 30,
    "early_stopping_patience": 3,
    "min_delta": 0.001,
}

# Preprocessing improvements
PREPROCESSING_CONFIG = {
    "preserve_coreference": True,
    "normalize_concepts": False,
    "handle_multiword": False,
    "fix_malformed_amr": True,
    "remove_variables": False,
    "clean_whitespace": True,
    "validate_structure": True,
}

# System Configuration
SYSTEM_CONFIG = {
    "num_workers": 4,
    "pin_memory": True,
    "use_cache": True,
}

# ============================================================================
# FIXED PROMPT TEMPLATE - Simple and clear, matches training format
# ============================================================================
PROMPT_TEMPLATE = """Chuyển câu tiếng Việt sau sang AMR (Abstract Meaning Representation) theo định dạng Penman:

Câu: {sentence}

AMR:
"""

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s"
        },
        "simple": {
            "format": "%(levelname)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "training.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["console", "file"]
        }
    }
}
