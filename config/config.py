"""
Vietnamese AMR Parser Configuration
Optimized settings for improved SMATCH scores
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# NOTE: Directories must be created manually before training
# Run: mkdir -p data outputs logs outputs/checkpoints

# Model Configuration - IMPROVED for better performance
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MAX_SEQ_LENGTH = 2048  # Sufficient for complex AMR structures

# LoRA Configuration - Optimized for Vietnamese AMR
LORA_CONFIG = {
    "r": 128,                    # Increased rank for better capacity
    "lora_alpha": 256,           # 2x rank for stability
    "lora_dropout": 0.05,        # Small dropout for regularization
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
    # task_type: auto-added by unsloth

}

# Training Configuration - Optimized for convergence
TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 20,              # Increased for better convergence
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,    # Effective batch size: 16
    "warmup_steps": 50,                  # Gradual warmup
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 3,
    "fp16": True,
    "optim": "adamw_8bit",
    "lr_scheduler_type": "cosine",
    "seed": 42,
}

# Inference Configuration
INFERENCE_CONFIG = {
    "temperature": 0.1,              # Lower for more deterministic output
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.15,
    "max_new_tokens": 512,           # Increased for complex AMRs
    "do_sample": True,
}

# Data Configuration
DATA_CONFIG = {
    "train_files": ["train_amr_1.txt", "train_amr_2.txt"],
    "public_test_file": "public_test.txt",
    "public_test_ground_truth": "public_test_ground_truth.txt",
    "private_test_file": "private_test.txt",
    "validation_split": 0.05,        # 5% for validation
    "max_samples": None,              # None = use all data
}

# Hugging Face Configuration
HF_CONFIG = {
    "repo_name": "vietnamese-amr-qwen-improved",
    "private": False,
    "push_to_hub": True,
    "hub_strategy": "every_save",
}

# Model Save Configuration
MODEL_SAVE_NAME = "vlsp_amr_qwen_improved_v2"
OUTPUT_PREFIX = "vietnamese_amr"

# Evaluation Configuration
EVAL_CONFIG = {
    "compute_smatch": True,
    "smatch_timeout": 30,            # seconds per comparison
    "early_stopping_patience": 3,     # Stop if no improvement
    "min_delta": 0.001,              # Minimum improvement threshold
}

# Preprocessing improvements
PREPROCESSING_CONFIG = {
    "preserve_coreference": True,
    "normalize_concepts": True,
    "handle_multiword": True,
    "fix_malformed_amr": True,
    "remove_variables": False,  # Keep variables during training
}

# System Configuration
SYSTEM_CONFIG = {
    "num_workers": 4,
    "pin_memory": True,
    "use_cache": True,
}

# Prompt Template
PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Convert the following Vietnamese sentence to Abstract Meaning Representation (AMR) format. Ensure proper concept alignment and preserve co-references.

### Input:
{sentence}

### Response:
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
