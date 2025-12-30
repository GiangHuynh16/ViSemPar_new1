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

# Model Configuration - BASELINE (Same as MTUP for fair comparison)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Same as MTUP
MAX_SEQ_LENGTH = 512  # CRITICAL: Reduced to 512 to fit in memory (MTUP uses 2048 but baseline has OOM)

# Quantization - Disabled (same as MTUP)
USE_4BIT_QUANTIZATION = False  # Disabled - bitsandbytes not available

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
    "num_train_epochs": 15,              # Same as MTUP
    "per_device_train_batch_size": 1,    # Reduced to 1 (MTUP uses 2 but OOM here)
    "gradient_accumulation_steps": 16,   # Increased to 16 to keep effective batch = 16
    "warmup_steps": 100,                 # Same as MTUP
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 200,                   # Same as MTUP 7B
    "save_total_limit": 5,               # Same as MTUP 7B
    "fp16": True,
    "optim": "adamw_torch",              # FIXED: Use standard AdamW (adamw_8bit requires bitsandbytes)
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "load_best_model_at_end": True,      # ADDED: Load best checkpoint at end
    "metric_for_best_model": "loss",     # ADDED: Use loss as metric
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
    "repo_name": "vietnamese-amr-baseline-7b",  # Updated for 7B baseline
    "private": False,
    "push_to_hub": False,                       # Set to True when ready
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

# Preprocessing improvements - Minimal for LLM
# Philosophy: Let the LLM learn from raw data, minimal preprocessing
PREPROCESSING_CONFIG = {
    "preserve_coreference": True,       # Keep coreference info
    "normalize_concepts": False,        # DON'T normalize - let LLM learn variations
    "handle_multiword": False,          # DON'T preprocess - let LLM learn underscore patterns
    "fix_malformed_amr": True,          # Only fix obviously broken AMRs
    "remove_variables": False,          # Keep variables during training
    "clean_whitespace": True,           # Basic whitespace cleaning only
    "validate_structure": True,         # Validate parentheses balance
}

# System Configuration
SYSTEM_CONFIG = {
    "num_workers": 4,
    "pin_memory": True,
    "use_cache": True,
}

# Prompt Template - Enhanced for better LLM performance
PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt. Hãy chuyển đổi câu sau sang định dạng AMR (Abstract Meaning Representation).

Quy tắc quan trọng:
- Sử dụng khái niệm tiếng Việt có dấu gạch dưới (ví dụ: chủ_tịch, môi_trường)
- Gán biến cho mỗi khái niệm (ví dụ: c / chủ_tịch)
- Sử dụng quan hệ chuẩn AMR (:ARG0, :ARG1, :time, :location, etc.)
- Giữ nguyên cấu trúc cây với dấu ngoặc đơn cân bằng
- Đảm bảo tất cả biến được định nghĩa trước khi sử dụng

Câu tiếng Việt: {sentence}

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
