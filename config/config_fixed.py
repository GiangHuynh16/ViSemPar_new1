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
    "num_train_epochs": 15,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 5,
    "fp16": False,
    "bf16": True,
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",
    "gradient_checkpointing": False,  # CRITICAL: Disable for LoRA compatibility
}

# Inference Configuration
INFERENCE_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.15,
    "max_new_tokens": 512,
    "do_sample": True,
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
# FIXED PROMPT TEMPLATE - Clear Penman format requirements
# ============================================================================
PROMPT_TEMPLATE = """Bạn là chuyên gia ngôn ngữ học máy tính, chuyên về phân tích ngữ nghĩa tiếng Việt.
Hãy chuyển đổi câu văn sau sang định dạng AMR (Abstract Meaning Representation) theo đúng **chuẩn Penman**.

Các quy tắc bắt buộc:
1. Sử dụng định dạng Penman: (biến / khái niệm :quan-hệ (biến2 / khái niệm2))
2. Khái niệm tiếng Việt đa âm tiết phải dùng dấu gạch dưới (ví dụ: c / chính_phủ, p / phát_triển)
3. Sử dụng các quan hệ chuẩn: :ARG0, :ARG1, :ARG2, :time, :location, :mod, :poss, v.v.
4. Đảm bảo cấu trúc cây với các dấu đóng mở ngoặc đơn hoàn toàn cân bằng
5. Mỗi khái niệm chỉ nên được gán một biến duy nhất trong toàn bộ cấu trúc
6. KHÔNG thêm giải thích, chỉ trả về cấu trúc AMR thuần túy

Câu tiếng Việt: {sentence}

AMR (Penman):
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
