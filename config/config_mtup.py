"""
MTUP Configuration for Vietnamese AMR Parser
Optimized for smaller models (4B parameters) with multi-task unified prompt
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints_mtup"

# ==============================================================================
# MODEL CONFIGURATION - SMALLER MODELS FOR FASTER TRAINING
# ==============================================================================

# Available models (choose one):
MODELS = {
    # Qwen 2.5 models (RECOMMENDED)
    'qwen2.5-7b': "Qwen/Qwen2.5-7B-Instruct",
    'qwen2.5-3b': "Qwen/Qwen2.5-3B-Instruct",
    'qwen2.5-1.5b': "Qwen/Qwen2.5-1.5B-Instruct",
    'qwen2.5-0.5b': "Qwen/Qwen2.5-0.5B-Instruct",

    # Qwen 3 models (NEWEST - if available)
    'qwen3-4b': "Qwen/Qwen3-4B-Instruct",  # If released

    # Gemma models
    'gemma-2-9b': "google/gemma-2-9b-it",
    'gemma-2-2b': "google/gemma-2-2b-it",

    # Phi models (Microsoft - very efficient)
    'phi-3-mini': "microsoft/Phi-3-mini-4k-instruct",  # 3.8B
    'phi-3.5-mini': "microsoft/Phi-3.5-mini-instruct",  # 3.8B
}

# Default model for MTUP training
# NOTE: Using 3B instead of 7B due to GPU memory constraints (no quantization)
# 7B requires bitsandbytes for quantization which is not available
MODEL_NAME = MODELS['qwen2.5-3b']
MAX_SEQ_LENGTH = 2048  # Sufficient for MTUP format with 2 tasks

# ==============================================================================
# QUANTIZATION CONFIGURATION
# ==============================================================================

# Use 4-bit quantization for memory efficiency
# Set to False if you have enough VRAM and want faster training
# NOTE: Requires bitsandbytes package - disabled if not installed
USE_4BIT_QUANTIZATION = False  # Disabled - bitsandbytes not available

# ==============================================================================
# LORA CONFIGURATION - OPTIMIZED FOR SMALLER MODELS
# ==============================================================================

LORA_CONFIG = {
    "r": 64,                     # Reduced rank for smaller models (was 128)
    "lora_alpha": 128,           # 2x rank for stability
    "lora_dropout": 0.05,        # Small dropout
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
}

# ==============================================================================
# TRAINING CONFIGURATION - OPTIMIZED FOR MTUP
# ==============================================================================

TRAINING_CONFIG = {
    "learning_rate": 2e-4,              # OPTIMIZED: Lower for stable training
    "num_train_epochs": 15,              # IMPROVED: More epochs for better convergence (was 10)
    "per_device_train_batch_size": 4,    # 3B model can handle larger batch
    "gradient_accumulation_steps": 4,    # Effective batch size: 16
    "warmup_steps": 100,                 # More warmup for stability
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 200,                   # IMPROVED: Save more frequently (was 250)
    "save_total_limit": 5,               # IMPROVED: Keep more checkpoints (was 3)
    "fp16": True,
    "optim": "adamw_8bit",
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "load_best_model_at_end": True,      # ADDED: Load best checkpoint at end
    "metric_for_best_model": "loss",     # ADDED: Use loss as metric
}

# ==============================================================================
# INFERENCE CONFIGURATION
# ==============================================================================

INFERENCE_CONFIG = {
    "temperature": 0.2,              # Low for deterministic, but not too low
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "max_new_tokens": 768,           # Longer for MTUP (2 outputs)
    "do_sample": True,
}

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

DATA_CONFIG = {
    "train_files": ["train_amr_1.txt", "train_amr_2.txt"],
    "public_test_file": "public_test.txt",
    "public_test_ground_truth": "public_test_ground_truth.txt",
    "private_test_file": "private_test.txt",
    "validation_split": 0.1,         # OPTIMIZED: 10% for better validation monitoring
    "max_samples": None,              # None = use all data
}

# ==============================================================================
# HUGGING FACE CONFIGURATION
# ==============================================================================

HF_CONFIG = {
    "repo_name": "vietnamese-amr-mtup-qwen",
    "private": False,
    "push_to_hub": False,            # Set to True when ready
    "hub_strategy": "every_save",
}

# ==============================================================================
# MODEL SAVE CONFIGURATION
# ==============================================================================

MODEL_SAVE_NAME = "vlsp_amr_mtup_v1"
OUTPUT_PREFIX = "vietnamese_amr_mtup"

# ==============================================================================
# EVALUATION CONFIGURATION
# ==============================================================================

EVAL_CONFIG = {
    "compute_smatch": True,
    "smatch_timeout": 30,
    "early_stopping_patience": 3,
    "min_delta": 0.001,
}

# ==============================================================================
# MTUP PREPROCESSING CONFIGURATION
# ==============================================================================

MTUP_CONFIG = {
    # Template selection: 'v1_formal', 'v2_natural', 'v3_instructional',
    #                     'v4_compact', 'v5_cot', 'recommended'
    "template_name": "v2_natural",    # RECOMMENDED for Vietnamese

    # Format options
    "use_graph_format": True,          # True: multi-line AMR, False: linearized

    # Task configuration
    "num_tasks": 2,                    # Number of tasks in prompt (currently 2)
    "task_order": ["parsing", "binding"],  # Task execution order

    # Validation
    "validate_both_outputs": True,     # Validate both task outputs
    "skip_invalid_examples": False,    # Keep or skip invalid examples
}

# ==============================================================================
# SYSTEM CONFIGURATION
# ==============================================================================

SYSTEM_CONFIG = {
    "num_workers": 4,
    "pin_memory": True,
    "use_cache": True,
    "mixed_precision": "fp16",
}

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

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
            "filename": str(LOG_DIR / "training_mtup.log"),
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

# ==============================================================================
# MODEL COMPARISON TABLE
# ==============================================================================

MODEL_COMPARISON = """
┌─────────────────────┬────────────┬───────────────┬─────────────────────┐
│ Model               │ Parameters │ Speed         │ Recommended Use     │
├─────────────────────┼────────────┼───────────────┼─────────────────────┤
│ Qwen2.5-7B          │ 7B         │ Baseline      │ Best accuracy       │
│ Qwen2.5-3B ⭐       │ 3B         │ 2.3x faster   │ Fast iteration      │
│ Qwen2.5-1.5B        │ 1.5B       │ 4.7x faster   │ Rapid prototyping   │
│ Qwen3-4B (new) ⭐    │ 4B         │ 1.75x faster  │ Best balance        │
│ Gemma-2-2b          │ 2B         │ 3.5x faster   │ Resource limited    │
│ Phi-3.5-mini ⭐      │ 3.8B       │ 1.8x faster   │ Efficient learning  │
└─────────────────────┴────────────┴───────────────┴─────────────────────┘

⭐ = Recommended for MTUP training

MTUP Benefits with Smaller Models:
✅ Explicit task decomposition → Easier learning
✅ 2-stage supervision → Better guidance
✅ Smaller models can achieve similar performance
✅ 2-5x faster training time
✅ Less GPU memory required
"""

# ==============================================================================
# QUICK MODEL SWITCH
# ==============================================================================

def set_model(model_key: str):
    """
    Quick switch to different model

    Args:
        model_key: One of the keys in MODELS dict
                  ('qwen2.5-3b', 'qwen3-4b', 'gemma-2-2b', etc.)
    """
    global MODEL_NAME
    if model_key in MODELS:
        MODEL_NAME = MODELS[model_key]
        print(f"✅ Model switched to: {MODEL_NAME}")
        return MODEL_NAME
    else:
        available = ', '.join(MODELS.keys())
        raise ValueError(f"Invalid model_key. Available: {available}")


# ==============================================================================
# RECOMMENDED CONFIGURATIONS BY USE CASE
# ==============================================================================

CONFIGS_BY_USE_CASE = {
    "quick_test": {
        "model": "qwen2.5-1.5b",
        "epochs": 5,
        "batch_size": 8,
        "max_samples": 500,
    },
    "fast_iteration": {
        "model": "qwen2.5-3b",
        "epochs": 10,
        "batch_size": 4,
        "max_samples": None,
    },
    "best_accuracy": {
        "model": "qwen2.5-7b",
        "epochs": 15,
        "batch_size": 2,
        "max_samples": None,
    },
    "production": {
        "model": "qwen3-4b",  # When available
        "epochs": 15,
        "batch_size": 4,
        "max_samples": None,
    }
}


def get_config_for_use_case(use_case: str) -> dict:
    """
    Get recommended configuration for specific use case

    Args:
        use_case: One of 'quick_test', 'fast_iteration', 'best_accuracy', 'production'
    """
    if use_case not in CONFIGS_BY_USE_CASE:
        available = ', '.join(CONFIGS_BY_USE_CASE.keys())
        raise ValueError(f"Invalid use_case. Available: {available}")
    return CONFIGS_BY_USE_CASE[use_case]


# ==============================================================================
# PRINT CONFIGURATION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MTUP CONFIGURATION FOR VIETNAMESE AMR")
    print("=" * 80)
    print(f"\n📦 Model: {MODEL_NAME}")
    print(f"📊 Max Sequence Length: {MAX_SEQ_LENGTH}")
    print(f"🎯 LoRA Rank: {LORA_CONFIG['r']}")
    print(f"📈 Training Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"🎨 Template: {MTUP_CONFIG['template_name']}")
    print(f"📝 Batch Size: {TRAINING_CONFIG['per_device_train_batch_size']} × {TRAINING_CONFIG['gradient_accumulation_steps']} = {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")

    print(MODEL_COMPARISON)

    print("\n" + "=" * 80)
    print("QUICK START COMMANDS")
    print("=" * 80)
    print("\n# Quick test (1.5B model, 500 samples, 5 epochs)")
    print("python train_mtup.py --use-case quick_test")
    print("\n# Fast iteration (3B model, full data, 10 epochs) ⭐ RECOMMENDED")
    print("python train_mtup.py --use-case fast_iteration")
    print("\n# Best accuracy (7B model, full data, 15 epochs)")
    print("python train_mtup.py --use-case best_accuracy")
    print("\n# Custom model")
    print("python train_mtup.py --model qwen2.5-3b --epochs 12")
    print("=" * 80)
