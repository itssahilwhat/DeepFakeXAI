import os
import logging
import torch


class Config:
    # Directory setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "wacv_data")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    ONNX_DIR = os.path.join(PROJECT_ROOT, "onnx_models")

    # Dataset configurations
    DATASET_CONFIGS = {
        "celebahq": {"has_masks": True, "use_temporal": False},
        "ffhq": {"has_masks": True, "use_temporal": False},
        "dolos": {"has_masks": True, "use_temporal": True}
    }

    # Dataset sizes - OPTIMIZED FOR 90%+ METRICS AND 100+ IT/SEC
    TRAIN_SIZE = 50000  # Increased from 25000 for better metrics
    VAL_SIZE = 5000     # Increased from 2500 for better validation
    TEST_SIZE = 5000    # Increased from 2500 for better testing

    # ===== ULTRA-SPEED OPTIMIZATION FOR 100+ IT/SEC =====
    MODEL_NAME = "EfficientNetLiteTemporal"
    NUM_CLASSES = 1
    INPUT_SIZE = (160, 160)  # Reduced from 192x192 for maximum speed boost
    PRETRAINED = True
    
    # Ultra-optimized batch sizing for 100+ it/sec
    BATCH_SIZE = 80  # Increased from 64 for better GPU utilization
    ACCUMULATION_STEPS = 1  # No gradient accumulation needed
    
    # Optimized learning parameters for fast convergence
    LEARNING_RATE = 5e-4  # Increased from 3e-4 for faster convergence
    EPOCHS = 12  # Reduced from 15 for faster training
    WEIGHT_DECAY = 1e-5  # Reduced for faster convergence
    
    # Memory optimization - ULTRA SPEED
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True  # Enable for faster data transfer
    NUM_WORKERS = 8  # Optimized to 8 based on speed test results
    PRESERVE_ORIGINAL = False
    PRESERVE_TEST_ORIGINAL = True
    
    # Mixed precision training - CRITICAL FOR SPEED
    USE_AMP = True  # Automatic Mixed Precision
    GRADIENT_CHECKPOINTING = False  # Disabled for speed
    
    # Scheduler parameters - Fast convergence
    SCHEDULER_TYPE = "cosine_annealing_warm_restarts"
    SCHEDULER_T0 = 4  # Reduced from 5 for faster convergence
    SCHEDULER_T_MULT = 2  # Double restart interval
    SCHEDULER_ETA_MIN = 1e-6  # Increased minimum learning rate

    # Enhanced features - ULTRA SPEED OPTIMIZED
    USE_COLLABORATIVE = True  # Enable for better performance
    USE_CLIP_EXPLAINER = False  # Disabled for speed
    USE_WEAK_SUPERVISION = True  # Enable weak supervision
    USE_DIFFUSION_AUG = False  # Disabled for speed
    USE_DICE_LOSS = True  # Enable Dice loss
    USE_FOCAL_LOSS = False  # Disabled for speed
    
    # Temporal consistency - DISABLED FOR SPEED
    TEMPORAL_WINDOW_SIZE = 0  # Disable temporal consistency
    TEMPORAL_WEIGHT = 0.0  # No temporal loss
    
    # Augmentation control - MINIMAL FOR SPEED
    AUGMENT_TRAIN = True
    AUGMENT_VALID = False  # Disable for faster validation
    AUGMENT_TEST = False

    # Model parameters - SPEED OPTIMIZED
    BOUNDARY_KERNEL_SIZE = 3  # Reduced for speed
    DROPOUT_RATE = 0.03  # Reduced from 0.05 for speed
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 6  # Reduced from 8 for faster training
    
    # Robustness testing parameters
    ROBUSTNESS_NOISE_LEVELS = [0, 5, 10, 15, 20]
    ROBUSTNESS_COMPRESSION_LEVELS = [95, 85, 75, 65, 55]
    ROBUSTNESS_BLUR_LEVELS = [0, 2, 5, 8, 11]

    # XAI parameters - ULTRA SPEED
    LIME_NUM_SAMPLES = 50  # Reduced from 100 for speed
    LIME_HIDE_COLOR = 0
    GRADCAM_TARGET_LAYER = "backbone.features.0"  # Use first Conv2d for guaranteed GradCAM compatibility
    
    # CLIP parameters
    CLIP_MODEL_NAME = "ViT-B/32"  # Efficient CLIP model
    
    # Optimizer configuration - ULTRA SPEED
    OPTIMIZER_CONFIG = {
        "type": "AdamW",
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False  # Disable for memory efficiency
    }
    
    # Loss function weights - OPTIMIZED FOR 90%+ ACCURACY
    LOSS_WEIGHTS = {
        "dice_weight": 0.9,  # Increased from 0.8 for better segmentation
        "bce_weight": 0.1,  # Reduced from 0.15
        "focal_weight": 0.0, # Disabled for speed
        "temporal_weight": TEMPORAL_WEIGHT
    }
    
    # Performance monitoring - ULTRA SPEED
    LOG_INTERVAL = 20  # Increased from 10 for less overhead
    SAVE_INTERVAL = 2  # Save every 2 epochs
    
    # Memory management
    CLEAR_CACHE_INTERVAL = 3  # Reduced from 5 for more frequent cleanup
    MAX_GRAD_NORM = 0.3  # Reduced from 0.5 for better stability

    @staticmethod
    def setup_logging(log_level=logging.INFO):
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()]
        )