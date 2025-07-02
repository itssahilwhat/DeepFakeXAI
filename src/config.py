# src/config.py
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
        "dolos": {"has_masks": True, "use_temporal": True}  # Enable temporal for Dolos
    }

    # Model and training parameters
    MODEL_NAME = "EfficientNetLiteTemporal"
    NUM_CLASSES = 1
    INPUT_SIZE = (224, 224)
    PRETRAINED = True
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    WEIGHT_DECAY = 5e-5
    PIN_MEMORY = True
    NUM_WORKERS = min(12, os.cpu_count())
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PRESERVE_ORIGINAL = False
    # Scheduler parameters
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.2

    # Enhancement flags
    USE_COLLABORATIVE = True
    USE_CLIP_EXPLAINER = True
    USE_WEAK_SUPERVISION = False
    USE_DIFFUSION_AUG = True
    TRAIN_SIZE = 80000
    VAL_SIZE = 10000
    TEST_SIZE = 10000
    TEMPORAL_WINDOW_SIZE = 3  # New: Temporal frames to consider
    BOUNDARY_KERNEL_SIZE = 5  # New: Kernel size for edge detection

    # Optimizer configuration
    OPTIMIZER_CONFIG = {
        "type": "Adam",
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "betas": (0.9, 0.999),
        "amsgrad": True  # Better stability
    }

    @staticmethod
    def setup_logging(log_level=logging.INFO):
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()]
        )