import os
import logging
import torch


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "wacv_data")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    ONNX_DIR = os.path.join(PROJECT_ROOT, "onnx_models")

    DATASET_CONFIGS = {
        "celebahq": {"has_masks": True, "use_temporal": False},
        "ffhq": {"has_masks": True, "use_temporal": False}
    }

    MODEL_NAME = "EfficientNetLiteTemporal"
    NUM_CLASSES = 1
    INPUT_SIZE = (224, 224)
    PRETRAINED = True

    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    EPOCHS = 5
    WEIGHT_DECAY = 1e-5

    OPTIMIZER_CONFIG = {
        "type": "Adam",
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "betas": (0.9, 0.999)
    }

    MAX_TRAIN_SAMPLES = 60000
    MAX_VALID_SAMPLES = 6000

    SCHEDULER_STEP_SIZE = 5
    SCHEDULER_GAMMA = 0.1

    PIN_MEMORY = True
    NUM_WORKERS = 4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup_logging(log_level=logging.INFO):
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()]
        )