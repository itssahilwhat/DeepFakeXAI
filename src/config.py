# src/config.py

import os
import torch

class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'wacv_data')
    CELEBAHQ_PATH = os.path.join(DATA_ROOT, 'celebahq')
    FFHQ_PATH = os.path.join(DATA_ROOT, 'ffhq')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    CHECKPOINT_DIR = os.path.join(LOG_DIR, 'checkpoints')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

    # Data
    IMG_SIZE = 224
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    RANDOM_SEED = 42

    # Model - Using Hyperparameter Sweep Results
    BACKBONE = 'mobilenetv3_small_100'
    PRETRAINED = True
    DROPOUT = 0.25
    SEGMENTATION = True
    NUM_CLASSES = 2

    # --- FINAL OPTIMIZED TRAINING STRATEGY ---
    EPOCHS = 40
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 2.1e-4
    GRAD_CLIP = 1.0
    AMP = True
    SCHEDULER = 'cosine'
    EARLY_STOPPING = 7

    # Loss Weights
    LOSS_CLS_WEIGHT = 1.0
    LOSS_SEG_WEIGHT = 2.2

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PIN_MEMORY = True

    # --- XAI & Evaluation Configuration ---
    XAI_ENABLED = True
    XAI_TARGET_LAYER = 'backbone.blocks.5'
    XAI_EVAL_SAMPLES = 50
    XAI_FLIP_SCORE_BATCH_SIZE = 16