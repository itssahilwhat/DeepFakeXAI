import os
import torch

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'wacv_data')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs', 'training')
    CHECKPOINT_DIR = os.path.join(LOG_DIR, 'checkpoints')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    
    # Workflow configuration
    WORKFLOW = 'classification'  # 'classification', 'segmentation', 'xai'
    MODEL_TYPE = 'resnet_se'  # 'resnet_se' for ResNet-34 with SE-blocks
    
    # Data configuration
    IMG_SIZE = 224  # Standard size for SE-CNN
    BATCH_SIZE = 8  # Reduced from 16 to fit in 3.68 GiB GPU
    ACCUMULATION_STEPS = 2  # Increased to maintain effective batch size
    NUM_WORKERS = 8
    RANDOM_SEED = 42
    PIN_MEMORY = True
    
    # Training strategy
    USE_STRATIFIED_SAMPLING = False  # Disabled to avoid conflict with WeightedRandomSampler
    USE_FOCAL_LOSS = True
    USE_LABEL_SMOOTHING = True
    USE_MIXUP = True
    USE_CUTMIX = True
    USE_WEIGHTED_SAMPLING = True  # Keep this for balanced sampling
    USE_EMA = True
    
    # Loss configuration
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    EMA_DECAY = 0.999
    
    # SE-CNN configuration (Dasgupta et al., 2025)
    SE_CNN_FILTERS = [32, 64, 128, 256, 512]
    SE_CNN_REDUCTION_RATIO = 16
    SE_CNN_DROPOUT = 0.3
    
    # Alternative CNN configuration
    PRETRAINED_BACKBONE = 'efficientnet_lite0'  # 'efficientnet_lite0', 'densenet121', 'resnet50'
    PRETRAINED_DROPOUT = 0.3
    PRETRAINED_NUM_CLASSES = 2
    
    # Segmentation configuration
    SEGMENTATION_BACKBONE = 'efficientnet_b0'
    SEGMENTATION_PRETRAINED = True
    SEGMENTATION_DROPOUT = 0.2
    SEGMENTATION_NUM_CLASSES = 1
    UNET_FILTERS = [64, 128, 256, 512]
    
    # Training Configuration
    EPOCHS = 100
    LEARNING_RATE = 1e-4  # As per literature
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    AMP = True  # Mixed precision to save memory
    SCHEDULER = 'cosine_warmup'
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing to save memory
    
    # Loss weights
    LOSS_CLS_WEIGHT = 1.0
    LOSS_SEG_WEIGHT = 1.0
    LOSS_FOCAL_WEIGHT = 1.0
    LOSS_DICE_WEIGHT = 0.5
    
    # Advanced techniques
    USE_CROSS_VALIDATION = True
    CV_FOLDS = 5
    USE_TEST_TIME_AUGMENTATION = True
    USE_MODEL_ENSEMBLING = True
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # XAI configuration
    XAI_ENABLED = True
    XAI_METHODS = ['grad_cam', 'lime', 'shap']
    XAI_TARGET_LAYER = 'final_conv'  # For SE-CNN
    XAI_EVAL_SAMPLES = 100
    XAI_FLIP_SCORE_BATCH_SIZE = 8
    
    # Evaluation
    JPEG_QUALITIES = [10, 20, 30, 50, 70, 90]
    NOISE_LEVELS = [5, 20, 30, 40, 50]
    BLUR_LEVELS = [3, 5, 7, 9, 11]
    
    # Training
    COMPILE = False
    EARLY_STOPPING_PATIENCE = 15
    SAVE_BEST_MODELS = 5
    
    # Performance targets (from literature)
    TARGET_ACCURACY = 0.94  # Dasgupta et al., 2025
    TARGET_F1 = 0.94
    TARGET_AUC = 0.985  # Dasgupta et al., 2025
    TARGET_IOU = 0.90
    TARGET_DICE = 0.90
    
    # Cross-dataset evaluation
    CROSS_GENERATOR_EVAL = True
    GENERATORS = ['p2', 'repaint-p2', 'lama', 'pluralistic', 'ldm']
    EXTERNAL_DATASETS = ['faceforensics', 'celebdf']
    
    # Edge deployment
    EXPORT_ONNX = True
    EXPORT_TFLITE = True
    QUANTIZATION = True
    TARGET_MODEL_SIZE_MB = 10
    TARGET_INFERENCE_TIME_MS = 100
    
    # Dataset
    MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'wacv_data', 'manifest.csv')
