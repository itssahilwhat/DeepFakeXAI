import os

# Compute paths relative to this config file for robustness
config_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(config_dir)

# Paths
data_root     = os.path.join(project_root, "data/wacv_data")
manifest_csv  = os.path.join(data_root, "manifest.csv")
dir_ckpt      = os.path.join(project_root, "checkpoints"); os.makedirs(dir_ckpt, exist_ok=True)
images_out    = os.path.join(project_root, "outputs"); os.makedirs(images_out, exist_ok=True)

# Hyperparameters - Optimized for RTX 3050 4GB VRAM
batch_size    = 16  # Optimized for RTX 3050 based on testing
lr            = 1e-4
epochs        = 30
seg_loss_w    = 0.5

# Models
det_backbone  = "convnext_tiny"
img_size      = (224, 224)

# Balancing
balanced_batch = True