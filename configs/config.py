import os
# Paths
data_root     = "data/wacv_data"
manifest_csv  = os.path.join(data_root, "manifest.csv")
dir_ckpt      = "checkpoints"; os.makedirs(dir_ckpt, exist_ok=True)
images_out    = "outputs"; os.makedirs(images_out, exist_ok=True)
# Hyperparameters
batch_size    = 32
lr            = 1e-4
epochs        = 30
seg_loss_w    = 0.5
# Models
det_backbone  = "convnext_tiny"
img_size      = (224, 224)
# Balancing
balanced_batch = True