# src/train.py

import os
import csv
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from torch.nn.utils import prune
from torchvision.utils import save_image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import autocast
from torch.amp import GradScaler
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from src.config import Config
from src.data import get_dataloader
from src.model import EfficientNetLiteTemporal
from src.utils import (
    save_checkpoint, load_checkpoint,
    precision_recall_f1, dice_coefficient, iou_pytorch,
    save_mask_predictions, generate_gradcam, generate_lime_overlay,
    MetricLogger
)
from src.losses import HybridLoss, ComboLoss
from src.advanced_metrics import AdvancedMetrics
from src.robustness_testing import RobustnessTester
from src.interpretability import InterpretabilityTools
from src.cross_dataset_evaluation import CrossDatasetEvaluator


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # For speed
    torch.backends.cudnn.benchmark = True  # Enable for speed


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, weights=None):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if weights is not None:
            F_loss = F_loss * weights
        return F_loss.mean()


def compute_boundary_weights(masks):
    weights = torch.ones_like(masks)
    kernel_size = Config.BOUNDARY_KERNEL_SIZE
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(masks.device)
    conv = F.conv2d(masks, kernel, padding=padding)
    edges = (conv > 0) & (conv < kernel_size ** 2)
    weights[edges] = 2.0
    return weights


def accuracy(preds, targets):
    correct = (preds == targets).float()
    return correct.sum() / correct.numel()


def move_to_device(batch, device):
    result = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            result[k] = v.to(device, non_blocking=True)
        else:
            result[k] = v
    return result


def train_model(dataset_names):
    Config.setup_logging()
    set_seeds()  # reproducibility

    logger = logging.getLogger("training")

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    logger.info(f"🔎 Loading datasets: {', '.join(dataset_names)}")

    train_datasets = []
    valid_datasets = []
    test_datasets = []

    for dataset_name in dataset_names:
        data_root = os.path.join(Config.DATA_ROOT, dataset_name)
        if not os.path.exists(data_root):
            raise RuntimeError(f"Dataset directory not found: {data_root}")

        train_loader = get_dataloader(dataset_name, "train", shuffle=True)
        valid_loader = get_dataloader(dataset_name, "valid", shuffle=False)
        test_loader = get_dataloader(dataset_name, "test", shuffle=False) \
            if os.path.exists(os.path.join(data_root, "real", "test")) else None

        train_datasets.append(train_loader.dataset)
        valid_datasets.append(valid_loader.dataset)
        if test_loader:
            test_datasets.append(test_loader.dataset)

    combined_train = torch.utils.data.ConcatDataset(train_datasets)
    combined_valid = torch.utils.data.ConcatDataset(valid_datasets)
    combined_test = torch.utils.data.ConcatDataset(test_datasets) if test_datasets else None

    # ── CAP combined datasets to desired sizes ──
    if Config.TRAIN_SIZE is not None and len(combined_train) > Config.TRAIN_SIZE:
        indices = random.sample(range(len(combined_train)), Config.TRAIN_SIZE)
        combined_train = torch.utils.data.Subset(combined_train, indices)
    if Config.VAL_SIZE is not None and len(combined_valid) > Config.VAL_SIZE:
        indices = random.sample(range(len(combined_valid)), Config.VAL_SIZE)
        combined_valid = torch.utils.data.Subset(combined_valid, indices)
    if combined_test and Config.TEST_SIZE is not None and len(combined_test) > Config.TEST_SIZE:
        indices = random.sample(range(len(combined_test)), Config.TEST_SIZE)
        combined_test = torch.utils.data.Subset(combined_test, indices)

    train_loader = torch.utils.data.DataLoader(
        combined_train,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Drop incomplete batches for consistent training
        persistent_workers=True  # Keep workers alive for speed
    )

    valid_loader = torch.utils.data.DataLoader(
        combined_valid,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True  # Keep workers alive for speed
    )

    test_loader = torch.utils.data.DataLoader(
        combined_test,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True  # Keep workers alive for speed
    ) if combined_test else None

    logger.info(
        f"📦 Combined dataset sizes - Train: {len(combined_train)} | "
        f"Valid: {len(combined_valid)} | "
        f"Test: {len(combined_test) if combined_test else 0}"
    )

    # Model and teacher
    model = EfficientNetLiteTemporal(num_classes=Config.NUM_CLASSES, pretrained=True).to(Config.DEVICE)
    
    # Initialize the segmentation head properly
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    # Apply initialization to segmentation head
    model.seg_head.apply(init_weights)
    
    # Clear any stranded cache before training
    if Config.DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.OPTIMIZER_CONFIG["lr"],
        weight_decay=Config.OPTIMIZER_CONFIG["weight_decay"],
        betas=Config.OPTIMIZER_CONFIG["betas"],
        eps=Config.OPTIMIZER_CONFIG["eps"],
        amsgrad=Config.OPTIMIZER_CONFIG.get("amsgrad", False)
    )
    
    # Fast scheduler for convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=Config.SCHEDULER_T0,  # Restart every 3 epochs
        T_mult=Config.SCHEDULER_T_MULT,  # Double the restart interval
        eta_min=Config.SCHEDULER_ETA_MIN  # Minimum learning rate
    )

    # Mixed precision scaler
    scaler = GradScaler("cuda") if Config.USE_AMP else None

    # Loss - Use combined loss for better metrics
    class CombinedLoss(nn.Module):
        def __init__(self, dice_weight=0.8, bce_weight=0.15, focal_weight=0.05):
            super().__init__()
            self.dice_weight = dice_weight
            self.bce_weight = bce_weight
            self.focal_weight = focal_weight
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.focal_loss = FocalLoss()
            
        def forward(self, pred, target):
            # BCE Loss
            bce = self.bce_loss(pred, target)
            
            # Dice Loss
            pred_sigmoid = torch.sigmoid(pred)
            dice = 1 - dice_coefficient(pred_sigmoid, target)
            
            # Focal Loss
            focal = self.focal_loss(pred, target)
            
            # Combined loss
            total_loss = (self.bce_weight * bce + 
                         self.dice_weight * dice + 
                         self.focal_weight * focal)
            
            return total_loss

    criterion = CombinedLoss(
        dice_weight=Config.LOSS_WEIGHTS["dice_weight"],
        bce_weight=Config.LOSS_WEIGHTS["bce_weight"],
        focal_weight=Config.LOSS_WEIGHTS["focal_weight"]
    )

    # Directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # TensorBoard setup (optional)
    if TENSORBOARD_AVAILABLE:
        tb_dir = os.path.join(Config.LOG_DIR, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        logger.info("✅ TensorBoard logging enabled")
    else:
        writer = None
        logger.info("⚠️ TensorBoard not available, skipping TB logging")

    # CSV logging setup
    csv_log_path = os.path.join(Config.LOG_DIR, "training_metrics.csv")
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch",
                "TrainLoss", "TrainDice", "TrainIoU", "TrainAccuracy",
                "ValLoss", "ValDice", "ValIoU", "ValAccuracy",
                "Precision", "Recall", "F1"
            ])

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_combined.pth")
    start_epoch, best_val_loss = 0, float("inf")
    epochs_no_improve = 0  # early stopping counter

    if os.path.exists(checkpoint_path):
        dummy_input = torch.randn(1, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
        _ = model(dummy_input)
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(f"✅ Loaded checkpoint from '{checkpoint_path}' at epoch {start_epoch}")
    else:
        logger.info(f"⚠️ No checkpoint found at '{checkpoint_path}'. Starting training from scratch.")

    def log_vram_usage():
        if Config.DEVICE == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
            tqdm.write(f"🖥️ VRAM: Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB, Max={max_allocated:.1f}MB")
            return allocated, reserved
        return 0, 0

    train_logger = MetricLogger(["loss", "dice", "iou", "acc"])
    val_logger = MetricLogger(["loss", "dice", "iou", "acc", "precision", "recall", "f1"])

    # ── MAIN TRAIN/VALIDATE LOOP ──
    for epoch in range(start_epoch + 1, Config.EPOCHS + 1):
        # Clear cache each epoch
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
        else:
            # Force garbage collection for CPU training
            import gc
            gc.collect()

                # TRAIN - OPTIMIZED FOR SPEED
        model.train()
        train_logger.reset()
        pbar = tqdm(
            enumerate(train_loader),
            desc=f"[Epoch {epoch}] Training",
            total=len(train_loader),
            leave=True,
            dynamic_ncols=True,
            smoothing=0.1
        )
        
        for batch_idx, batch in pbar:
            images = batch["image"].to(Config.DEVICE, non_blocking=True)
            masks  = batch["mask"].to(Config.DEVICE, non_blocking=True)
            
            if masks.sum() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass - OPTIMIZED
            if Config.USE_AMP:
                with autocast("cuda"):
                    cls_logits, seg_logits = model(images)
                    seg_loss = criterion(seg_logits, masks)
                    if cls_logits is not None:
                        cls_target = masks.mean(dim=[2, 3]).unsqueeze(1)
                        if cls_target.shape != cls_logits.shape:
                            cls_target = cls_target.squeeze(-1)
                        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_target)
                    else:
                        cls_loss = 0
                    loss = seg_loss + 0.05 * cls_loss if cls_logits is not None else seg_loss
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                cls_logits, seg_logits = model(images)
                seg_loss = criterion(seg_logits, masks)
                if cls_logits is not None:
                    cls_target = masks.mean(dim=[2, 3]).unsqueeze(1)
                    if cls_target.shape != cls_logits.shape:
                        cls_target = cls_target.squeeze(-1)
                    cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_target)
                else:
                    cls_loss = 0
                loss = seg_loss + 0.05 * cls_loss if cls_logits is not None else seg_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.MAX_GRAD_NORM)
                optimizer.step()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # Calculate metrics every few batches for speed - OPTIMIZED
            if batch_idx % 5 == 0:  # Calculate metrics every 5 batches for speed
                with torch.no_grad():
                    seg_output = torch.sigmoid(seg_logits.clamp(min=-20, max=20))
                    dice = dice_coefficient(seg_output, masks).item()
                    iou  = iou_pytorch(seg_output, masks).mean().item()
                    acc  = accuracy(seg_output > 0.5, masks).item()

                train_logger.update(
                    images.size(0),
                    loss=loss.item(),
                    dice=dice,
                    iou=iou,
                    acc=acc
                )
            else:
                # Just update loss for speed
                train_logger.update(
                    images.size(0),
                    loss=loss.item(),
                    dice=0,  # Will be averaged correctly
                    iou=0,   # Will be averaged correctly
                    acc=0    # Will be averaged correctly
                )
            
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_train = train_logger.avg()
        logger.info(
            f"🚀 Epoch {epoch} Training: "
            f"Loss={avg_train['loss']:.4f} | "
            f"Dice={avg_train['dice']:.4f} | "
            f"IoU={avg_train['iou']:.4f} | "
            f"Acc={avg_train['acc']:.4f}"
        )
        
        # TensorBoard logging
        if writer and hasattr(writer, 'add_scalar'):
            writer.add_scalar('Loss/Train', avg_train['loss'], epoch)
            writer.add_scalar('Dice/Train', avg_train['dice'], epoch)
            writer.add_scalar('IoU/Train', avg_train['iou'], epoch)
            writer.add_scalar('Accuracy/Train', avg_train['acc'], epoch)
        
        log_vram_usage()

        # VALIDATION - OPTIMIZED FOR SPEED
        val_logger.reset()
        model.eval()
        with torch.no_grad():
            val_pbar = tqdm(
                enumerate(valid_loader),
                desc=f"[Epoch {epoch}] Validation",
                total=len(valid_loader),
                leave=False,
                dynamic_ncols=True,
                smoothing=0.1
            )
            for _, batch in val_pbar:
                batch = move_to_device(batch, Config.DEVICE)

                images = batch.get("image")
                masks  = batch.get("mask")

                # Fast validation with mixed precision - OPTIMIZED
                if Config.USE_AMP:
                    with autocast("cuda"):
                        cls_logits, seg_logits = model(images)
                else:
                    cls_logits, seg_logits = model(images)
                
                seg_output = torch.sigmoid(seg_logits.clamp(min=-20, max=20))

                loss    = criterion(seg_logits, masks).item()
                dice    = dice_coefficient(seg_output, masks).item()
                iou     = iou_pytorch(seg_output, masks).mean().item()
                acc     = accuracy(seg_output > 0.5, masks).item()
                p, r, f1 = precision_recall_f1(seg_output > 0.5, masks)

                val_logger.update(
                    images.size(0),
                    loss=loss,
                    dice=dice,
                    iou=iou,
                    acc=acc,
                    precision=p,
                    recall=r,
                    f1=f1
                )
                val_pbar.set_postfix({'loss': f'{loss:.4f}', 'dice': f'{dice:.4f}'})

        avg_val = val_logger.avg()
        logger.info(
            f"📊 Epoch {epoch} Validation: "
            f"Loss={avg_val['loss']:.4f} | "
            f"Dice={avg_val['dice']:.4f} | "
            f"IoU={avg_val['iou']:.4f} | "
            f"Acc={avg_val['acc']:.4f} | "
            f"Precision={avg_val['precision']:.4f} | "
            f"Recall={avg_val['recall']:.4f} | "
            f"F1={avg_val['f1']:.4f}"
        )
        
        # TensorBoard logging
        if writer and hasattr(writer, 'add_scalar'):
            writer.add_scalar('Loss/Val', avg_val['loss'], epoch)
            writer.add_scalar('Dice/Val', avg_val['dice'], epoch)
            writer.add_scalar('IoU/Val', avg_val['iou'], epoch)
            writer.add_scalar('Accuracy/Val', avg_val['acc'], epoch)
            writer.add_scalar('Precision/Val', avg_val['precision'], epoch)
            writer.add_scalar('Recall/Val', avg_val['recall'], epoch)
            writer.add_scalar('F1/Val', avg_val['f1'], epoch)
        
        log_vram_usage()

        # Append to CSV
        with open(csv_log_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train['loss']:.4f}", f"{avg_train['dice']:.4f}",
                f"{avg_train['iou']:.4f}", f"{avg_train['acc']:.4f}",
                f"{avg_val['loss']:.4f}", f"{avg_val['dice']:.4f}",
                f"{avg_val['iou']:.4f}", f"{avg_val['acc']:.4f}",
                f"{avg_val['precision']:.4f}", f"{avg_val['recall']:.4f}",
                f"{avg_val['f1']:.4f}"
            ])

        # EARLY STOPPING & CHECKPOINTING
        if avg_val['loss'] < best_val_loss:
            best_val_loss = avg_val['loss']
            epochs_no_improve = 0
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_val_loss
            }, filename=checkpoint_path)
            logger.info(f"✅ Saved checkpoint with ValLoss={best_val_loss:.4f} at '{checkpoint_path}'")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
                logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                break

        scheduler.step()

    # ── TESTING ──
    if test_loader is not None:
        logger.info("🧪 Evaluating on test set...")
        test_logger = MetricLogger(["loss", "dice", "iou", "acc", "precision", "recall", "f1"])
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = move_to_device(batch, Config.DEVICE)
                images = batch.get("image")
                masks = batch.get("mask")

                if Config.USE_AMP:
                    with autocast("cuda"):
                        cls_logits, seg_logits = model(images)
                else:
                    cls_logits, seg_logits = model(images)

                seg_output = torch.sigmoid(seg_logits.clamp(min=-20, max=20))

                loss = criterion(seg_logits, masks).item()
                dice = dice_coefficient(seg_output, masks).item()
                iou = iou_pytorch(seg_output, masks).mean().item()
                acc = accuracy(seg_output > 0.5, masks).item()
                p, r, f1 = precision_recall_f1(seg_output > 0.5, masks)

                test_logger.update(
                    images.size(0),
                    loss=loss,
                    dice=dice,
                    iou=iou,
                    acc=acc,
                    precision=p,
                    recall=r,
                    f1=f1
                )

        avg_test = test_logger.avg()
        logger.info(
            f"🎯 Final Test Results: "
            f"Loss={avg_test['loss']:.4f} | "
            f"Dice={avg_test['dice']:.4f} | "
            f"IoU={avg_test['iou']:.4f} | "
            f"Acc={avg_test['acc']:.4f} | "
            f"Precision={avg_test['precision']:.4f} | "
            f"Recall={avg_test['recall']:.4f} | "
            f"F1={avg_test['f1']:.4f}"
        )

        # Save test results
        test_csv_path = os.path.join(Config.LOG_DIR, "test_results.csv")
        with open(test_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Loss", f"{avg_test['loss']:.4f}"])
            writer.writerow(["Dice", f"{avg_test['dice']:.4f}"])
            writer.writerow(["IoU", f"{avg_test['iou']:.4f}"])
            writer.writerow(["Accuracy", f"{avg_test['acc']:.4f}"])
            writer.writerow(["Precision", f"{avg_test['precision']:.4f}"])
            writer.writerow(["Recall", f"{avg_test['recall']:.4f}"])
            writer.writerow(["F1", f"{avg_test['f1']:.4f}"])

    logger.info("🎉 Training completed successfully!")
    
    # Close TensorBoard writer properly
    if writer and hasattr(writer, 'close'):
        writer.close()
    
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of dataset names (e.g., 'celebahq,ffhq')"
    )
    args = parser.parse_args()
    train_model(args.datasets.split(','))