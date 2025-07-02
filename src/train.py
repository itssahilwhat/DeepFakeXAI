import os
import csv
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.nn.utils import prune
from torchvision.utils import save_image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from config import Config
from data import get_dataloader
from model import EfficientNetLiteTemporal
from utils import (
    save_checkpoint, load_checkpoint, validation_step, test_step,
    precision_recall_f1, dice_coefficient, iou_pytorch,
    save_mask_predictions, generate_gradcam, generate_pseudo_masks,
    evaluate_metrics, MetricLogger
)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # For modern GPUs
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False  # For speed

# ✅ 4️⃣ Reproducibility seeds
def set_seeds(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# ✅ 1️⃣ Utility function to move all tensors to device
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
    set_seeds()  # ✅ 4️⃣

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

    train_loader = torch.utils.data.DataLoader(
        combined_train,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=2 if Config.DEVICE == "cuda" else None,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )

    valid_loader = torch.utils.data.DataLoader(
        combined_valid,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=2 if Config.DEVICE == "cuda" else None,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )

    test_loader = torch.utils.data.DataLoader(
        combined_test,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=2 if Config.DEVICE == "cuda" else None,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    ) if combined_test else None

    logger.info(
        f"📦 Combined dataset sizes - Train: {len(combined_train)} | "
        f"Valid: {len(combined_valid)} | "
        f"Test: {len(combined_test) if combined_test else 0}"
    )

    model = EfficientNetLiteTemporal(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED).to(Config.DEVICE)
    teacher = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to(Config.DEVICE)
    teacher.classifier = nn.Linear(teacher.classifier[1].in_features, 1).to(Config.DEVICE)
    teacher.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.OPTIMIZER_CONFIG["lr"],
        weight_decay=Config.OPTIMIZER_CONFIG["weight_decay"],
        betas=Config.OPTIMIZER_CONFIG["betas"],
        amsgrad=Config.OPTIMIZER_CONFIG.get("amsgrad", False)
    )
    scheduler = StepLR(optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA)
    scaler = GradScaler()
    criterion = FocalLoss(alpha=0.25, gamma=2)
    mse_loss = nn.MSELoss()

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

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
            tqdm.write(f"🖥️ VRAM Usage: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")
            return allocated, reserved
        return 0, 0

    train_logger = MetricLogger(["loss", "dice", "iou", "acc"])
    val_logger = MetricLogger(["loss", "dice", "iou", "acc", "precision", "recall", "f1"])

    prev_outputs = None

    for epoch in range(start_epoch + 1, Config.EPOCHS + 1):
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
            # Move data with non_blocking and channels_last
            images = batch["image"].to(Config.DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            masks = batch["mask"].to(Config.DEVICE, non_blocking=True)
            labels = batch["label"].to(Config.DEVICE, non_blocking=True)

            weights = compute_boundary_weights(masks)

            if Config.USE_WEAK_SUPERVISION:
                with torch.no_grad():
                    _, pseudo_masks = model(images)
                valid_mask = (masks is not None).float()
                final_masks = valid_mask * masks + (1 - valid_mask) * (pseudo_masks > 0.5).float()
            else:
                final_masks = masks

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                cls_output, seg_logits = model(
                    images,
                    prev_x=prev_outputs if prev_outputs is not None and prev_outputs.size(0) == images.size(0) else None
                )

                seg_loss = criterion(seg_logits, final_masks, weights)

                cls_loss = F.binary_cross_entropy_with_logits(cls_output, labels.float().view(-1, 1)) \
                    if Config.USE_COLLABORATIVE and cls_output is not None else 0

                temporal_loss = 0.1 * F.mse_loss(seg_logits, prev_outputs) \
                    if prev_outputs is not None and prev_outputs.size(0) == seg_logits.size(0) else 0

                with torch.no_grad():
                    teacher_output = torch.sigmoid(teacher(images))
                distill_loss = 0.5 * mse_loss(
                    F.avg_pool2d(seg_logits, kernel_size=seg_logits.shape[2:]).squeeze(-1).squeeze(-1),
                    teacher_output
                )

                loss = seg_loss + cls_loss + temporal_loss + distill_loss

            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            prev_outputs = seg_logits.detach()

            with torch.no_grad():
                seg_output = torch.sigmoid(seg_logits)
                dice = dice_coefficient(seg_output, final_masks).item()
                iou = iou_pytorch(seg_output, final_masks).mean().item()
                acc = accuracy(seg_output > 0.5, final_masks).item()

            train_logger.update(
                images.size(0),
                loss=loss.item(),
                dice=dice,
                iou=iou,
                acc=acc
            )

            pbar.set_postfix(loss=loss.item(), dice=dice)

        avg_train = train_logger.avg()
        logger.info(
            f"🚀 Epoch {epoch} Training: "
            f"Loss={avg_train['loss']:.4f} | "
            f"Dice={avg_train['dice']:.4f} | "
            f"IoU={avg_train['iou']:.4f} | "
            f"Acc={avg_train['acc']:.4f}"
        )
        log_vram_usage()

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
                masks = batch.get("mask")

                _, seg_logits = model(images)
                seg_output = torch.sigmoid(seg_logits)

                loss = criterion(seg_logits, masks).item()
                dice = dice_coefficient(seg_output, masks).item()
                iou = iou_pytorch(seg_output, masks).mean().item()
                acc = accuracy(seg_output > 0.5, masks).item()
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


        # Log validation metrics
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
        log_vram_usage()
        # Save to CSV
        with open(csv_log_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train['loss']:.4f}", f"{avg_train['dice']:.4f}",
                f"{avg_train['iou']:.4f}", f"{avg_train['acc']:.4f}",
                f"{avg_val['loss']:.4f}", f"{avg_val['dice']:.4f}",
                f"{avg_val['iou']:.4f}", f"{avg_val['acc']:.4f}",
                f"{avg_val['precision']:.4f}", f"{avg_val['recall']:.4f}", f"{avg_val['f1']:.4f}"
            ])

        if avg_val['loss'] < best_val_loss:
            best_val_loss = avg_val['loss']
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_val_loss
            }, filename=checkpoint_path)
            logger.info(f"✅ Saved checkpoint with ValLoss={best_val_loss:.4f} at '{checkpoint_path}'")

        scheduler.step()

        # Testing
    if test_loader is not None:
        logger.info("🧪 Evaluating on test set...")
        test_logger = MetricLogger(["loss", "dice", "iou", "acc", "precision", "recall", "f1"])
        model.eval()
        with torch.no_grad():
            test_pbar = tqdm(
                enumerate(test_loader),
                desc="Test Evaluation",
                total=len(test_loader),
                leave=True,
                dynamic_ncols=True,
                smoothing=0.1
            )

            for _, batch in test_pbar:
                log_vram_usage()  # ✅ Log VRAM during test
                batch = move_to_device(batch, Config.DEVICE)

                images = batch.get("image")
                masks = batch.get("mask")

                _, seg_logits = model(images)
                seg_output = torch.sigmoid(seg_logits)

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
                test_pbar.set_postfix({'loss': f'{loss:.4f}', 'dice': f'{dice:.4f}'})

        avg_test = test_logger.avg()
        logger.info(
            f"🧪 Test Metrics: "
            f"Loss={avg_test['loss']:.4f} | "
            f"Dice={avg_test['dice']:.4f} | "
            f"IoU={avg_test['iou']:.4f} | "
            f"Acc={avg_test['acc']:.4f} | "
            f"Precision={avg_test['precision']:.4f} | "
            f"Recall={avg_test['recall']:.4f} | "
            f"F1={avg_test['f1']:.4f}"
        )

        # Save test results to CSV
        test_csv_path = os.path.join(Config.LOG_DIR, "test_results.csv")
        with open(test_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for metric, value in avg_test.items():
                writer.writerow([metric, f"{value:.4f}"])

        # Save sample predictions
    logger.info("💾 Saving sample predictions...")
    sample_dir = os.path.join(Config.OUTPUT_DIR, "sample_predictions")
    os.makedirs(sample_dir, exist_ok=True)

    sample_batch = next(iter(test_loader)) if test_loader else next(iter(valid_loader))
    sample_batch = move_to_device(sample_batch, Config.DEVICE)

    images = sample_batch["image"]
    masks = sample_batch["mask"]

    with torch.no_grad():
        _, seg_logits = model(images)
        predictions = torch.sigmoid(seg_logits).cpu()

    save_mask_predictions(
        images.cpu(),
        masks.cpu(),
        predictions,
        out_dir=sample_dir
    )

    logger.info("🏁 Training completed successfully!")


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