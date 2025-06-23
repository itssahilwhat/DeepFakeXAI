import os
import csv
import logging
import time
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
    precision_recall_f1, dice_coefficient, iou_pytorch, save_mask_predictions, generate_gradcam
)


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
    kernel = torch.ones(1, 1, 3, 3).to(masks.device)
    for i in range(masks.shape[0]):
        mask = masks[i:i+1]
        edges = F.conv2d(mask, kernel, padding=1) > 0
        weights[i:i+1][edges] = 2.0
    return weights


def train_model(dataset_name):
    Config.setup_logging()
    logger = logging.getLogger("training")
    logger.info("🔎 Loading datasets with memory-safe batching...")

    dataset_list = dataset_name.split(",")
    train_loaders, valid_loaders, test_loaders = [], [], []

    for ds in dataset_list:
        ds = ds.strip()
        train_loader = get_dataloader(ds, "train")
        valid_loader = get_dataloader(ds, "valid")
        test_path = os.path.join(Config.DATA_ROOT, ds, "real", "test")
        if os.path.exists(test_path):
            test_loader = get_dataloader(ds, "test")
            test_loaders.append((ds, test_loader))
        else:
            logging.warning(f"⚠️ Skipping test split for '{ds}' — not found.")
        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)

    train_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in train_loaders])
    valid_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in valid_loaders])
    test_dataset = torch.utils.data.ConcatDataset([loader[1].dataset for loader in test_loaders]) if test_loaders else None

    logger.info(f"📦 Final sizes — Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset) if test_dataset else 0}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY) if test_dataset else None

    model = EfficientNetLiteTemporal(num_classes=1, pretrained=Config.PRETRAINED).to(Config.DEVICE)
    teacher = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to(Config.DEVICE)
    teacher.classifier = nn.Linear(teacher.classifier[1].in_features, 1).to(Config.DEVICE)
    teacher.eval()

    logger.info(f"🚀 Models moved to {Config.DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.OPTIMIZER_CONFIG["lr"],
                                 weight_decay=Config.OPTIMIZER_CONFIG["weight_decay"],
                                 betas=Config.OPTIMIZER_CONFIG["betas"])
    scheduler = StepLR(optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA)
    scaler = GradScaler(device=Config.DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2)
    mse_loss = nn.MSELoss()

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    csv_log_path = os.path.join(Config.LOG_DIR, "training_metrics.csv")
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "TrainLoss", "ValLoss", "Dice", "IoU", "Precision", "Recall", "F1"])

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_combined.pth")
    start_epoch, best_val_loss = 0, float("inf")
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(f"✅ Loaded checkpoint from '{checkpoint_path}' at epoch {start_epoch}")
    else:
        logger.info(f"⚠️ No checkpoint found at '{checkpoint_path}'. Starting training from scratch.")

    def log_vram_usage():
        if Config.DEVICE == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"🖥️ VRAM Usage: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")
            return allocated, reserved
        return 0, 0

    prev_outputs = None
    for epoch in range(start_epoch + 1, Config.EPOCHS + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=True, dynamic_ncols=True)

        for batch in pbar:
            log_vram_usage()
            images = batch["image"].to(Config.DEVICE)
            masks = batch["mask"].to(Config.DEVICE)
            weights = compute_boundary_weights(masks)

            optimizer.zero_grad()
            with autocast(device_type=Config.DEVICE):
                outputs = model(images)
                focal_loss = criterion(outputs, masks, weights)
                temporal_loss = 0
                if prev_outputs is not None:
                    temporal_loss = 0.1 * F.mse_loss(outputs, prev_outputs)

                pooled_student = F.avg_pool2d(outputs, kernel_size=outputs.shape[2:]).squeeze(-1).squeeze(-1)
                with torch.no_grad():
                    teacher_output = torch.sigmoid(teacher(images))
                distill_loss = mse_loss(pooled_student, teacher_output)

                loss = focal_loss + temporal_loss + 0.5 * distill_loss

            prev_outputs = outputs.detach()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        model.eval()
        val_losses, dice_scores, iou_scores, precisions, recalls, f1s = [], [], [], [], [], []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"[Epoch {epoch}] Validation", leave=True, dynamic_ncols=True):
                log_vram_usage()
                loss, preds, labels = validation_step(model, batch, criterion)
                val_losses.append(loss)
                dice_scores.append(dice_coefficient(preds, labels).item())
                iou_scores.append(iou_pytorch(preds, labels).mean().item())
                p, r, f1 = precision_recall_f1(preds, labels)
                precisions.append(p)
                recalls.append(r)
                f1s.append(f1)

        avg_val_loss = sum(val_losses) / len(val_losses)
        log_metrics = {
            "Dice": sum(dice_scores) / len(dice_scores),
            "IoU": sum(iou_scores) / len(iou_scores),
            "Precision": sum(precisions) / len(precisions),
            "Recall": sum(recalls) / len(recalls),
            "F1": sum(f1s) / len(f1s),
        }

        logger.info(f"📊 Epoch {epoch}: TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, "
                    f"Dice={log_metrics['Dice']:.4f}, IoU={log_metrics['IoU']:.4f}, "
                    f"Precision={log_metrics['Precision']:.4f}, Recall={log_metrics['Recall']:.4f}, F1={log_metrics['F1']:.4f}")

        with open(csv_log_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}", f"{log_metrics['Dice']:.4f}", f"{log_metrics['IoU']:.4f}", f"{log_metrics['Precision']:.4f}", f"{log_metrics['Recall']:.4f}", f"{log_metrics['F1']:.4f}"])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_val_loss
            }, filename=checkpoint_path)
            logger.info(f"✅ Saved checkpoint with ValLoss={best_val_loss:.4f} at '{checkpoint_path}'")

        scheduler.step()

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.2)

    model.eval()
    quantized_model = model  # You can replace with torch.quantization.convert if needed
    save_checkpoint({"state_dict": quantized_model.state_dict()}, filename=os.path.join(Config.CHECKPOINT_DIR, "quantized_model.pth"))
    logger.info("✅ Saved quantized model")

    if test_loader:
        logger.info("🧪 Evaluating on test set...")
        quantized_model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test Evaluation", dynamic_ncols=True):
                log_vram_usage()
                loss, preds, labels = test_step(quantized_model, batch, criterion)
                save_mask_predictions(batch["image"], batch["mask"], preds, os.path.join(Config.OUTPUT_DIR, "test_preds"))
                cam_map = generate_gradcam(quantized_model, batch["image"].to(Config.DEVICE), quantized_model.decoder.conv0)
                save_image(torch.from_numpy(cam_map).unsqueeze(0), os.path.join(Config.OUTPUT_DIR, f"cam_{int(time.time()*1000)}.png"))
    else:
        logger.info("⚠️ No test data available. Skipping test evaluation.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    train_model(args.dataset)
