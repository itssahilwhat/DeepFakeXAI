# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.config import Config
from src.dataset import get_dataloader
from src.model import MultiTaskDeepfakeModel
from src.losses import DiceLoss
from src.utils import accuracy, precision, recall, f1, dice_coef, iou_score, mae, pixel_accuracy, log_metrics
import numpy as np
import random
from collections import defaultdict
import glob


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train():
    set_seed(Config.RANDOM_SEED)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(os.path.join(Config.LOG_DIR, 'tensorboard'))

    # --- Data Loaders ---
    train_loader = get_dataloader('train', batch_size=Config.BATCH_SIZE)
    val_loader = get_dataloader('valid', batch_size=Config.BATCH_SIZE)
    print(f"Data loaded. Train: {len(train_loader.dataset)}, Validation: {len(val_loader.dataset)}")

    # --- Model, Losses, and Optimizer ---
    model = MultiTaskDeepfakeModel().to(Config.DEVICE)
    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=Config.AMP)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # --- RESUME LOGIC ---
    start_epoch = 0
    best_val_f1 = 0.0

    epoch_checkpoints = glob.glob(os.path.join(Config.CHECKPOINT_DIR, 'epoch_*.pth'))
    if epoch_checkpoints:
        latest_checkpoint = max(epoch_checkpoints, key=os.path.getctime)
        start_epoch = int(os.path.basename(latest_checkpoint).split('_')[1].split('.')[0])
        print(f"📄 Resuming training from epoch {start_epoch}...")
        checkpoint = torch.load(latest_checkpoint, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)

    patience = 0

    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        train_metrics = defaultdict(list)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Train]", leave=False)

        for imgs, masks, labels in pbar:
            imgs, masks, labels = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=Config.AMP):
                cls_logits, seg_logits = model(imgs)
                cls_targets = nn.functional.one_hot(labels, Config.NUM_CLASSES).float()
                loss = Config.LOSS_CLS_WEIGHT * bce_loss(cls_logits, cls_targets) + \
                       Config.LOSS_SEG_WEIGHT * dice_loss(seg_logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            # --- MODIFICATION: Calculate all metrics for training ---
            pred_classes = cls_logits.detach().argmax(axis=1).cpu().numpy()
            true_classes = labels.cpu().numpy()
            pred_masks = torch.sigmoid(seg_logits).detach().cpu().numpy() > 0.5
            true_masks = masks.cpu().numpy()

            train_metrics['loss'].append(loss.item())
            train_metrics['accuracy'].append(accuracy(true_classes, pred_classes))
            train_metrics['precision'].append(precision(true_classes, pred_classes))
            train_metrics['recall'].append(recall(true_classes, pred_classes))
            train_metrics['f1'].append(f1(true_classes, pred_classes))
            train_metrics['dice'].append(dice_coef(true_masks, pred_masks))
            train_metrics['iou'].append(iou_score(true_masks, pred_masks))
            # --------------------------------------------------------

        model.eval()
        val_metrics = defaultdict(list)
        with torch.no_grad():
            for imgs, masks, labels in val_loader:
                imgs, masks, labels = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), labels.to(Config.DEVICE)
                with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=Config.AMP):
                    cls_logits, seg_logits = model(imgs)
                    loss_val = Config.LOSS_CLS_WEIGHT * bce_loss(cls_logits, nn.functional.one_hot(labels, Config.NUM_CLASSES).float()) + \
                               Config.LOSS_SEG_WEIGHT * dice_loss(seg_logits, masks)


                # --- MODIFICATION: Calculate all metrics for validation ---
                pred_classes = cls_logits.argmax(axis=1).cpu().numpy()
                true_classes = labels.cpu().numpy()
                pred_masks = torch.sigmoid(seg_logits).cpu().numpy() > 0.5
                true_masks = masks.cpu().numpy()

                val_metrics['loss'].append(loss_val.item())
                val_metrics['accuracy'].append(accuracy(true_classes, pred_classes))
                val_metrics['precision'].append(precision(true_classes, pred_classes))
                val_metrics['recall'].append(recall(true_classes, pred_classes))
                val_metrics['f1'].append(f1(true_classes, pred_classes))
                val_metrics['dice'].append(dice_coef(true_masks, pred_masks))
                val_metrics['iou'].append(iou_score(true_masks, pred_masks))
                # ----------------------------------------------------------

        avg_train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
        avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

        log_metrics(avg_train_metrics, step=epoch + 1, prefix='Train')
        log_metrics(avg_val_metrics, step=epoch + 1, prefix='Validation')
        writer.add_scalars('Performance/F1_Score', {'train': avg_train_metrics['f1'], 'val': avg_val_metrics['f1']}, epoch + 1)
        writer.add_scalars('Performance/Dice_Coef', {'train': avg_train_metrics['dice'], 'val': avg_val_metrics['dice']}, epoch + 1)
        writer.add_scalars('Loss', {'train': avg_train_metrics['loss'], 'val': avg_val_metrics['loss']}, epoch + 1)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch + 1)

        scheduler.step()

        # --- CHECKPOINTING LOGIC ---
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
        }, os.path.join(Config.CHECKPOINT_DIR, f'epoch_{epoch + 1}.pth'))

        if avg_val_metrics['f1'] > best_val_f1:
            best_val_f1 = avg_val_metrics['f1']
            patience = 0
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'))
            print(f"✅ New best model saved with validation F1: {best_val_f1:.4f}")
        else:
            patience += 1
            if patience >= Config.EARLY_STOPPING:
                print(f"🟨 Early stopping at epoch {epoch + 1}.")
                break

    print(f"\n🏆 Project Complete. Best Validation F1: {best_val_f1:.4f}")
    writer.close()


if __name__ == "__main__":
    train()