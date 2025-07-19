# scripts/hyperparam_sweep.py

import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
import copy

# Setup paths before importing local modules
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.model import MultiTaskDeepfakeModel
from src.dataset import DeepfakeDataset
from src.losses import DiceLoss

# --- HYPERPARAMETER SEARCH SPACE ---
SEARCH_SPACE = {
    'lr': (1e-4, 5e-3),
    'dropout': (0.2, 0.5),
    'weight_decay': (1e-5, 1e-3),
    'loss_seg_weight': (1.5, 4.0),
}


# --- END OF SEARCH SPACE ---

def run_trial(params):
    """
    Runs a single training and validation trial and returns metrics and model weights.
    """
    print(f"\n🚀 Starting Trial with params: { {k: f'{v:.5f}' for k, v in params.items()} }")

    # --- Setup Model ---
    model = MultiTaskDeepfakeModel(
        backbone_name='mobilenetv3_small_100',
        dropout=params['dropout']
    ).to(Config.DEVICE)

    # Re-enabled multiprocessing for speed
    train_ds = DeepfakeDataset(split='train')
    val_ds = DeepfakeDataset(split='valid')
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS,
                            pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scaler = torch.amp.GradScaler(enabled=True)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    # --- Train for 1 Epoch ---
    model.train()
    for imgs, masks, labels in tqdm(train_loader, desc="Training 1 Epoch", leave=False):
        imgs, masks, labels = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), labels.to(Config.DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=True):
            cls_logits, seg_logits = model(imgs)
            loss_cls = bce_loss(cls_logits, torch.nn.functional.one_hot(labels, 2).float())
            loss_seg = dice_loss(seg_logits, masks)
            loss = loss_cls + params['loss_seg_weight'] * loss_seg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # --- Evaluate on Validation Set ---
    model.eval()
    val_metrics = defaultdict(list)
    with torch.no_grad():
        for imgs, masks, labels in tqdm(val_loader, desc="Validating", leave=False):
            imgs, masks, labels = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), labels.to(Config.DEVICE)
            with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=True):
                cls_logits, seg_logits = model(imgs)

            from src.utils import f1, dice_coef, precision, recall, iou_score
            pred_classes = cls_logits.argmax(axis=1).cpu().numpy()
            true_classes = labels.cpu().numpy()
            pred_masks = torch.sigmoid(seg_logits).cpu().numpy() > 0.5
            true_masks = masks.cpu().numpy()

            val_metrics['val_f1'].append(f1(true_classes, pred_classes))
            val_metrics['val_dice'].append(dice_coef(true_masks, pred_masks))
            val_metrics['val_precision'].append(precision(true_classes, pred_classes))
            val_metrics['val_recall'].append(recall(true_classes, pred_classes))
            val_metrics['val_iou'].append(iou_score(true_masks, pred_masks))

    trial_results = {k: np.mean(v) for k, v in val_metrics.items()}
    return trial_results, model.cpu().state_dict()


def main():
    parser = argparse.ArgumentParser(description="Resumable Hyperparameter Sweep")
    parser.add_argument('--num-trials', type=int, default=20, help="Total number of random configurations to test.")
    args = parser.parse_args()

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    output_path = os.path.join(Config.LOG_DIR, "hyperparameter_sweep_results.csv")

    # --- RESUME LOGIC ---
    results = []
    start_trial = 1
    if os.path.exists(output_path):
        print(f"📄 Found existing results file. Resuming sweep from last point.")
        results_df = pd.read_csv(output_path)
        results = results_df.to_dict('records')
        start_trial = len(results) + 1
    # -------------------

    best_f1 = -1.0
    best_state_dict = None

    for i in range(start_trial, args.num_trials + 1):
        print(f"\n--- Trial {i}/{args.num_trials} ---")
        params = {
            'lr': 10 ** random.uniform(np.log10(SEARCH_SPACE['lr'][0]), np.log10(SEARCH_SPACE['lr'][1])),
            'dropout': random.uniform(SEARCH_SPACE['dropout'][0], SEARCH_SPACE['dropout'][1]),
            'weight_decay': 10 ** random.uniform(np.log10(SEARCH_SPACE['weight_decay'][0]),
                                                 np.log10(SEARCH_SPACE['weight_decay'][1])),
            'loss_seg_weight': random.uniform(SEARCH_SPACE['loss_seg_weight'][0], SEARCH_SPACE['loss_seg_weight'][1])
        }

        trial_results, current_state_dict = run_trial(params)

        # Save checkpoint for the current trial
        trial_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'sweep_trial_{i}.pth')
        torch.save(current_state_dict, trial_checkpoint_path)

        # Check for new best model
        if trial_results['val_f1'] > best_f1:
            # Also check if a previous best existed in a resumed run
            if len(results) > 0:
                best_f1_in_file = pd.DataFrame(results)['val_f1'].max()
                if trial_results['val_f1'] > best_f1_in_file:
                    best_f1 = trial_results['val_f1']
                    best_state_dict = current_state_dict
                    print(f"✅ New best trial found with Val F1: {best_f1:.4f}")
            else:  # First run
                best_f1 = trial_results['val_f1']
                best_state_dict = current_state_dict
                print(f"✅ New best trial found with Val F1: {best_f1:.4f}")

        results.append({**params, **trial_results})

        # --- SAVE RESULTS AFTER EVERY TRIAL ---
        pd.DataFrame(results).to_csv(output_path, index=False)
        # --------------------------------------

    # --- Save the single best model from all trials ---
    if best_state_dict:
        # Load the best model from the whole file to be sure
        final_df = pd.read_csv(output_path)
        best_trial_index = final_df['val_f1'].idxmax()
        best_trial_num = best_trial_index + 1

        best_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"sweep_trial_{best_trial_num}.pth")
        if os.path.exists(best_checkpoint_path):
            best_state_dict = torch.load(best_checkpoint_path)
            save_path = os.path.join(Config.CHECKPOINT_DIR, "best_sweep_model.pth")
            torch.save(best_state_dict, save_path)
            print(f"\n✅ Best model (Trial {best_trial_num}) from sweep saved to: {save_path}")

    # --- Final Summary ---
    final_df = pd.read_csv(output_path)
    final_df = final_df.sort_values(by='val_f1', ascending=False)

    print("\n\n" + "=" * 50)
    print("🏆 HYPERPARAMETER SWEEP COMPLETE 🏆")
    print(f"Results CSV saved to: {output_path}")
    print("\n--- Top 5 Performing Configurations ---")
    print(final_df.head(5).to_string())
    print("=" * 50)


if __name__ == "__main__":
    main()