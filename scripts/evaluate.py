# scripts/evaluate.py

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import cv2

# Setup paths
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Imports ---
from src.config import Config
from src.model import MultiTaskDeepfakeModel
from src.dataset import get_dataloader
from torch.utils.data import DataLoader
from src.utils import accuracy, precision, recall, f1, dice_coef, iou_score, mae, pixel_accuracy, log_metrics
from src.explainability import GradCAM


def compute_flip_score(model, image, explanation_map, batch_size=16):
    """
    Computes the flip score (deletion metric) for a given explanation.
    A lower score is better, indicating the most important pixels were identified.
    """
    num_pixels = image.shape[1] * image.shape[2]
    # Get pixel indices sorted by importance (descending)
    important_pixels_indices = np.argsort(explanation_map.flatten())[::-1]

    original_prob = torch.sigmoid(model(image.unsqueeze(0).to(Config.DEVICE))[0]).max().item()

    scores = []
    # Create a baseline image (e.g., blurred)
    blurred_image = cv2.GaussianBlur(image.permute(1, 2, 0).numpy(), (21, 21), 0)
    blurred_image = torch.from_numpy(blurred_image.transpose(2, 0, 1))

    # Iteratively remove the most important pixels and get new probability
    for i in range(0, num_pixels, batch_size):
        perturbed_image = image.clone()
        # Get the indices for the current batch of pixels to remove
        indices_to_remove = important_pixels_indices[i:i + batch_size]
        # Convert flat indices to 2D coordinates
        rows = indices_to_remove // image.shape[2]
        cols = indices_to_remove % image.shape[2]
        # Replace pixels with the blurred version
        perturbed_image[:, rows, cols] = blurred_image[:, rows, cols]

        with torch.no_grad():
            prob = torch.sigmoid(model(perturbed_image.unsqueeze(0).to(Config.DEVICE))[0]).max().item()
        scores.append(prob)

    # The flip score is the area under the probability curve
    return np.trapz(scores) / len(scores)


def main():
    print("--- 🚀 Starting Full Evaluation Pipeline 🚀 ---")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 1. Load Model
    model = MultiTaskDeepfakeModel().to(Config.DEVICE)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.eval()
    print("✅ Model loaded successfully.")

    # 2. Load Test Data
    test_loader = get_dataloader('test', batch_size=Config.BATCH_SIZE)
    print(f"✅ Test data loaded: {len(test_loader.dataset)} samples.")

    # 3. Calculate Detection and Segmentation Metrics
    print("\n--- Computing Detection & Segmentation Metrics ---")
    metrics = defaultdict(list)
    for imgs, masks, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, masks, labels = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), labels.to(Config.DEVICE)
        with torch.no_grad():
            cls_logits, seg_logits = model(imgs)

        pred_classes = cls_logits.detach().argmax(axis=1).cpu().numpy()
        true_classes = labels.cpu().numpy()
        pred_masks = torch.sigmoid(seg_logits).detach().cpu().numpy() > 0.5
        true_masks = masks.cpu().numpy()

        metrics['f1'].append(f1(true_classes, pred_classes))
        metrics['accuracy'].append(accuracy(true_classes, pred_classes))
        metrics['precision'].append(precision(true_classes, pred_classes))
        metrics['recall'].append(recall(true_classes, pred_classes))
        metrics['dice'].append(dice_coef(true_masks, pred_masks))
        metrics['iou'].append(iou_score(true_masks, pred_masks))
        metrics['mae'].append(mae(true_masks, pred_masks))
        metrics['pixel_acc'].append(pixel_accuracy(true_masks, pred_masks))

    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print("\n--- 📊 Test Set Performance ---")
    log_metrics(avg_metrics, prefix="Final")

    # 4. Calculate XAI Metrics (on a subset for speed)
    if Config.XAI_ENABLED:
        print("\n--- Computing XAI Evaluation Metrics ---")
        grad_cam = GradCAM(model, Config.XAI_TARGET_LAYER)
        xai_metrics = defaultdict(list)

        # Take a subset of the test data for XAI evaluation
        xai_dataset = torch.utils.data.Subset(test_loader.dataset, range(Config.XAI_EVAL_SAMPLES))
        xai_loader = DataLoader(xai_dataset, batch_size=1)  # Process one by one

        for img, mask, label in tqdm(xai_loader, desc="Evaluating XAI"):
            img = img.to(Config.DEVICE)

            # Generate Grad-CAM explanation
            cam_map = grad_cam(img, class_idx=label.item())

            # Localization-Aware IoU: How well does explanation match ground truth mask?
            loc_iou = iou_score(mask.numpy(), (cam_map > 0.5))
            xai_metrics['loc_iou'].append(loc_iou)

            # Flip Score (Fidelity): How faithful is the explanation?
            flip = compute_flip_score(model, img.squeeze(0).cpu(), cam_map, batch_size=Config.XAI_FLIP_SCORE_BATCH_SIZE)
            xai_metrics['flip_score'].append(flip)

        avg_xai_metrics = {k: np.mean(v) for k, v in xai_metrics.items()}
        print("\n--- 📊 XAI Evaluation Performance ---")
        log_metrics(avg_xai_metrics, prefix="XAI")
        grad_cam.remove_hooks()

    print("\n--- ✅ Evaluation Complete ---")


if __name__ == '__main__':
    main()