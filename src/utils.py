# src/utils.py

import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from src.config import Config
from torch.amp import autocast
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from lime import lime_image
from skimage.segmentation import mark_boundaries


class MetricLogger:
    def __init__(self, metrics):
        self.metrics = metrics
        self.reset()

    def reset(self):
        self.values = {metric: 0 for metric in self.metrics}
        self.count = 0

    def update(self, batch_size, **kwargs):
        for metric, value in kwargs.items():
            if metric in self.values:
                self.values[metric] += value * batch_size
        self.count += batch_size

    def avg(self):
        return {metric: value / self.count for metric, value in self.values.items()}


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    dir_path = os.path.dirname(filename)
    if dir_path != "":
        os.makedirs(dir_path, exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint.get("model_state_dict")), strict=False)
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_loss", float("inf"))


def dice_coefficient(outputs, labels, threshold=0.5, eps=1e-6):
    outputs = (outputs > threshold).float()
    labels = labels.float()
    intersection = (outputs * labels).sum()
    return (2 * intersection + eps) / (outputs.sum() + labels.sum() + eps)

def iou_pytorch(outputs, labels, threshold=0.5, eps=1e-6):
    outputs = (outputs > threshold).float()
    labels = labels.float()
    intersection = (outputs * labels).sum((1, 2))
    union = (outputs + labels - outputs * labels).sum((1, 2))
    return (intersection + eps) / (union + eps)

def precision_recall_f1(outputs, labels, threshold=0.5, eps=1e-6):
    outputs = (outputs > threshold).float()
    labels = labels.float()
    tp = (outputs * labels).sum().item()
    fp = (outputs * (1 - labels)).sum().item()
    fn = ((1 - outputs) * labels).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1



def save_mask_predictions(images, masks, predictions, out_dir, realistic_overlay=True):
    os.makedirs(out_dir, exist_ok=True)
    for i, (img, mask, pred) in enumerate(zip(images, masks, predictions)):
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        img_np = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        mask_np = (mask.cpu().squeeze().numpy() * 255).astype(np.uint8)
        pred_np = pred.detach().cpu().squeeze().numpy()

        if mask_np.shape != img_np.shape[:2]:
            mask_np = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))
        if pred_np.shape != img_np.shape[:2]:
            pred_np = cv2.resize(pred_np, (img_np.shape[1], img_np.shape[0]))

        norm_pred = 255 * (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-6)
        heatmap   = cv2.applyColorMap(norm_pred.astype(np.uint8), cv2.COLORMAP_JET)

        if realistic_overlay:
            threshold = 0.2
            alpha_mask = (pred_np > threshold).astype(np.float32)
            alpha_mask = cv2.GaussianBlur(alpha_mask, (11, 11), 0)
            blended = img_np.astype(np.float32)
            for c in range(3):
                blended[:,:,c] = img_np[:,:,c] * (1 - alpha_mask) + heatmap[:,:,c] * alpha_mask
            blended = blended.astype(np.uint8)
        else:
            blended = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)

        mask_color = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        pred_color = cv2.cvtColor(norm_pred.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        side_by_side = cv2.hconcat([img_np, mask_color, pred_color, blended])

        fname = f"pred_{int(time.time() * 1000)}_{i}.png"
        cv2.imwrite(os.path.join(out_dir, fname), side_by_side)


def generate_gradcam(model, input_tensor, target_layer=None):
    """
    Returns a single H×W×3 numpy RGB heatmap for GradCAM++.
    """
    model.eval()
    if target_layer is None:
        # fallback to last conv layer
        target_layer = list(model.named_modules())[-2][1]
    cam = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer]
    )
    grayscale_cam = cam(input_tensor.cpu().numpy(), targets=[SemanticSegmentationTarget(0, None)])[0]
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def generate_lime_overlay(model, input_tensor):
    """
    Returns an H×W×3 uint8 LIME overlay image.
    """
    model.eval()
    img_np = (input_tensor[0].permute(1,2,0).numpy() * 255).astype(np.uint8)

    def batch_forward(images):
        tensors = torch.stack([
            torch.from_numpy(img.astype(np.float32)/255.).permute(2,0,1).to(Config.DEVICE)
            for img in images
        ])
        with torch.no_grad():
            _, seg_logits = model(tensors)
            probs = torch.sigmoid(seg_logits).cpu().numpy()
        return probs.transpose(0,2,3,1)  # (N, H, W, C)

    explainer = lime_image.LimeImageExplainer()
    exp = explainer.explain_instance(
        img_np,
        batch_forward,
        top_labels=1,
        hide_color=0,
        num_samples=100
    )
    temp, mask = exp.get_image_and_mask(
        label=exp.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    overlay = mark_boundaries(img_np, mask)
    return (overlay * 255).astype(np.uint8)