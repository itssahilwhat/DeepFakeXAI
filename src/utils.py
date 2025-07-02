import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from config import Config
from torch.amp import autocast
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget


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


def test_step(model, batch, criterion):
    images = batch["image"].to(Config.DEVICE)
    masks = batch["mask"].to(Config.DEVICE)
    _, seg_logits = model(images)  # Get logits
    loss = criterion(seg_logits, masks)
    seg_output = torch.sigmoid(seg_logits)  # Apply sigmoid for metrics
    return loss.item(), seg_output, masks


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_loss", float("inf"))


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, threshold=0.5):
    outputs_bin = (outputs > threshold).float()
    labels_bin = (labels > threshold).float()
    intersection = (outputs_bin * labels_bin).sum((1, 2))
    union = (outputs_bin + labels_bin - outputs_bin * labels_bin).sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def dice_coefficient(outputs: torch.Tensor, labels: torch.Tensor, threshold=0.5):
    outputs = (outputs > threshold).float()
    labels = labels.float()
    intersection = (outputs * labels).sum()
    return (2 * intersection) / (outputs.sum() + labels.sum() + 1e-6)


def precision_recall_f1(outputs: torch.Tensor, labels: torch.Tensor, threshold=0.5):
    outputs = (outputs > threshold).float()
    labels = labels.float()
    tp = (outputs * labels).sum().item()
    fp = (outputs * (1 - labels)).sum().item()
    fn = ((1 - outputs) * labels).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


def validation_step(model, batch, criterion):
    images = batch["image"].to(Config.DEVICE)
    masks = batch["mask"].to(Config.DEVICE)
    _, seg_logits = model(images)  # Get logits
    loss = criterion(seg_logits, masks)
    seg_output = torch.sigmoid(seg_logits)  # Apply sigmoid for metrics
    return loss.item(), seg_output, masks


def save_mask_predictions(images, masks, predictions, out_dir, realistic_overlay=True):
    os.makedirs(out_dir, exist_ok=True)

    for i, (img, mask, pred) in enumerate(zip(images, masks, predictions)):
        # FIXED: Proper color conversion
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        mask_np = (mask.cpu().squeeze().numpy() * 255).astype(np.uint8)
        pred_np = pred.detach().cpu().squeeze().numpy()

        # Resize if needed
        if mask_np.shape != img_np.shape[:2]:
            mask_np = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))
        if pred_np.shape != img_np.shape[:2]:
            pred_np = cv2.resize(pred_np, (img_np.shape[1], img_np.shape[0]))

        norm_pred = 255 * (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-6)
        norm_pred = norm_pred.astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_pred, cv2.COLORMAP_JET)

        if realistic_overlay:
            threshold = 0.2
            alpha_mask = (pred_np > threshold).astype(np.float32)
            alpha_mask = cv2.GaussianBlur(alpha_mask, (11, 11), 0)
            alpha_mask = np.clip(alpha_mask, 0, 1)

            blended = img_np.astype(np.float32)
            for c in range(3):
                blended[:, :, c] = img_np[:, :, c] * (1 - alpha_mask) + heatmap[:, :, c] * alpha_mask
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            blended = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)

        mask_color = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        pred_color = cv2.cvtColor(norm_pred, cv2.COLOR_GRAY2BGR)
        side_by_side = cv2.hconcat([img_np, mask_color, pred_color, blended])

        fname = f"pred_{int(time.time() * 1000)}_{i}.png"
        cv2.imwrite(os.path.join(out_dir, fname), side_by_side)



# In src/utils.py, modify the generate_gradcam function
def generate_gradcam(model, input_tensor, target_layer, realistic_overlay=True):
    import cv2
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    cache = {}

    def forward_hook(module, input, output):
        cache[id(module)] = output.detach()

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

        # Get model output
        with autocast(device_type=Config.DEVICE):
            outputs = model(input_tensor)

        # Extract segmentation output from possible tuple
        seg_output = outputs[1] if isinstance(outputs, tuple) else outputs

        # Create target based on segmentation output (add channel dimension)
        mask = (seg_output[0].squeeze() > 0.5).cpu().numpy()
        targets = [SemanticSegmentationTarget(0, mask)]

        # Generate CAM map using the original model output before sigmoid
        logits_output = model(input_tensor)
        seg_logits = logits_output[1] if isinstance(logits_output, tuple) else logits_output
        cam_map = cam(input_tensor, targets)[0]

        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-6)
        cam_map_uint8 = np.uint8(cam_map * 255)
        heatmap = cv2.applyColorMap(cam_map_uint8, cv2.COLORMAP_JET)

        input_np = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        input_np = np.clip(input_np * 255.0, 0, 255).astype(np.uint8)

        if heatmap.shape[:2] != input_np.shape[:2]:
            heatmap = cv2.resize(heatmap, (input_np.shape[1], input_np.shape[0]))

        if realistic_overlay:
            threshold = 0.2
            alpha_mask = (cam_map > threshold).astype(np.float32)
            alpha_mask = cv2.GaussianBlur(alpha_mask, (11, 11), 0)
            alpha_mask = np.clip(alpha_mask, 0, 1)

            overlay = input_np.astype(np.float32)
            for c in range(3):
                overlay[:, :, c] = input_np[:, :, c] * (1 - alpha_mask) + heatmap[:, :, c] * alpha_mask
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        else:
            overlay = cv2.addWeighted(input_np, 0.6, heatmap, 0.4, 0)

        timestamp = int(time.time() * 1000)
        side_by_side = cv2.hconcat([input_np, heatmap, overlay])
        heatmap_path = os.path.join(Config.OUTPUT_DIR, f"cam_heatmap_{timestamp}.png")
        compare_path = os.path.join(Config.OUTPUT_DIR, f"cam_overlay_{timestamp}.png")
        cv2.imwrite(heatmap_path, heatmap)
        cv2.imwrite(compare_path, side_by_side)

    finally:
        handle.remove()

    return cam_map


def generate_pseudo_masks(model, images, labels, threshold=0.5):
    """Generate pseudo masks using model's attention"""
    model.eval()
    with torch.no_grad():
        _, pred_masks = model(images)
        # Create binary masks from predictions
        pseudo_masks = (pred_masks > threshold).float()
        # Only keep masks for fake images
        pseudo_masks[labels == 0] = 0
    return pseudo_masks


def evaluate_metrics(model, loader):
    model.eval()
    metrics = {
        "dice": [], "iou": [], "precision": [],
        "recall": [], "f1": [], "inference_time": []
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(Config.DEVICE)
            masks = batch["mask"].to(Config.DEVICE)
            labels = batch["label"].to(Config.DEVICE)

            start_time = time.time()
            cls_output, seg_output = model(images)
            inference_time = time.time() - start_time

            # Classification metrics
            if cls_output is not None:
                pred_labels = (torch.sigmoid(cls_output) > 0.5).float()
                precision, recall, f1 = precision_recall_f1(
                    pred_labels,
                    labels.float().view(-1, 1)
                )
                metrics["precision"].append(precision)
                metrics["recall"].append(recall)
                metrics["f1"].append(f1)

            # Segmentation metrics
            dice = dice_coefficient(seg_output, masks).item()
            iou = iou_pytorch(seg_output, masks).mean().item()
            metrics["dice"].append(dice)
            metrics["iou"].append(iou)
            metrics["inference_time"].append(inference_time)

    # Aggregate results
    results = {k: np.mean(v) for k, v in metrics.items()}
    results["samples_per_second"] = Config.BATCH_SIZE / np.mean(metrics["inference_time"])
    return results