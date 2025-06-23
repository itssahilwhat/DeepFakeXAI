import os
import time
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from config import Config
from torch.amp import autocast
from pytorch_grad_cam import GradCAMPlusPlus


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    dir_path = os.path.dirname(filename)
    if dir_path != "":
        os.makedirs(dir_path, exist_ok=True)
    torch.save(state, filename)


def test_step(model, batch, criterion):
    images = batch["image"].to(Config.DEVICE)
    masks = batch["mask"].to(Config.DEVICE)
    outputs = model(images)
    loss = criterion(outputs, masks)
    return loss.item(), outputs, masks


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
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


def training_step(model, batch, criterion, optimizer, scaler=None):
    images = batch["image"].to(Config.DEVICE)
    masks = batch["mask"].to(Config.DEVICE)
    optimizer.zero_grad()
    with autocast(device_type=Config.DEVICE):
        outputs = model(images)
        loss = criterion(outputs, masks)
    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss.item()


def validation_step(model, batch, criterion):
    images = batch["image"].to(Config.DEVICE)
    masks = batch["mask"].to(Config.DEVICE)
    outputs = model(images)
    loss = criterion(outputs, masks)
    return loss.item(), outputs, masks


def save_mask_predictions(images, masks, predictions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for img, mask, pred in zip(images, masks, predictions):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        mask_np = mask.cpu().squeeze().numpy()
        pred_np = pred.cpu().squeeze().numpy()
        overlay = (pred_np > 0.5).astype(np.uint8) * 255
        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        pil_mask = Image.fromarray(overlay)
        fname = f"{int(time.time()*1000)}.png"
        save_path = os.path.join(out_dir, fname)
        pil_img.paste(pil_mask, (0, 0), pil_mask)
        pil_img.save(save_path)


def generate_gradcam(model, input_tensor, target_layer):
    # Cache for feature maps
    cache = {}
    def forward_hook(module, input, output):
        cache[id(module)] = output.detach()

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
        cam_map = cam(input_tensor)[0]
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-6)
    finally:
        handle.remove()
    return cam_map


def region_query(model, image, region):
    cropped = image[:, :, region[1]:region[3], region[0]:region[2]]
    with torch.no_grad():
        pred = model(cropped)
    return pred


def confidence_masking(pred, threshold=0.5):
    mask = (pred > threshold).float()
    return mask