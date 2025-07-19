import os
import torch
import numpy as np
import cv2
from src.config import Config
from src.model import MultiTaskDeepfakeModel
from src.dataset import get_dataloader
from torch.utils.data import DataLoader

WILD_DIR = os.path.join(Config.DATA_ROOT, 'wild_images')
CHECKPOINT = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
BATCH_SIZE = 8
OUTPUT_DIR = os.path.join(Config.OUTPUT_DIR, 'wild_predictions')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    model = MultiTaskDeepfakeModel(
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        dropout=Config.DROPOUT,
        segmentation=Config.SEGMENTATION,
        attention=Config.ATTENTION
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    dataset = get_dataloader(WILD_DIR, 'wild', batch_size=BATCH_SIZE, img_size=Config.IMG_SIZE, aug_strong=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    with torch.no_grad():
        for i, (imgs, masks, labels) in enumerate(dataset):
            imgs = imgs.to(Config.DEVICE)
            cls_logits, seg_logits = model(imgs)
            preds_cls = torch.sigmoid(cls_logits).cpu().numpy() > 0.5
            preds_seg = torch.sigmoid(seg_logits).cpu().numpy()
            for j in range(imgs.size(0)):
                idx = i * BATCH_SIZE + j
                img_np = imgs[j].cpu().numpy().transpose(1,2,0)
                img_np = (img_np * 255).astype(np.uint8)
                pred_cls = preds_cls[j]
                pred_seg = (preds_seg[j,0] > 0.5).astype(np.uint8) * 255
                out_img = np.concatenate([
                    img_np,
                    np.stack([pred_seg]*3, axis=-1)
                ], axis=1)
                out_path = os.path.join(OUTPUT_DIR, f'wild_pred_{idx}_cls{pred_cls}.png')
                cv2.imwrite(out_path, out_img)
    print(f"Saved wild predictions to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 