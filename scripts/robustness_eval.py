import os
import torch
import numpy as np
import cv2
from src.config import Config
from src.model import MultiTaskDeepfakeModel
from src.dataset import get_dataloader
from src.utils import accuracy, dice_coef, iou_score, f1

RESULTS_CSV = os.path.join(Config.LOG_DIR, 'robustness_results.csv')
BATCH_SIZE = 16
NUM_BATCHES = 10

JPEG_QUALITIES = Config.JPEG_QUALITIES
NOISE_LEVELS = Config.NOISE_LEVELS
BLUR_LEVELS = Config.BLUR_LEVELS

def perturb_image(img, jpeg_quality=None, noise_level=None, blur_level=None):
    img_np = (img.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    if jpeg_quality is not None:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode('.jpg', img_np, encode_param)
        img_np = cv2.imdecode(encimg, 1)
    if noise_level is not None and noise_level > 0:
        noise = np.random.normal(0, noise_level, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
    if blur_level is not None and blur_level > 0:
        img_np = cv2.GaussianBlur(img_np, (blur_level|1, blur_level|1), 0)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    img_t = torch.from_numpy(img_np.transpose(2,0,1)).float() / 255.0
    return img_t

def main():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    model = MultiTaskDeepfakeModel(
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        dropout=Config.DROPOUT,
        segmentation=Config.SEGMENTATION,
        attention=Config.ATTENTION
    )
    model.load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'), map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    loader = get_dataloader(Config.CELEBAHQ_PATH, 'valid', batch_size=BATCH_SIZE, img_size=Config.IMG_SIZE, aug_strong=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    with open(RESULTS_CSV, 'w') as f:
        f.write('Perturbation,Level,Acc,Dice,IoU,F1\n')
        for pert_type, levels in [('jpeg', JPEG_QUALITIES), ('noise', NOISE_LEVELS), ('blur', BLUR_LEVELS)]:
            for level in levels:
                accs, dices, ious, f1s = [], [], [], []
                for i, (imgs, masks, labels) in enumerate(loader):
                    if i >= NUM_BATCHES:
                        break
                    perturbed = torch.stack([
                        perturb_image(img, jpeg_quality=level if pert_type=='jpeg' else None,
                                           noise_level=level if pert_type=='noise' else None,
                                           blur_level=level if pert_type=='blur' else None)
                        for img in imgs
                    ]).to(Config.DEVICE)
                    with torch.no_grad():
                        cls_logits, seg_logits = model(perturbed)
                        preds = torch.sigmoid(cls_logits).cpu().numpy() > 0.5
                        acc = accuracy(labels.cpu().numpy(), preds.squeeze().astype(int))
                        dice = dice_coef(masks.cpu().numpy(), torch.sigmoid(seg_logits).cpu().numpy() > 0.5)
                        iou = iou_score(masks.cpu().numpy(), torch.sigmoid(seg_logits).cpu().numpy() > 0.5)
                        f1val = f1(labels.cpu().numpy(), preds.squeeze().astype(int))
                        accs.append(acc)
                        dices.append(dice)
                        ious.append(iou)
                        f1s.append(f1val)
                f.write(f'{pert_type},{level},{np.mean(accs):.4f},{np.mean(dices):.4f},{np.mean(ious):.4f},{np.mean(f1s):.4f}\n')
                print(f'{pert_type}={level}: Acc={np.mean(accs):.4f}, Dice={np.mean(dices):.4f}, IoU={np.mean(ious):.4f}, F1={np.mean(f1s):.4f}')

if __name__ == "__main__":
    main() 