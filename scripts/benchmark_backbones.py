import os
import time
import torch
import numpy as np
from src.config import Config
from src.model import MultiTaskDeepfakeModel
from src.dataset import get_dataloader
from src.utils import accuracy, dice_coef, iou_score, f1
import argparse

BACKBONES = ['mobilenet_v3_small', 'efficientnet_b0']
RESULTS_CSV = os.path.join(Config.LOG_DIR, 'backbone_benchmark_results.csv')
BATCH_SIZE = 16
NUM_BATCHES = 10

os.makedirs(Config.LOG_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained model checkpoint')
    args = parser.parse_args()
    with open(RESULTS_CSV, 'w') as f:
        f.write('Backbone,Params,Throughput,VRAM,Acc,Dice,IoU,F1\n')
        for backbone in BACKBONES:
            print(f'\n=== Benchmarking {backbone} ===')
            model = MultiTaskDeepfakeModel(
                backbone_name=backbone,
                num_classes=Config.NUM_CLASSES,
                pretrained=False,
                dropout=Config.DROPOUT,
                segmentation=Config.SEGMENTATION,
                attention=Config.ATTENTION
            ).to(Config.DEVICE)
            if args.checkpoint and os.path.exists(args.checkpoint):
                print(f'Loading checkpoint: {args.checkpoint}')
                model.load_state_dict(torch.load(args.checkpoint, map_location=Config.DEVICE))
            model.eval()
            params = sum(p.numel() for p in model.parameters())
            loader = get_dataloader('valid', BATCH_SIZE, Config.IMG_SIZE, False, Config.NUM_WORKERS, Config.PIN_MEMORY)
            start = time.time()
            torch.cuda.reset_peak_memory_stats() if Config.DEVICE == 'cuda' else None
            accs, dices, ious, f1s = [], [], [], []
            for i, (imgs, masks, labels) in enumerate(loader):
                if i >= NUM_BATCHES:
                    break
                imgs, masks, labels = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), labels.to(Config.DEVICE)
                with torch.no_grad():
                    cls_logits, seg_logits = model(imgs)
                    preds = torch.sigmoid(cls_logits).cpu().numpy()
                    if Config.NUM_CLASSES == 2:
                        pred_classes = preds.argmax(axis=1)
                        true_classes = labels.cpu().numpy().argmax(axis=1) if labels.ndim > 1 else labels.cpu().numpy()
                        acc = accuracy(true_classes, pred_classes)
                        f1val = f1(true_classes, pred_classes)
                    else:
                        pred_classes = preds.argmax(axis=1)
                        acc = accuracy(labels.cpu().numpy(), pred_classes)
                        f1val = f1(labels.cpu().numpy(), pred_classes)
                    pred_mask = torch.sigmoid(seg_logits).cpu().numpy() > 0.5
                    if i == 0:
                        print(f"[DEBUG] mask sum: {masks.cpu().numpy().sum()}, pred_mask sum: {pred_mask.sum()}")
                    dice = dice_coef(masks.cpu().numpy(), pred_mask)
                    iou = iou_score(masks.cpu().numpy(), pred_mask)
                    accs.append(acc)
                    dices.append(dice)
                    ious.append(iou)
                    f1s.append(f1val)
            end = time.time()
            throughput = (i+1) / (end - start)
            vram = torch.cuda.max_memory_allocated() / 1024**2 if Config.DEVICE == 'cuda' else 0
            f.write(f'{backbone},{params},{throughput:.2f},{vram:.1f},{np.mean(accs):.4f},{np.mean(dices):.4f},{np.mean(ious):.4f},{np.mean(f1s):.4f}\n')
            print(f'Backbone: {backbone}, Params: {params}, Throughput: {throughput:.2f} it/sec, VRAM: {vram:.1f} MB')
            print(f'Acc: {np.mean(accs):.4f}, Dice: {np.mean(dices):.4f}, IoU: {np.mean(ious):.4f}, F1: {np.mean(f1s):.4f}')

if __name__ == "__main__":
    main() 