import os
import torch
import numpy as np
from src.config import Config
from src.model import MultiTaskDeepfakeModel
from src.dataset import get_dataloader
from src.utils import accuracy, dice_coef, iou_score, f1

RESULTS_CSV = os.path.join(Config.LOG_DIR, 'cross_dataset_results.csv')
BATCH_SIZE = 16
NUM_BATCHES = 10
DATASETS = [Config.CELEBAHQ_PATH, Config.FFHQ_PATH]

os.makedirs(Config.LOG_DIR, exist_ok=True)

with open(RESULTS_CSV, 'w') as f:
    f.write('TrainDataset,TestDataset,Acc,Dice,IoU,F1\n')
    for train_ds in DATASETS:
        for test_ds in DATASETS:
            if train_ds == test_ds:
                continue
            print(f'=== Train={os.path.basename(train_ds)}, Test={os.path.basename(test_ds)} ===')
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
            loader = get_dataloader(test_ds, 'valid', batch_size=BATCH_SIZE, img_size=Config.IMG_SIZE, aug_strong=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
            accs, dices, ious, f1s = [], [], [], []
            for i, (imgs, masks, labels) in enumerate(loader):
                if i >= NUM_BATCHES:
                    break
                imgs, masks, labels = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), labels.to(Config.DEVICE)
                with torch.no_grad():
                    cls_logits, seg_logits = model(imgs)
                    preds = torch.sigmoid(cls_logits).cpu().numpy() > 0.5
                    acc = accuracy(labels.cpu().numpy(), preds.squeeze().astype(int))
                    dice = dice_coef(masks.cpu().numpy(), torch.sigmoid(seg_logits).cpu().numpy() > 0.5)
                    iou = iou_score(masks.cpu().numpy(), torch.sigmoid(seg_logits).cpu().numpy() > 0.5)
                    f1val = f1(labels.cpu().numpy(), preds.squeeze().astype(int))
                    accs.append(acc)
                    dices.append(dice)
                    ious.append(iou)
                    f1s.append(f1val)
            f.write(f'{os.path.basename(train_ds)},{os.path.basename(test_ds)},{np.mean(accs):.4f},{np.mean(dices):.4f},{np.mean(ious):.4f},{np.mean(f1s):.4f}\n')
            print(f'Acc={np.mean(accs):.4f}, Dice={np.mean(dices):.4f}, IoU={np.mean(ious):.4f}, F1={np.mean(f1s):.4f}') 