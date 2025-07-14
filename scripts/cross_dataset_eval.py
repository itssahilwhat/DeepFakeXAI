import os
import csv
import logging
import torch
from src.config import Config
from src.model import EfficientNetLiteTemporal
from src.data import get_dataloader
from src.utils import precision_recall_f1, dice_coefficient, iou_pytorch
from src.test_system import test_inference_speed
import numpy as np

def evaluate_cross_dataset(train_dataset, test_dataset, model_path=None):
    """Evaluate model trained on one dataset and tested on another"""
    logging.info(f"🔄 Cross-dataset evaluation: Train on {train_dataset}, Test on {test_dataset}")
    
    if model_path is None:
        model_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{train_dataset}.pth")
    
    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        return None
    
    # Load model
    model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_loader = get_dataloader(test_dataset, "test", batch_size=Config.BATCH_SIZE)
    
    # Evaluate
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_acc = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(Config.DEVICE)
            masks = batch["mask"].to(Config.DEVICE)
            
            cls_logits, seg_logits = model(images)
            seg_output = torch.sigmoid(seg_logits)
            
            # Calculate metrics
            dice = dice_coefficient(seg_output, masks).item()
            iou = iou_pytorch(seg_output, masks).mean().item()
            acc = (seg_output > 0.5).float().eq(masks).float().mean().item()
            
            total_dice += dice
            total_iou += iou
            total_acc += acc
            
            all_predictions.extend((seg_output > 0.5).cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
    
    # Calculate final metrics
    num_batches = len(test_loader)
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    avg_acc = total_acc / num_batches
    
    # Calculate precision, recall, F1
    precision, recall, f1 = precision_recall_f1(
        torch.tensor(all_predictions), 
        torch.tensor(all_targets)
    )
    
    results = {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'dice': avg_dice,
        'iou': avg_iou,
        'accuracy': avg_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    logging.info(f"Cross-dataset results ({train_dataset} → {test_dataset}):")
    logging.info(f"  Dice: {avg_dice:.4f}")
    logging.info(f"  IoU: {avg_iou:.4f}")
    logging.info(f"  Accuracy: {avg_acc:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1: {f1:.4f}")
    
    return results

def run_all_cross_dataset_evaluations():
    """Run cross-dataset evaluation for all dataset combinations"""
    datasets = ["celebahq", "ffhq", "dolos"]
    results = []
    
    for train_dataset in datasets:
        for test_dataset in datasets:
            if train_dataset != test_dataset:
                result = evaluate_cross_dataset(train_dataset, test_dataset)
                if result:
                    results.append(result)
    
    # Save results
    csv_path = os.path.join(Config.LOG_DIR, "cross_dataset_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    logging.info(f"✅ Cross-dataset results saved to {csv_path}")
    return results

if __name__ == "__main__":
    Config.setup_logging()
    run_all_cross_dataset_evaluations() 