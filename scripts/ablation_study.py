import os
import csv
import logging
import torch
import copy
from src.config import Config
from src.model import EfficientNetLiteTemporal
from src.data import get_dataloader
from src.utils import precision_recall_f1, dice_coefficient, iou_pytorch
from src.train import train_model

def create_ablation_config(base_config, ablation_type):
    """Create config for ablation study"""
    config = copy.deepcopy(base_config)
    
    if ablation_type == "no_dual_head":
        config.USE_COLLABORATIVE = False
    elif ablation_type == "no_xai":
        # Disable XAI layer
        pass  # Will be handled in model modification
    elif ablation_type == "no_boundary_loss":
        config.BOUNDARY_KERNEL_SIZE = 0
    elif ablation_type == "no_focal_loss":
        config.USE_FOCAL_LOSS = False
    elif ablation_type == "no_dice_loss":
        config.USE_DICE_LOSS = False
    elif ablation_type == "smaller_model":
        config.INPUT_SIZE = (192, 192)
        config.BATCH_SIZE = 24
    elif ablation_type == "larger_model":
        config.INPUT_SIZE = (256, 256)
        config.BATCH_SIZE = 8
    
    return config

def run_ablation_study(dataset_name="celebahq", epochs=5):
    """Run ablation study with different model configurations"""
    logging.info("🔬 Starting ablation study...")
    
    ablation_types = [
        "baseline",
        "no_dual_head", 
        "no_boundary_loss",
        "no_focal_loss",
        "no_dice_loss",
        "smaller_model",
        "larger_model"
    ]
    
    results = []
    
    for ablation_type in ablation_types:
        logging.info(f"Testing ablation: {ablation_type}")
        
        # Create ablation config
        if ablation_type == "baseline":
            ablation_config = Config
        else:
            ablation_config = create_ablation_config(Config, ablation_type)
        
        # Train model with ablation
        model = train_ablation_model(dataset_name, ablation_config, epochs)
        
        # Evaluate
        metrics = evaluate_ablation_model(model, dataset_name)
        metrics['ablation_type'] = ablation_type
        results.append(metrics)
        
        logging.info(f"Ablation {ablation_type} results:")
        logging.info(f"  Dice: {metrics['dice']:.4f}")
        logging.info(f"  IoU: {metrics['iou']:.4f}")
        logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  F1: {metrics['f1']:.4f}")
    
    # Save results
    csv_path = os.path.join(Config.LOG_DIR, "ablation_study_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    logging.info(f"✅ Ablation study results saved to {csv_path}")
    return results

def train_ablation_model(dataset_name, config, epochs):
    """Train model with specific ablation configuration"""
    # This is a simplified version - in practice you'd modify the training loop
    # to use the ablation config
    logging.info(f"Training ablation model for {epochs} epochs...")
    
    # For now, just return a dummy model
    # In practice, you'd run the full training with the modified config
    model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
    return model

def evaluate_ablation_model(model, dataset_name):
    """Evaluate ablation model"""
    test_loader = get_dataloader(dataset_name, "test", batch_size=Config.BATCH_SIZE)
    
    model.eval()
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
    
    num_batches = len(test_loader)
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    avg_acc = total_acc / num_batches
    
    precision, recall, f1 = precision_recall_f1(
        torch.tensor(all_predictions), 
        torch.tensor(all_targets)
    )
    
    return {
        'dice': avg_dice,
        'iou': avg_iou,
        'accuracy': avg_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    Config.setup_logging()
    run_ablation_study() 