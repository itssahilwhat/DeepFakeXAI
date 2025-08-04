import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os
import json
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import img_size, images_out, dir_ckpt
from src.models.models import DetectionModel, SegmentationModel
from src.training.trainer import DetectionTrainer, SegmentationTrainer
from src.explainability.explain import explain_gradcam, explain_lime, overlay_numpy, evaluate_explainability
from src.data.data_utils import get_dataloader
import os

def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tfm(img)

def evaluate_test_set():
    """Run comprehensive evaluation on test set"""
    print("Evaluating on test set...")
    
    # Load models
    det = DetectionModel()
    seg = SegmentationModel()
    
    if os.path.exists(os.path.join(dir_ckpt, 'detect.pth')):
        det.load_state_dict(torch.load(os.path.join(dir_ckpt, 'detect.pth'), map_location='cpu'))
        det.eval()
        print("✓ Detection model loaded")
    else:
        print("✗ Detection model not found")
        det = None
        
    if os.path.exists(os.path.join(dir_ckpt, 'seg.pth')):
        seg.load_state_dict(torch.load(os.path.join(dir_ckpt, 'seg.pth'), map_location='cpu'))
        seg.eval()
        print("✓ Segmentation model loaded")
    else:
        print("✗ Segmentation model not found")
        seg = None
    
    # Get test loader
    test_loader = get_dataloader('test')
    
    # Evaluation metrics
    if det:
        det_correct, det_total = 0, 0
        all_det_preds = []
        all_det_labels = []
        all_det_probs = []
        
    if seg:
        seg_iou_scores = []
        seg_dice_scores = []
        seg_pixel_accuracies = []
    
    # Explainability metrics (on subset with masks)
    explainability_metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, (x, y, mask) in enumerate(test_loader):
            if det:
                out = det(x)
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(1)
                
                det_correct += (preds == y).sum().item()
                det_total += y.size(0)
                
                all_det_preds.extend(preds.cpu().numpy())
                all_det_labels.extend(y.cpu().numpy())
                all_det_probs.extend(probs[:, 1].cpu().numpy())
            
            if seg and mask.sum() > 0:
                out = seg(x)
                pred = (torch.sigmoid(out) > 0.5).float()
                
                # IoU
                inter = (pred * mask).sum().item()
                union = (pred + mask).clamp(0,1).sum().item()
                iou = inter / (union + 1e-8)
                seg_iou_scores.append(iou)
                
                # Dice
                dice = 2*inter / (pred.sum().item() + mask.sum().item() + 1e-8)
                seg_dice_scores.append(dice)
                
                # Pixel Accuracy
                correct_pixels = ((pred == mask) & (mask > 0)).sum().item()
                total_pixels = (mask > 0).sum().item()
                pixel_acc = correct_pixels / (total_pixels + 1e-8)
                seg_pixel_accuracies.append(pixel_acc)
            
            # Explainability evaluation (on first 50 samples with masks)
            if det and seg and mask.sum() > 0 and batch_idx < 50:
                for i in range(x.size(0)):
                    if mask[i].sum() > 0:  # Only evaluate samples with masks
                        img_tensor = x[i]
                        img_np = np.array(Image.open(f"data/wacv_data/images/{batch_idx}_{i}.png").resize(img_size))
                        target_class = int(y[i].item())
                        ground_truth_mask = mask[i][0].cpu().numpy()
                        
                        try:
                            metrics, gradcam, lime_mask = evaluate_explainability(
                                det, img_tensor, img_np, target_class, ground_truth_mask
                            )
                            
                            for key, value in metrics.items():
                                explainability_metrics[key].append(value)
                        except Exception as e:
                            print(f"Warning: Explainability evaluation failed for sample {batch_idx}_{i}: {e}")
                            continue
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SET EVALUATION")
    print("="*60)
    
    if det:
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        
        det_acc = det_correct / det_total if det_total > 0 else 0
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_det_labels, all_det_preds, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(all_det_labels, all_det_probs)
        except ValueError:
            auc = 0.5
        
        print(f"\n🔍 DETECTION METRICS:")
        print(f"  Accuracy:     {det_acc:.4f}")
        print(f"  Precision:    {precision:.4f}")
        print(f"  Recall:       {recall:.4f}")
        print(f"  F1-Score:     {f1:.4f}")
        print(f"  ROC-AUC:      {auc:.4f}")
    
    if seg:
        avg_iou = np.mean(seg_iou_scores) if seg_iou_scores else 0
        avg_dice = np.mean(seg_dice_scores) if seg_dice_scores else 0
        avg_pixel_acc = np.mean(seg_pixel_accuracies) if seg_pixel_accuracies else 0
        
        print(f"\n🎯 SEGMENTATION METRICS:")
        print(f"  IoU:          {avg_iou:.4f}")
        print(f"  Dice:         {avg_dice:.4f}")
        print(f"  Pixel Acc:    {avg_pixel_acc:.4f}")
        print(f"  Samples:      {len(seg_iou_scores)}")
    
    if explainability_metrics:
        print(f"\n🧠 EXPLAINABILITY METRICS:")
        for key, values in explainability_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {key}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save detailed results
    results = {
        'detection': {
            'accuracy': det_acc if det else 0,
            'precision': precision if det else 0,
            'recall': recall if det else 0,
            'f1': f1 if det else 0,
            'auc': auc if det else 0
        },
        'segmentation': {
            'iou': avg_iou if seg else 0,
            'dice': avg_dice if seg else 0,
            'pixel_accuracy': avg_pixel_acc if seg else 0,
            'num_samples': len(seg_iou_scores) if seg else 0
        },
        'explainability': {
            key: {'mean': np.mean(values), 'std': np.std(values)} 
            for key, values in explainability_metrics.items() if values
        }
    }
    
    with open(os.path.join(images_out, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {os.path.join(images_out, 'evaluation_results.json')}")

def main():
    parser = argparse.ArgumentParser(description='Explainable Deepfake Detection Pipeline')
    parser.add_argument('--mode', required=True, choices=['train_detect','train_seg','eval'])
    parser.add_argument('--image', type=str, help='Image path for evaluation')
    args = parser.parse_args()
    
    if args.mode == 'train_detect':
        print("Training Detection Model...")
        DetectionTrainer().fit()
        print("✓ Detection training complete")
        
    elif args.mode == 'train_seg':
        print("Training Segmentation Model...")
        SegmentationTrainer().fit()
        print("✓ Segmentation training complete")
        
    elif args.mode == 'eval':
        if args.image:
            # Single image evaluation
            print(f"Evaluating image: {args.image}")
            
            # Load models
            det = DetectionModel()
            seg = SegmentationModel()
            
            if os.path.exists(os.path.join(dir_ckpt, 'detect.pth')):
                det.load_state_dict(torch.load(os.path.join(dir_ckpt, 'detect.pth'), map_location='cpu'))
                det.eval()
            else:
                print("✗ Detection model not found")
                return
                
            if os.path.exists(os.path.join(dir_ckpt, 'seg.pth')):
                seg.load_state_dict(torch.load(os.path.join(dir_ckpt, 'seg.pth'), map_location='cpu'))
                seg.eval()
            else:
                print("✗ Segmentation model not found")
                return
            
            # Process image
            x = preprocess_image(args.image)
            logits = det(x.unsqueeze(0)).softmax(1).detach().cpu().numpy()[0]
            seg_mask = torch.sigmoid(seg(x.unsqueeze(0))).detach().cpu().numpy()[0]
            
            # Load original image for visualization
            img_np = np.array(Image.open(args.image).resize(img_size))
            
            # Generate explanations
            gradcam = explain_gradcam(det, x, int(logits.argmax()))
            
            # LIME prediction function
            def lime_predict_fn(images):
                # Convert LIME images back to tensor format
                batch = []
                for img in images:
                    # LIME provides images in 0-1 range, convert to tensor format
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                    # Normalize like ImageNet
                    img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    batch.append(img_tensor)
                batch_tensor = torch.stack(batch)
                with torch.no_grad():
                    return det(batch_tensor).softmax(1).cpu().numpy()
            
            lime_mask = explain_lime(lime_predict_fn, img_np)
            
            # Create visualization grid
            grid = np.concatenate([
                img_np,
                (gradcam*255).astype(np.uint8),
                (seg_mask[0]*255).astype(np.uint8),
                (overlay_numpy(img_np/255., lime_mask, 0.5)*255).astype(np.uint8)
            ], axis=1)
            
            out_path = os.path.join(images_out, 'eval_grid.png')
            Image.fromarray(grid).save(out_path)
            print(f"✓ Results saved to: {out_path}")
            print(f"Detection logits: Real={logits[0]:.4f}, Fake={logits[1]:.4f}")
            print(f"Prediction: {'FAKE' if logits.argmax() == 1 else 'REAL'}")
            
        else:
            # Test set evaluation
            evaluate_test_set()

if __name__ == '__main__':
    main()