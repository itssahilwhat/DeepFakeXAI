import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms
from src.config import Config
from src.model import EfficientNetLiteTemporal
from src.utils import generate_gradcam, generate_lime_overlay
import logging
from src.lime_explainer import LIMEDeepfakeExplainer

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MetricsVisualizer:
    """Generate comprehensive metrics visualization for deepfake detection results"""
    
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or Config.LOG_DIR
        self.output_dir = os.path.join(Config.OUTPUT_DIR, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_training_metrics(self, csv_path=None):
        """Plot training and validation metrics over epochs"""
        if csv_path is None:
            csv_path = os.path.join(self.log_dir, "training_metrics.csv")
            
        if not os.path.exists(csv_path):
            logging.warning(f"Training metrics CSV not found: {csv_path}")
            return
            
        # Read CSV data
        epochs, train_loss, train_dice, train_iou, train_acc = [], [], [], [], []
        val_loss, val_dice, val_iou, val_acc = [], [], [], []
        precision, recall, f1 = [], [], []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row['Epoch']))
                    
                    # Handle different column formats
                    if 'TrainLoss' in row and 'ValLoss' in row:
                        # Full format with separate train/val columns
                        train_loss.append(float(row['TrainLoss']))
                        val_loss.append(float(row['ValLoss']))
                        
                        if 'TrainDice' in row:
                            train_dice.append(float(row['TrainDice']))
                            train_iou.append(float(row['TrainIoU']))
                            train_acc.append(float(row['TrainAccuracy']))
                            val_dice.append(float(row['ValDice']))
                            val_iou.append(float(row['ValIoU']))
                            val_acc.append(float(row['ValAccuracy']))
                        else:
                            # Use same values for train and val
                            dice_val = float(row['Dice']) if row['Dice'] != 'nan' else 0.0
                            train_dice.append(dice_val)
                            val_dice.append(dice_val)
                            
                            iou_val = float(row['IoU']) if row['IoU'] != 'nan' else 0.0
                            train_iou.append(iou_val)
                            val_iou.append(iou_val)
                            
                            acc_val = float(row['F1']) if row['F1'] != 'nan' else 0.0
                            train_acc.append(acc_val)
                            val_acc.append(acc_val)
                    else:
                        # Simple format with just one loss column
                        loss_val = float(row['TrainLoss']) if 'TrainLoss' in row else float(row['ValLoss'])
                        train_loss.append(loss_val)
                        val_loss.append(loss_val)
                        
                        dice_val = float(row['Dice']) if row['Dice'] != 'nan' else 0.0
                        train_dice.append(dice_val)
                        val_dice.append(dice_val)
                        
                        iou_val = float(row['IoU']) if row['IoU'] != 'nan' else 0.0
                        train_iou.append(iou_val)
                        val_iou.append(iou_val)
                        
                        acc_val = float(row['F1']) if row['F1'] != 'nan' else 0.0
                        train_acc.append(acc_val)
                        val_acc.append(acc_val)
                    
                    # Handle precision, recall, F1
                    if 'Precision' in row and row['Precision'] != 'nan':
                        precision.append(float(row['Precision']))
                    else:
                        precision.append(0.0)
                        
                    if 'Recall' in row and row['Recall'] != 'nan':
                        recall.append(float(row['Recall']))
                    else:
                        recall.append(0.0)
                        
                    if 'F1' in row and row['F1'] != 'nan':
                        f1.append(float(row['F1']))
                    else:
                        f1.append(0.0)
                        
                except (ValueError, KeyError) as e:
                    logging.warning(f"Skipping row due to error: {e}")
                    continue
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Deepfake Detection Training Metrics', fontsize=16, fontweight='bold')
        
        # Loss plot
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice Score plot
        axes[0, 1].plot(epochs, train_dice, 'b-', label='Train Dice', linewidth=2)
        axes[0, 1].plot(epochs, val_dice, 'r-', label='Val Dice', linewidth=2)
        axes[0, 1].set_title('Dice Score Over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU plot
        axes[0, 2].plot(epochs, train_iou, 'b-', label='Train IoU', linewidth=2)
        axes[0, 2].plot(epochs, val_iou, 'r-', label='Val IoU', linewidth=2)
        axes[0, 2].set_title('IoU Over Epochs')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('IoU')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1, 0].plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
        axes[1, 0].plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2)
        axes[1, 0].set_title('Accuracy Over Epochs')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision, Recall, F1 plot
        axes[1, 1].plot(epochs, precision, 'g-', label='Precision', linewidth=2)
        axes[1, 1].plot(epochs, recall, 'm-', label='Recall', linewidth=2)
        axes[1, 1].plot(epochs, f1, 'c-', label='F1-Score', linewidth=2)
        axes[1, 1].set_title('Precision, Recall, F1 Over Epochs')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Final metrics summary
        final_metrics = {
            'Final Val Loss': f"{val_loss[-1]:.4f}",
            'Final Val Dice': f"{val_dice[-1]:.4f}",
            'Final Val IoU': f"{val_iou[-1]:.4f}",
            'Final Val Acc': f"{val_acc[-1]:.4f}",
            'Final Precision': f"{precision[-1]:.4f}",
            'Final Recall': f"{recall[-1]:.4f}",
            'Final F1': f"{f1[-1]:.4f}"
        }
        
        axes[1, 2].axis('off')
        y_pos = 0.9
        for metric, value in final_metrics.items():
            axes[1, 2].text(0.1, y_pos, f"{metric}: {value}", 
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            y_pos -= 0.12
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✅ Training metrics plot saved to {self.output_dir}/training_metrics.png")
        
    def plot_test_results(self, csv_path=None):
        """Plot test results as a bar chart"""
        if csv_path is None:
            csv_path = os.path.join(self.log_dir, "test_results.csv")
            
        if not os.path.exists(csv_path):
            logging.warning(f"Test results CSV not found: {csv_path}")
            return
            
        # Read test results
        metrics, values = [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.append(row['Metric'])
                values.append(float(row['Value']))
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Test Set Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'test_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✅ Test results plot saved to {self.output_dir}/test_results.png")


class XAIVisualizer:
    """Generate XAI visualizations (GradCAM, LIME, CLIP) for deepfake detection"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(Config.CHECKPOINT_DIR, "best_combined.pth")
        self.output_dir = os.path.join(Config.OUTPUT_DIR, "xai_visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model if available
        self.model = None
        if os.path.exists(self.model_path):
            try:
                self.model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
                checkpoint = torch.load(self.model_path, map_location=Config.DEVICE)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                logging.info(f"✅ Loaded model from {self.model_path}")
            except Exception as e:
                logging.warning(f"Could not load model: {e}")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(Config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Setup LIME explainer
        if self.model is not None:
            self.lime_explainer = LIMEDeepfakeExplainer(self.model, self.transform)
        else:
            self.lime_explainer = None
        
    def create_xai_comparison(self, image_path, save_name=None):
        """Create side-by-side comparison of original, mask, GradCAM, LIME, and CLIP"""
        if self.model is None:
            logging.error("Model not loaded. Cannot generate XAI visualizations.")
            return
            
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            input_tensor = self.transform(image).unsqueeze(0).to(Config.DEVICE)
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return
        
        # Get model predictions
        with torch.no_grad():
            cls_output, seg_output = self.model(input_tensor)
            prediction = torch.sigmoid(seg_output).cpu().numpy()[0, 0]
        
        # Generate XAI visualizations
        try:
            # GradCAM
            gradcam_result = generate_gradcam(self.model, input_tensor)
            
            # LIME (simplified - you may need to implement this based on your lime_explainer)
            lime_result = self._generate_lime_overlay(original_image)
            
            # CLIP (placeholder - implement based on your CLIP explainer)
            clip_result = self._generate_clip_overlay(original_image)
            
        except Exception as e:
            logging.warning(f"Error generating XAI visualizations: {e}")
            # Create placeholder images
            gradcam_result = np.array(original_image)
            lime_result = np.array(original_image)
            clip_result = np.array(original_image)
        
        # Create comparison grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Deepfake Detection XAI Comparison', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Predicted mask
        axes[0, 1].imshow(prediction, cmap='hot', alpha=0.8)
        axes[0, 1].set_title(f'Predicted Mask\n(Confidence: {prediction.max():.3f})', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Overlay mask on original
        axes[0, 2].imshow(original_image)
        axes[0, 2].imshow(prediction, cmap='hot', alpha=0.6)
        axes[0, 2].set_title('Mask Overlay', fontweight='bold')
        axes[0, 2].axis('off')
        
        # GradCAM
        axes[1, 0].imshow(gradcam_result)
        axes[1, 0].set_title('GradCAM', fontweight='bold')
        axes[1, 0].axis('off')
        
        # LIME
        axes[1, 1].imshow(lime_result)
        axes[1, 1].set_title('LIME', fontweight='bold')
        axes[1, 1].axis('off')
        
        # CLIP
        axes[1, 2].imshow(clip_result)
        axes[1, 2].set_title('CLIP', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save with appropriate name
        if save_name is None:
            save_name = f"xai_comparison_{os.path.basename(image_path).split('.')[0]}.png"
        
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✅ XAI comparison saved to {self.output_dir}/{save_name}")
        
    def _generate_lime_overlay(self, image):
        """Generate LIME overlay using real LIMEDeepfakeExplainer"""
        if self.lime_explainer is not None:
            return np.array(self.lime_explainer.explain(image))
        else:
            return np.array(image)
    
    def _generate_clip_overlay(self, image):
        """Generate CLIP overlay (placeholder implementation)"""
        # This is a placeholder - implement based on your CLIP explainer
        try:
            # You can implement this based on your existing CLIP explainer
            return np.array(image)
        except:
            return np.array(image)


def generate_all_visualizations(model_path=None, image_paths=None):
    """Generate all visualizations: metrics plots and XAI comparisons"""
    logging.info("🎨 Generating comprehensive visualizations...")
    
    # Generate metrics plots
    metrics_viz = MetricsVisualizer()
    metrics_viz.plot_training_metrics()
    metrics_viz.plot_test_results()
    
    # Generate XAI visualizations if model and images are provided
    if model_path and image_paths:
        xai_viz = XAIVisualizer(model_path)
        for i, img_path in enumerate(image_paths):
            if os.path.exists(img_path):
                xai_viz.create_xai_comparison(img_path, f"xai_sample_{i+1}.png")
            else:
                logging.warning(f"Image not found: {img_path}")
    
    logging.info("✅ All visualizations generated successfully!")


if __name__ == "__main__":
    # Example usage
    Config.setup_logging()
    
    # Generate metrics plots only
    generate_all_visualizations()
    
    # Generate XAI visualizations (uncomment if you have model and test images)
    # model_path = "checkpoints/best_combined.pth"
    # test_images = ["path/to/test1.jpg", "path/to/test2.jpg"]
    # generate_all_visualizations(model_path, test_images) 