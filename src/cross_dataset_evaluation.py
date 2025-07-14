import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from src.config import Config
from src.utils import dice_coefficient, iou_pytorch, precision_recall_f1
from src.data import get_dataloader


class CrossDatasetEvaluator:
    """Cross-dataset evaluation for generalization testing"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = {}
    
    def evaluate_on_dataset(self, dataset_name, subset='test'):
        """Evaluate model on a specific dataset"""
        try:
            # Load dataset
            dataloader = get_dataloader(dataset_name, subset, shuffle=False)
            
            # Evaluation metrics
            total_loss = 0.0
            total_dice = 0.0
            total_iou = 0.0
            total_accuracy = 0.0
            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            num_samples = 0
            
            self.model.eval()
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Evaluating on {dataset_name}"):
                    # Move to device
                    images = batch["image"].to(self.device, non_blocking=True)
                    masks = batch["mask"].to(self.device, non_blocking=True)
                    
                    # Forward pass
                    predictions = self.model(images)
                    if isinstance(predictions, tuple):
                        _, seg_logits = predictions
                    else:
                        seg_logits = predictions
                    
                    seg_output = torch.sigmoid(seg_logits)
                    
                    # Compute loss
                    loss = F.binary_cross_entropy_with_logits(seg_logits, masks)
                    
                    # Compute metrics
                    dice = dice_coefficient(seg_output, masks).item()
                    iou = iou_pytorch(seg_output, masks).mean().item()
                    pred_binary = (seg_output > 0.5).float()
                    precision, recall, f1 = precision_recall_f1(pred_binary, masks)
                    accuracy = (pred_binary == masks).float().mean().item()
                    
                    # Accumulate
                    batch_size = images.size(0)
                    total_loss += loss.item() * batch_size
                    total_dice += dice * batch_size
                    total_iou += iou * batch_size
                    total_accuracy += accuracy * batch_size
                    total_precision += precision * batch_size
                    total_recall += recall * batch_size
                    total_f1 += f1 * batch_size
                    num_samples += batch_size
            
            # Compute averages
            avg_metrics = {
                'loss': total_loss / num_samples,
                'dice': total_dice / num_samples,
                'iou': total_iou / num_samples,
                'accuracy': total_accuracy / num_samples,
                'precision': total_precision / num_samples,
                'recall': total_recall / num_samples,
                'f1': total_f1 / num_samples,
                'num_samples': num_samples
            }
            
            return avg_metrics
            
        except Exception as e:
            print(f"Error evaluating on {dataset_name}: {e}")
            return None
    
    def evaluate_all_datasets(self, dataset_names):
        """Evaluate model on multiple datasets"""
        results = {}
        
        for dataset_name in dataset_names:
            print(f"\n🔍 Evaluating on {dataset_name}...")
            metrics = self.evaluate_on_dataset(dataset_name)
            
            if metrics is not None:
                results[dataset_name] = metrics
                print(f"✅ {dataset_name} Results:")
                for metric, value in metrics.items():
                    if metric != 'num_samples':
                        print(f"   {metric}: {value:.4f}")
            else:
                print(f"❌ Failed to evaluate on {dataset_name}")
        
        self.results = results
        return results
    
    def compute_generalization_score(self):
        """Compute overall generalization score"""
        if not self.results:
            return 0.0
        
        # Compute average performance across datasets
        avg_metrics = {}
        for metric in ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']:
            values = [result[metric] for result in self.results.values() if metric in result]
            if values:
                avg_metrics[metric] = np.mean(values)
        
        # Compute generalization score (weighted average)
        weights = {
            'dice': 0.3,
            'iou': 0.2,
            'accuracy': 0.2,
            'precision': 0.1,
            'recall': 0.1,
            'f1': 0.1
        }
        
        generalization_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in avg_metrics:
                generalization_score += avg_metrics[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            generalization_score /= total_weight
        
        return generalization_score
    
    def generate_cross_dataset_report(self, save_path=None):
        """Generate comprehensive cross-dataset evaluation report"""
        if not self.results:
            print("No results available. Run evaluate_all_datasets first.")
            return
        
        # Compute generalization score
        gen_score = self.compute_generalization_score()
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append("CROSS-DATASET EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Overall Generalization Score: {gen_score:.4f}")
        report.append("")
        
        # Individual dataset results
        for dataset_name, metrics in self.results.items():
            report.append(f"📊 {dataset_name.upper()}")
            report.append("-" * 40)
            for metric, value in metrics.items():
                if metric != 'num_samples':
                    report.append(f"{metric:12}: {value:.4f}")
            report.append(f"num_samples: {metrics['num_samples']}")
            report.append("")
        
        # Summary statistics
        report.append("📈 SUMMARY STATISTICS")
        report.append("-" * 40)
        
        for metric in ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']:
            values = [result[metric] for result in self.results.values() if metric in result]
            if values:
                report.append(f"{metric:12}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        # Print report
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to {save_path}")
        
        return report_text
    
    def plot_cross_dataset_results(self, save_path=None):
        """Plot cross-dataset results"""
        if not self.results:
            print("No results available. Run evaluate_all_datasets first.")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data for plotting
            datasets = list(self.results.keys())
            metrics = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                values = [self.results[dataset][metric] for dataset in datasets if metric in self.results[dataset]]
                
                if values:
                    bars = axes[i].bar(datasets, values)
                    axes[i].set_title(f'{metric.upper()} Score')
                    axes[i].set_ylabel('Score')
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("matplotlib or seaborn not available for plotting")
        except Exception as e:
            print(f"Error creating plot: {e}")
    
    def compare_with_baselines(self, baseline_results):
        """Compare results with baseline methods"""
        if not self.results:
            print("No results available. Run evaluate_all_datasets first.")
            return
        
        comparison = {}
        
        for dataset_name in self.results.keys():
            if dataset_name in baseline_results:
                comparison[dataset_name] = {}
                for metric in ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']:
                    if metric in self.results[dataset_name] and metric in baseline_results[dataset_name]:
                        our_score = self.results[dataset_name][metric]
                        baseline_score = baseline_results[dataset_name][metric]
                        improvement = our_score - baseline_score
                        comparison[dataset_name][metric] = {
                            'our_score': our_score,
                            'baseline_score': baseline_score,
                            'improvement': improvement,
                            'improvement_pct': (improvement / baseline_score * 100) if baseline_score > 0 else 0
                        }
        
        return comparison 