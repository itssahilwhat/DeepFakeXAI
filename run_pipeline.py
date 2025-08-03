#!/usr/bin/env python3

"""
Main execution script for the multi-supervision deepfake detection methodology.
This script orchestrates the entire pipeline from manifest generation to final evaluation.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def execute_command(command, description):
    """Execute a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"{description} completed successfully!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    else:
        print(f"{description} failed!")
        print(f"Error: {result.stderr}")
        return False
    
    return True

def verify_prerequisites():
    """Verify if all prerequisites are met"""
    print("Checking prerequisites...")
    
    # Check if data directory exists
    data_dir = Path("data/wacv_data")
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please ensure your dataset is organized in the expected structure:")
        print("data/wacv_data/")
        print("├── celebahq/")
        print("│   ├── real/")
        print("│   └── fake/")
        print("└── ffhq/")
        print("    ├── real/")
        print("    └── fake/")
        return False
    
    # Check if required packages are installed
    try:
        import torch
        import timm
        import wandb
        import cv2
        import pandas
        print("All required packages are installed")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main execution pipeline"""
    print("Deepfake Detection - Multi-Supervision Methodology")
    print("=" * 60)
    
    # Check prerequisites
    if not verify_prerequisites():
        print("Prerequisites not met. Exiting.")
        return
    
    # Step 1: Generate manifest
    print("\nSTEP 1: Generate Dataset Manifest")
    if not execute_command(
        "python scripts/preprocessing/build_manifest.py",
        "Generating dataset manifest with supervision types"
    ):
        return
    
    # Step 2: Train GradCAM Classifier
    print("\nSTEP 2: Train GradCAM Classifier")
    if not execute_command(
        "python scripts/train.py --model gradcam",
        "Training GradCAM classifier (Supervision A)"
    ):
        print("GradCAM training failed, but continuing...")
    
    # Step 3: Train Patch-Forensics Model
    print("\nSTEP 3: Train Patch-Forensics Model")
    if not execute_command(
        "python scripts/train.py --model patch",
        "Training Patch-Forensics model (Supervision A+B)"
    ):
        print("Patch-Forensics training failed, but continuing...")
    
    # Step 4: Train MobileNetV3 Model
    print("\nSTEP 4: Train MobileNetV3 Model")
    if not execute_command(
        "python scripts/train.py --model mobilenetv3",
        "Training MobileNetV3 model (Mobile-optimized)"
    ):
        print("MobileNetV3 training failed, but continuing...")
    
    # Step 5: Analyze Dataset
    print("\nSTEP 5: Analyze Dataset")
    if not execute_command(
        "python scripts/analyze.py",
        "Analyzing dataset composition and balance"
    ):
        print("Dataset analysis failed, but continuing...")
    
    # Step 6: Comprehensive Evaluation
    print("\nSTEP 6: Comprehensive Evaluation")
    if not execute_command(
        "python scripts/evaluation/evaluate_models.py",
        "Running comprehensive evaluation of all models"
    ):
        print("Evaluation failed, but check outputs directory for partial results")
    
    # Step 7: Export Models for Mobile
    print("\nSTEP 7: Export Models for Mobile")
    if not execute_command(
        "python scripts/export.py",
        "Exporting models for mobile deployment"
    ):
        print("Mobile export failed, but continuing...")
    
    # Step 8: Generate Report
    print("\nSTEP 8: Generate Final Report")
    generate_report()
    
    print("\nPipeline completed!")
    print("Check the following directories for results:")
    print("- outputs/: Evaluation results and visualizations")
    print("- logs/training/checkpoints/: Trained models")
    print("- data/wacv_data/manifest.csv: Dataset manifest")

def generate_report():
    """Generate a summary report"""
    print("Generating summary report...")
    
    report_content = """
# Deepfake Detection - Multi-Supervision Methodology Report

## Overview
This report summarizes the results of implementing the new multi-supervision approach for deepfake detection.

## Methodology
The new approach addresses the core issue of mask-mismatch by implementing three supervision types:

- **Supervision A**: Full supervision with high-quality masks
- **Supervision B**: Weak supervision with noisy/partial masks  
- **Supervision C**: No mask supervision (global fakes)

## Model Variants

### 1. GradCAM Classifier
- **Purpose**: Quick explainability and detection
- **Supervision**: Type A only
- **Output**: Classification + GradCAM heatmaps

### 2. Patch-Forensics Model
- **Purpose**: Balanced detection and localization
- **Supervision**: Types A + B
- **Output**: Classification + 37x37 patch masks

### 3. Attention Model
- **Purpose**: Smooth masks with robust training
- **Supervision**: Types A + B + C
- **Output**: Classification + learnable attention masks

## Key Improvements

1. **Mask Quality Analysis**: Automatic detection of supervision type based on mask coverage
2. **Multi-Supervision Training**: Different loss weights for different supervision types
3. **Comprehensive XAI Metrics**: IoU, Pixel Accuracy, Average Precision, Pixel AUC, Correlation
4. **Robust Evaluation**: Cross-validation across all model variants

## Expected Results

Based on the methodology, you should see:
- Improved training stability (no more mask-mismatch failures)
- Better localization accuracy for partial/noisy masks
- Comprehensive explainability metrics
- Robust performance across different fake generation methods

## Next Steps

1. Review the evaluation results in `outputs/comprehensive_evaluation_results.csv`
2. Examine visualizations in the `outputs/` directory
3. Compare performance across the three model variants
4. Consider model ensemble approaches for production deployment

## Files Modified/Created

- `scripts/preprocessing/generate_manifest.py`: New manifest generation with supervision types
- `src/preprocessing/dataset.py`: Updated to handle multi-supervision
- `src/detection/model.py`: Added three new model variants
- `src/xai/explainability.py`: Enhanced XAI metrics and visualization
- `train_gradcam.py`: GradCAM classifier training
- `train_patch.py`: Patch-Forensics training
- `scripts/evaluation/comprehensive_validation.py`: Comprehensive evaluation
- `requirements.txt`: Updated dependencies

## Troubleshooting

If you encounter issues:

1. **Manifest Generation**: Check dataset structure matches expected format
2. **Training Failures**: Verify GPU memory and batch size settings
3. **Evaluation Errors**: Ensure trained models exist in checkpoint directory
4. **Visualization Issues**: Check matplotlib and opencv installations

For detailed logs, check the wandb dashboard or training logs.
"""
    
    # Save report
    report_path = "outputs/methodology_report.md"
    os.makedirs("outputs", exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main() 