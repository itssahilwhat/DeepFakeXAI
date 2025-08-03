#!/usr/bin/env python3

"""
Complete setup and run script for the deepfake detection project.
This script handles dataset preparation, balance verification, and training execution.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    else:
        print(f"❌ {description} failed!")
        print(f"Error: {result.stderr}")
        if check:
            return False
        else:
            print("Continuing anyway...")
            return True

def verify_dataset_structure():
    """Verify dataset structure and balance"""
    print("\n🔍 Verifying Dataset Structure and Balance")
    print("="*50)
    
    # Check if manifest exists
    manifest_path = Path('data/wacv_data/manifest.csv')
    if not manifest_path.exists():
        print("❌ Dataset manifest not found!")
        print("Please run the manifest generation first:")
        print("python scripts/preprocessing/build_manifest.py")
        return False
    
    # Run dataset analysis
    print("📊 Analyzing dataset composition and balance...")
    if not run_command("python scripts/analyze.py", "Dataset Analysis"):
        return False
    
    return True

def check_balance_requirements():
    """Check if dataset meets balance requirements"""
    print("\n⚖️ Checking Balance Requirements")
    print("="*50)
    
    import pandas as pd
    
    manifest_path = Path('data/wacv_data/manifest.csv')
    df = pd.read_csv(manifest_path)
    
    # Check overall balance
    total_real = len(df[df['label'] == 0])
    total_fake = len(df[df['label'] == 1])
    total = len(df)
    
    print(f"📈 Overall Dataset Statistics:")
    print(f"  Total samples: {total:,}")
    print(f"  Real samples: {total_real:,} ({total_real/total*100:.1f}%)")
    print(f"  Fake samples: {total_fake:,} ({total_fake/total*100:.1f}%)")
    
    # Check split distribution
    print(f"\n📂 Split Distribution:")
    for split in ['train', 'valid', 'test']:
        split_df = df[df['split'] == split]
        split_real = len(split_df[split_df['label'] == 0])
        split_fake = len(split_df[split_df['label'] == 1])
        split_total = len(split_df)
        
        print(f"  {split.upper()}: {split_total:,} samples ({split_total/total*100:.1f}%)")
        print(f"    Real: {split_real:,} ({split_real/split_total*100:.1f}%)")
        print(f"    Fake: {split_fake:,} ({split_fake/split_total*100:.1f}%)")
        
        # Check if split is balanced (40-60% range)
        real_ratio = split_real / split_total
        fake_ratio = split_fake / split_total
        balanced = 0.4 <= real_ratio <= 0.6 and 0.4 <= fake_ratio <= 0.6
        
        print(f"    Balance: {'✅' if balanced else '❌'}")
        
        if not balanced:
            print(f"    ⚠️  {split.upper()} split is not balanced!")
    
    # Check split ratios (80-10-10)
    train_count = len(df[df['split'] == 'train'])
    valid_count = len(df[df['split'] == 'valid'])
    test_count = len(df[df['split'] == 'test'])
    
    train_ratio = train_count / total
    valid_ratio = valid_count / total
    test_ratio = test_count / total
    
    print(f"\n🎯 Split Ratios:")
    print(f"  Train: {train_ratio*100:.1f}% (target: 80%) {'✅' if 0.75 <= train_ratio <= 0.85 else '❌'}")
    print(f"  Valid: {valid_ratio*100:.1f}% (target: 10%) {'✅' if 0.08 <= valid_ratio <= 0.12 else '❌'}")
    print(f"  Test:  {test_ratio*100:.1f}% (target: 10%) {'✅' if 0.08 <= test_ratio <= 0.12 else '❌'}")
    
    # Overall assessment
    balanced_splits = all([
        0.4 <= len(df[df['split'] == split & df['label'] == 0]) / len(df[df['split'] == split]) <= 0.6
        for split in ['train', 'valid', 'test']
    ])
    
    correct_ratios = all([
        0.75 <= train_ratio <= 0.85,
        0.08 <= valid_ratio <= 0.12,
        0.08 <= test_ratio <= 0.12
    ])
    
    if balanced_splits and correct_ratios:
        print(f"\n🎉 Dataset meets all requirements!")
        print(f"  ✅ Balanced splits (40-60% real/fake)")
        print(f"  ✅ Correct ratios (80-10-10 train/valid/test)")
        return True
    else:
        print(f"\n⚠️  Dataset needs adjustment:")
        if not balanced_splits:
            print(f"  ❌ Some splits are not balanced")
        if not correct_ratios:
            print(f"  ❌ Split ratios are not 80-10-10")
        return False

def setup_environment():
    """Setup the environment and dependencies"""
    print("\n🔧 Setting Up Environment")
    print("="*50)
    
    # Check if requirements are installed
    print("📦 Checking dependencies...")
    try:
        import torch
        import timm
        import wandb
        import cv2
        import pandas
        import albumentations
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Installing requirements...")
        if not run_command("pip install -r requirements.txt", "Installing Requirements"):
            return False
    
    # Create necessary directories
    print("📁 Creating directories...")
    directories = [
        'logs/training/checkpoints',
        'outputs',
        'outputs/dataset_analysis',
        'outputs/evaluation_visualizations',
        'outputs/mobile_models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    return True

def run_training_pipeline():
    """Run the complete training pipeline"""
    print("\n🚀 Running Training Pipeline")
    print("="*50)
    
    models = ['gradcam', 'patch', 'mobilenetv3']
    
    for model in models:
        print(f"\n🎯 Training {model.upper()} model...")
        if not run_command(f"python scripts/train.py --model {model}", f"Training {model} model", check=False):
            print(f"⚠️  {model} training failed, but continuing...")
    
    return True

def run_evaluation():
    """Run evaluation and analysis"""
    print("\n📊 Running Evaluation and Analysis")
    print("="*50)
    
    # Run evaluation
    if not run_command("python scripts/evaluation/evaluate_models.py", "Model Evaluation", check=False):
        print("⚠️  Evaluation failed, but continuing...")
    
    # Run mobile export
    if not run_command("python scripts/export.py", "Mobile Export", check=False):
        print("⚠️  Mobile export failed, but continuing...")
    
    return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Setup and run deepfake detection project')
    parser.add_argument('--skip-setup', action='store_true', help='Skip environment setup')
    parser.add_argument('--skip-balance-check', action='store_true', help='Skip balance verification')
    parser.add_argument('--train-only', action='store_true', help='Run only training (skip setup and evaluation)')
    parser.add_argument('--models', nargs='+', choices=['gradcam', 'patch', 'mobilenetv3'], 
                       default=['gradcam', 'patch', 'mobilenetv3'], help='Models to train')
    
    args = parser.parse_args()
    
    print("🎯 Deepfake Detection Project - Complete Setup and Run")
    print("="*60)
    
    # Step 1: Setup environment
    if not args.skip_setup:
        if not setup_environment():
            print("❌ Environment setup failed. Exiting.")
            return
    
    # Step 2: Verify dataset structure and balance
    if not args.skip_balance_check:
        if not verify_dataset_structure():
            print("❌ Dataset verification failed. Exiting.")
            return
        
        if not check_balance_requirements():
            print("❌ Dataset balance requirements not met. Please fix your dataset.")
            return
    
    # Step 3: Run training pipeline
    if args.train_only:
        print("\n🎯 Running Training Only")
        for model in args.models:
            print(f"\n🎯 Training {model.upper()} model...")
            run_command(f"python scripts/train.py --model {model}", f"Training {model} model", check=False)
    else:
        # Run complete pipeline
        if not run_training_pipeline():
            print("❌ Training pipeline failed. Exiting.")
            return
        
        # Step 4: Run evaluation
        if not run_evaluation():
            print("❌ Evaluation failed. Exiting.")
            return
    
    print("\n🎉 Project execution completed!")
    print("\n📁 Check the following directories for results:")
    print("  - logs/training/checkpoints/: Trained models and checkpoints")
    print("  - outputs/: Evaluation results and visualizations")
    print("  - outputs/mobile_models/: Exported mobile models")
    print("  - data/wacv_data/manifest.csv: Dataset manifest")

if __name__ == "__main__":
    main() 