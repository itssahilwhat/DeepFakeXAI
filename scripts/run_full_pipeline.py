#!/usr/bin/env python3
"""
Complete Deepfake Detection Pipeline
Runs training, evaluation, XAI, ONNX export, and benchmarking
"""

import os
import sys
import logging
import subprocess
from src.config import Config

def run_command(cmd, description):
    """Run a command and log the result"""
    logging.info(f"🚀 {description}")
    logging.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ {description} failed: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False

def main():
    """Run the complete pipeline"""
    Config.setup_logging()
    logging.info("🎯 Starting Complete Deepfake Detection Pipeline")
    
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Train the model
    logging.info("="*60)
    logging.info("STEP 1: TRAINING")
    logging.info("="*60)
    
    train_success = run_command(
        "PYTHONPATH=. python src/train.py --datasets celebahq,ffhq",
        "Training model on celebahq and ffhq datasets"
    )
    
    if not train_success:
        logging.error("Training failed. Stopping pipeline.")
        return False
    
    # Step 2: Generate visualizations
    logging.info("="*60)
    logging.info("STEP 2: VISUALIZATIONS")
    logging.info("="*60)
    
    viz_success = run_command(
        "PYTHONPATH=. python src/visualization.py",
        "Generating metrics plots and XAI visualizations"
    )
    
    # Step 3: Export to ONNX
    logging.info("="*60)
    logging.info("STEP 3: ONNX EXPORT")
    logging.info("="*60)
    
    onnx_success = run_command(
        "PYTHONPATH=. python scripts/export_onnx.py",
        "Exporting model to ONNX format"
    )
    
    # Step 4: Benchmark ONNX
    if onnx_success:
        logging.info("="*60)
        logging.info("STEP 4: ONNX BENCHMARKING")
        logging.info("="*60)
        
        benchmark_success = run_command(
            "PYTHONPATH=. python scripts/onnx_benchmark.py",
            "Benchmarking ONNX model performance"
        )
    
    # Step 5: Cross-dataset evaluation
    logging.info("="*60)
    logging.info("STEP 5: CROSS-DATASET EVALUATION")
    logging.info("="*60)
    
    cross_eval_success = run_command(
        "PYTHONPATH=. python scripts/cross_dataset_eval.py",
        "Running cross-dataset evaluation"
    )
    
    # Step 6: Ablation study
    logging.info("="*60)
    logging.info("STEP 6: ABLATION STUDY")
    logging.info("="*60)
    
    ablation_success = run_command(
        "PYTHONPATH=. python scripts/ablation_study.py",
        "Running ablation study"
    )
    
    # Step 7: Final speed test
    logging.info("="*60)
    logging.info("STEP 7: FINAL SPEED TEST")
    logging.info("="*60)
    
    speed_success = run_command(
        "PYTHONPATH=. python src/test_system.py",
        "Final speed and performance validation"
    )
    
    # Summary
    logging.info("="*60)
    logging.info("PIPELINE COMPLETION SUMMARY")
    logging.info("="*60)
    
    results = {
        "Training": train_success,
        "Visualizations": viz_success,
        "ONNX Export": onnx_success,
        "ONNX Benchmark": benchmark_success if onnx_success else False,
        "Cross-Dataset Eval": cross_eval_success,
        "Ablation Study": ablation_success,
        "Speed Test": speed_success
    }
    
    for step, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logging.info(f"{step}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    logging.info(f"\nOverall: {success_count}/{total_count} steps completed successfully")
    
    if success_count == total_count:
        logging.info("🎉 COMPLETE PIPELINE SUCCESS!")
        logging.info("Your deepfake detection system is ready for production and research!")
    else:
        logging.warning("⚠️ Some steps failed. Check logs for details.")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 