#!/usr/bin/env python3
"""
Speed Optimization Script for Deepfake Detection
Target: 100+ it/sec training speed with 95%+ metrics
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.data import get_dataloader
from src.model import EfficientNetLiteTemporal
from src.train import set_seeds
from src.utils import dice_coefficient, iou_pytorch
from src.train import accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_training_speed():
    """Benchmark training speed with different optimizations"""
    
    logger.info("🚀 Starting Speed Optimization Benchmark")
    
    # Test different batch sizes
    batch_sizes = [32, 48, 64, 96]
    speeds = {}
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        # Temporarily modify config
        original_batch_size = Config.BATCH_SIZE
        Config.BATCH_SIZE = batch_size
        
        try:
            # Get dataloaders
            train_loader = get_dataloader("celebahq", "train", batch_size=batch_size)
            val_loader = get_dataloader("celebahq", "valid", batch_size=batch_size)
            test_loader = get_dataloader("celebahq", "test", batch_size=batch_size)
            
            # Create model
            model = EfficientNetLiteTemporal(
                num_classes=Config.NUM_CLASSES,
                pretrained=Config.PRETRAINED
            ).to(Config.DEVICE)
            
            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
            
            # Loss
            criterion = torch.nn.BCEWithLogitsLoss()
            
            # Mixed precision
            scaler = torch.amp.GradScaler() if Config.USE_AMP else None
            
            # Benchmark
            model.train()
            start_time = time.time()
            num_batches = 10
            
            for i, batch in enumerate(train_loader):
                if i >= num_batches:
                    break
                    
                images = batch["image"].to(Config.DEVICE, non_blocking=True)
                masks = batch["mask"].to(Config.DEVICE, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                if Config.USE_AMP:
                    with torch.amp.autocast(device_type="cuda"):
                        cls_logits, seg_logits = model(images)
                        loss = criterion(seg_logits, masks)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    cls_logits, seg_logits = model(images)
                    loss = criterion(seg_logits, masks)
                    loss.backward()
                    optimizer.step()
            
            end_time = time.time()
            total_time = end_time - start_time
            speed = (num_batches * batch_size) / total_time
            
            speeds[batch_size] = speed
            logger.info(f"  Batch size {batch_size}: {speed:.1f} it/sec")
            
        except Exception as e:
            logger.error(f"  Failed with batch size {batch_size}: {e}")
            speeds[batch_size] = 0
        
        finally:
            # Restore original batch size
            Config.BATCH_SIZE = original_batch_size
    
    # Find optimal batch size
    optimal_batch_size = max(speeds, key=speeds.get)
    optimal_speed = speeds[optimal_batch_size]
    
    logger.info(f"✅ Optimal batch size: {optimal_batch_size} ({optimal_speed:.1f} it/sec)")
    
    return optimal_batch_size, optimal_speed

def optimize_memory_settings():
    """Optimize memory settings for speed"""
    
    logger.info("🔧 Optimizing Memory Settings")
    
    # Test different worker counts
    worker_counts = [4, 8, 12, 16]
    speeds = {}
    
    for num_workers in worker_counts:
        logger.info(f"Testing {num_workers} workers")
        
        original_workers = Config.NUM_WORKERS
        Config.NUM_WORKERS = num_workers
        
        try:
            # Get dataloaders
            train_loader = get_dataloader("celebahq", "train", batch_size=Config.BATCH_SIZE)
            
            # Simple speed test
            start_time = time.time()
            num_batches = 5
            
            for i, batch in enumerate(train_loader):
                if i >= num_batches:
                    break
                _ = batch["image"].to(Config.DEVICE, non_blocking=True)
            
            end_time = time.time()
            speed = (num_batches * Config.BATCH_SIZE) / (end_time - start_time)
            speeds[num_workers] = speed
            
            logger.info(f"  {num_workers} workers: {speed:.1f} it/sec")
            
        except Exception as e:
            logger.error(f"  Failed with {num_workers} workers: {e}")
            speeds[num_workers] = 0
        
        finally:
            Config.NUM_WORKERS = original_workers
    
    optimal_workers = max(speeds, key=speeds.get)
    logger.info(f"✅ Optimal workers: {optimal_workers}")
    
    return optimal_workers

def generate_optimization_report():
    """Generate optimization report"""
    
    logger.info("📊 Generating Optimization Report")
    
    # Test current settings
    optimal_batch_size, optimal_speed = benchmark_training_speed()
    optimal_workers = optimize_memory_settings()
    
    # Generate recommendations
    recommendations = []
    
    if optimal_speed < 100:
        recommendations.append(f"⚠️ Current speed ({optimal_speed:.1f} it/sec) below target (100 it/sec)")
        recommendations.append(f"💡 Increase batch size to {optimal_batch_size}")
        recommendations.append(f"💡 Use {optimal_workers} workers")
        recommendations.append("💡 Enable mixed precision training")
        recommendations.append("💡 Reduce dataset size for faster iteration")
    else:
        recommendations.append(f"✅ Speed target achieved: {optimal_speed:.1f} it/sec")
    
    # Print recommendations
    logger.info("📋 Optimization Recommendations:")
    for rec in recommendations:
        logger.info(f"  {rec}")
    
    return {
        "optimal_batch_size": optimal_batch_size,
        "optimal_speed": optimal_speed,
        "optimal_workers": optimal_workers,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    set_seeds(42)
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        sys.exit(1)
    
    logger.info(f"🖥️ Using GPU: {torch.cuda.get_device_name()}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run optimization
    report = generate_optimization_report()
    
    logger.info("🎉 Speed optimization completed!") 