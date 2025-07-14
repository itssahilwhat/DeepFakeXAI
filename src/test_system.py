import torch
import time
import numpy as np
from src.config import Config
from src.model import EfficientNetLiteTemporal
from src.data import get_dataloader
import logging

def test_inference_speed():
    """Test inference speed to achieve 100 it/sec target"""
    Config.setup_logging()
    logger = logging.getLogger("speed_test")

    # Load model
    model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
    model.eval()

    # Create dummy data
    batch_size = Config.BATCH_SIZE
    dummy_input = torch.randn(batch_size, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
    
    # Warmup
    logger.info("🔥 Warming up GPU...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Test inference speed
    logger.info("⚡ Testing inference speed...")
    num_iterations = 100
    times = []

        with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                current_speed = batch_size / np.mean(times[-20:])
                logger.info(f"Batch {i+1}/{num_iterations}: {current_speed:.1f} it/sec")

    # Calculate metrics
    avg_time = np.mean(times)
    std_time = np.std(times)
    speed = batch_size / avg_time
    target_speed = 100
    
    logger.info(f"📊 Speed Test Results:")
    logger.info(f"   Average time per batch: {avg_time:.4f}s ± {std_time:.4f}s")
    logger.info(f"   Current speed: {speed:.1f} it/sec")
    logger.info(f"   Target speed: {target_speed} it/sec")
    logger.info(f"   Speed ratio: {speed/target_speed:.2f}x")
    
    if speed >= target_speed:
        logger.info("✅ Target speed achieved!")
    else:
        logger.info("❌ Target speed not achieved. Consider optimizations:")
        logger.info("   - Reduce input size")
        logger.info("   - Increase batch size")
        logger.info("   - Use mixed precision")
        logger.info("   - Optimize model architecture")
    
    return speed

def test_training_speed():
    """Test training speed with real data"""
    Config.setup_logging()
    logger = logging.getLogger("training_speed_test")
    
    # Load model and data
    model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
    model.train()
    
    # Try to load a small dataset
    try:
        dataloader = get_dataloader("celebahq", "train", batch_size=Config.BATCH_SIZE)
        logger.info(f"📦 Loaded dataset with {len(dataloader)} batches")
    except Exception as e:
        logger.warning(f"Could not load dataset: {e}")
        logger.info("Using dummy data for training speed test")
        # Create dummy dataloader
        dummy_data = torch.randn(Config.BATCH_SIZE, 3, *Config.INPUT_SIZE)
        dummy_masks = torch.randn(Config.BATCH_SIZE, 1, *Config.INPUT_SIZE)
        dataloader = [(dummy_data, dummy_masks)] * 50
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Mixed precision setup
    from torch import autocast
    from torch.amp import GradScaler
    scaler = GradScaler("cuda") if Config.USE_AMP else None
    
    # Test training speed
    logger.info("🚀 Testing training speed...")
    num_iterations = 50
    times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_iterations:
            break
            
        if isinstance(batch, (list, tuple)):
            images, masks = batch
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
        else:
            images = batch["image"].to(Config.DEVICE)
            masks = batch["mask"].to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        start_time = time.time()
        
        if Config.USE_AMP:
            with autocast("cuda"):
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
        
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        
        if (i + 1) % 10 == 0:
            current_speed = Config.BATCH_SIZE / np.mean(times[-10:])
            logger.info(f"Batch {i+1}/{num_iterations}: {current_speed:.1f} it/sec")
    
    # Calculate metrics
    avg_time = np.mean(times)
    speed = Config.BATCH_SIZE / avg_time
    target_speed = 100
    
    logger.info(f"📊 Training Speed Test Results:")
    logger.info(f"   Average time per batch: {avg_time:.4f}s")
    logger.info(f"   Current speed: {speed:.1f} it/sec")
    logger.info(f"   Target speed: {target_speed} it/sec")
    logger.info(f"   Speed ratio: {speed/target_speed:.2f}x")
    
    return speed

def optimize_for_speed():
    """Optimize configuration for maximum speed"""
    logger = logging.getLogger("speed_optimization")
    
    logger.info("🔧 Optimizing for maximum speed...")
    
    # Test current speed
    current_inference_speed = test_inference_speed()
    current_training_speed = test_training_speed()
    
    # Optimization suggestions
    logger.info("💡 Speed Optimization Suggestions:")
    
    if current_inference_speed < 100:
        logger.info("   Inference optimizations:")
        logger.info("   - Reduce input size to 192x192 or 160x160")
        logger.info("   - Increase batch size if memory allows")
        logger.info("   - Use torch.jit.script for model compilation")
        logger.info("   - Enable torch.backends.cudnn.benchmark")
    
    if current_training_speed < 100:
        logger.info("   Training optimizations:")
        logger.info("   - Use mixed precision training (AMP)")
        logger.info("   - Reduce number of workers if CPU bound")
        logger.info("   - Use persistent workers in dataloader")
        logger.info("   - Reduce augmentation frequency")
    
    return current_inference_speed, current_training_speed

if __name__ == "__main__":
    # Run speed tests
    inference_speed = test_inference_speed()
    training_speed = test_training_speed()
    
    print(f"\n🎯 Final Results:")
    print(f"Inference Speed: {inference_speed:.1f} it/sec")
    print(f"Training Speed: {training_speed:.1f} it/sec")
    
    if inference_speed >= 100 and training_speed >= 100:
        print("✅ Both targets achieved!")
    else:
        print("❌ Some targets not achieved. Run optimize_for_speed() for suggestions.")