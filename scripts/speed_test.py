import torch
import time
from src.config import Config
from src.model import MultiTaskDeepfakeModel

BATCH_SIZES = [1, 4, 8, 16, 32]
N_ITERS = 100

def main():
    model = MultiTaskDeepfakeModel(
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        dropout=Config.DROPOUT,
        segmentation=Config.SEGMENTATION,
        attention=Config.ATTENTION
    ).to(Config.DEVICE)
    model.eval()
    for batch_size in BATCH_SIZES:
        dummy_input = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
        torch.cuda.reset_peak_memory_stats() if Config.DEVICE == 'cuda' else None
        start = time.time()
        for _ in range(N_ITERS):
            with torch.no_grad():
                _ = model(dummy_input)
        end = time.time()
        throughput = N_ITERS / (end - start)
        vram = torch.cuda.max_memory_allocated() / 1024**2 if Config.DEVICE == 'cuda' else 0
        print(f'Batch size {batch_size}: Throughput: {throughput:.2f} it/sec, VRAM: {vram:.1f} MB')

if __name__ == "__main__":
    main() 