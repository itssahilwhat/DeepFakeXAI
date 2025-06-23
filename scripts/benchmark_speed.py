import time
import torch
import argparse
from torchvision import transforms
from PIL import Image

from src.config import Config
from src.model import EfficientNetLiteTemporal


def benchmark(batch_sizes=[1, 2, 4, 8, 16]):
    model = EfficientNetLiteTemporal(num_classes=1, pretrained=False).to(Config.DEVICE)
    checkpoint = torch.load(f"{Config.CHECKPOINT_DIR}/traced_model.pth", map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dummy_img = torch.zeros((1, 3, Config.INPUT_SIZE[0], Config.INPUT_SIZE[1])).to(Config.DEVICE)
    times = {}
    for bs in batch_sizes:
        inp = dummy_img.repeat(bs, 1, 1, 1)
        start = time.time()
        with torch.no_grad():
            # Run 5 Monte Carlo dropout iterations
            preds = [model(inp, mc_dropout=True) for _ in range(5)]
            pred = torch.mean(torch.stack(preds), dim=0)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times[bs] = elapsed
        print(f"Batch size {bs}: {elapsed:.4f}s")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    with open(f"{Config.OUTPUT_DIR}/speed_benchmark.json", "w") as f:
        json.dump(times, f, indent=2)
    print("Speed benchmarking complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    args = parser.parse_args()
    benchmark()