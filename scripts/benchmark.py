# scripts/benchmark.py
import os
import json
import time
import numpy as np
from src.data import get_dataloader
from src.model import EfficientNetLiteTemporal
from src.utils import evaluate_metrics
from src.config import Config


def run_benchmark(checkpoint_path, dataset_name):
    model = EfficientNetLiteTemporal(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.to(Config.DEVICE)
    model.eval()

    test_loader = get_dataloader(dataset_name, "test", shuffle=False)
    results = evaluate_metrics(model, test_loader)

    # Save results
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    report_path = os.path.join(Config.LOG_DIR, "benchmark_results.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Benchmark results saved to {report_path}")
    print("=== Final Metrics ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    run_benchmark(args.checkpoint, args.dataset)