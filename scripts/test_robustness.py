import os
import argparse
import json
import numpy as np
from PIL import Image
from torchvision import transforms

from src.config import Config
from src.model import EfficientNetLiteTemporal

def evaluate_robustness(dataset, subset, noise_levels, qty_levels):
    data_root = os.path.join(Config.DATA_ROOT, dataset)
    results = []
    transform = transforms.Compose([transforms.Resize(Config.INPUT_SIZE), transforms.ToTensor()])

    model = EfficientNetLiteTemporal(num_classes=1, pretrained=False)
    checkpoint = torch.load(os.path.join(Config.CHECKPOINT_DIR, f"best_{dataset}.pth"), map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(Config.DEVICE)
    model.eval()

    for nl in noise_levels:
        for qty in qty_levels:
            errors = []
            # Simulate robustness test by corrupting images
            for root, _, files in os.walk(os.path.join(data_root, "real", subset)):
                for fname in files:
                    if not fname.endswith(".jpg"): continue
                    img = Image.open(os.path.join(root, fname)).convert("RGB")
                    # Add Gaussian noise
                    img_np = np.array(img).astype(np.float32)
                    noise = np.random.normal(0, nl, img_np.shape).astype(np.float32)
                    img_noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
                    inp = transform(Image.fromarray(img_noisy)).unsqueeze(0).to(Config.DEVICE)
                    with torch.no_grad():
                        out = model(inp)
                    # Just record mean output as measure (placeholder)
                    errors.append(float(out.mean().cpu()))
                    if len(errors) >= qty:
                        break
                if len(errors) >= qty:
                    break
            avg_error = float(np.mean(errors))
            results.append({"noise": nl, "qty": qty, "score": avg_error})
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    fname = os.path.join(Config.OUTPUT_DIR, f"robustness_{dataset}_{subset}.json")
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved robustness results to {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test robustness")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str, default="test")
    parser.add_argument("--noise_levels", nargs="+", type=float, default=[0,5,10])
    parser.add_argument("--qty_levels", nargs="+", type=int, default=[100, 80, 60])
    args = parser.parse_args()
    evaluate_robustness(args.dataset, args.subset, args.noise_levels, args.qty_levels)
