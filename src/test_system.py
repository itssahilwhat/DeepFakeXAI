# src/test_system.py
import os
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from config import Config
from model import EfficientNetLiteTemporal
from utils import save_mask_predictions
from text_explainer import ArtifactTextExplainer


def test_system(folder, dataset):
    # Load model
    model = EfficientNetLiteTemporal(num_classes=1, pretrained=False)
    checkpoint = torch.load(os.path.join(Config.CHECKPOINT_DIR, f"best_{dataset}.pth"),
                            map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(Config.DEVICE)
    model.eval()

    explainer = ArtifactTextExplainer()
    transform = transforms.Compose([
        transforms.Resize(Config.INPUT_SIZE),
        transforms.ToTensor()
    ])

    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(folder, fname)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        input_img = Image.fromarray(image)
        input_tensor = transform(input_img).unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            cls_output, seg_output = model(input_tensor)
            pred = seg_output[0, 0].cpu().numpy()

        # Generate explanation
        artifacts = ["blur", "warp", "distortion"]
        explanation = explainer.explain(artifacts, input_img)
        print(f"Image: {fname}, Explanation: {explanation}")

        # Save results
        save_mask_predictions([input_tensor[0].cpu()], [torch.tensor(pred)], [torch.tensor(pred)],
                              out_dir=os.path.join(Config.OUTPUT_DIR, "explanations"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end test")
    parser.add_argument("--folder", type=str, required=True, help="Path to sample images")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    args = parser.parse_args()
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    test_system(args.folder, args.dataset)