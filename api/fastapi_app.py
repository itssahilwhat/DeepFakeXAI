import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.config import Config
from src.model import MultiTaskDeepfakeModel

app = FastAPI()

# Model loading with GPU/CPU fallback
def load_model():
    model = MultiTaskDeepfakeModel(
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        dropout=Config.DROPOUT,
        segmentation=Config.SEGMENTATION,
        attention=Config.ATTENTION
    )
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return image

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    images = [read_imagefile(await file.read()) for file in files]
    input_tensors = torch.stack([transform(img) for img in images]).to(Config.DEVICE)
    with torch.no_grad():
        cls_logits, seg_logits = model(input_tensors)
        probs = torch.sigmoid(cls_logits).cpu().numpy().tolist()
        is_fake = [float(p[1] if len(p) > 1 else p[0]) > 0.5 for p in probs]
    return JSONResponse({"probabilities": probs, "is_fake": is_fake})

@app.post("/export_onnx")
def export_onnx():
    dummy_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE).to("cpu")
    onnx_path = os.path.join(Config.ONNX_DIR, "model.onnx")
    os.makedirs(Config.ONNX_DIR, exist_ok=True)
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["class_output", "segmentation_output"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "class_output": {0: "batch_size"}, "segmentation_output": {0: "batch_size"}}
    )
    model.to(Config.DEVICE)
    return {"onnx_path": onnx_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 