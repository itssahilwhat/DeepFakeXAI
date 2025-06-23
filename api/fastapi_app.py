import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import uvicorn

from model import EfficientNetLiteTemporal
from config import Config
from utils import save_mask_predictions
from text_explainer import ArtifactTextExplainer

app = FastAPI()
model = EfficientNetLiteTemporal(num_classes=1, pretrained=False)
checkpoint = torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"), map_location=Config.DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.to(Config.DEVICE)
model.eval()

explainer = ArtifactTextExplainer(model_path=None)

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    transform = transforms.Compose([
        transforms.Resize(Config.INPUT_SIZE),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        prob = model(input_tensor).cpu().item()
    is_fake = prob > 0.5
    explanation = "No artifact"
    if is_fake:
        explanation = "Potential artifacts detected"
    # Save ONNX or something if requested
    return JSONResponse({"probability": prob, "is_fake": bool(is_fake), "explanation": explanation})

@app.post("/export_onnx")
def export_onnx():
    model_cpu = EfficientNetLiteTemporal(num_classes=1, pretrained=False).cpu()
    dummy_input = torch.randn(1, 3, *Config.INPUT_SIZE)
    onnx_path = os.path.join(Config.ONNX_DIR, "model.onnx")
    os.makedirs(Config.ONNX_DIR, exist_ok=True)
    torch.onnx.export(model_cpu, dummy_input, onnx_path, input_names=["input"], output_names=["output"])
    return {"onnx_path": onnx_path}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
