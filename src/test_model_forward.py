import torch
from model import EfficientNetLiteTemporal


def test_model_forward():
    model = EfficientNetLiteTemporal()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print("Model output shape:", output.shape)
    assert output.shape == (2, 1, 224, 224)


if __name__ == "__main__":
    test_model_forward()