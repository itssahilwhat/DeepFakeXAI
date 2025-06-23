import torch
from utils import dice_coefficient, iou_pytorch, save_checkpoint, load_checkpoint
from model import EfficientNetLiteTemporal
from config import Config


def test_metrics():
    pred = torch.rand(4, 1, 224, 224)
    gt = (torch.rand(4, 1, 224, 224) > 0.5).float()

    dice = dice_coefficient(pred, gt)
    iou = iou_pytorch(pred, gt).mean()

    print(f"Dice: {dice:.4f}")
    print(f"IoU:  {iou:.4f}")
    assert 0 <= dice <= 1
    assert 0 <= iou <= 1


def test_checkpoint():
    model = EfficientNetLiteTemporal()
    dummy_optimizer = torch.optim.Adam(model.parameters())

    save_checkpoint({
        "epoch": 3,
        "state_dict": model.state_dict(),
        "optimizer": dummy_optimizer.state_dict(),
        "best_loss": 0.1234
    }, filename="temp_test.pth")

    epoch, best_loss = load_checkpoint("temp_test.pth", model, dummy_optimizer)
    print("Checkpoint loaded:", epoch, best_loss)
    assert epoch == 3
    assert abs(best_loss - 0.1234) < 1e-4


if __name__ == "__main__":
    test_metrics()
    test_checkpoint()