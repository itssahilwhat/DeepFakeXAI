import os
import sys
import torch
import numpy as np
import subprocess
import argparse
from collections import Counter
from src.config import Config
from src.dataset import DeepfakeDataset
from src.model import MultiTaskDeepfakeModel
from src.utils import accuracy, dice_coef, iou_score, f1

def check_environment():
    print("Checking environment...")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA available: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")

def check_data(show_class_balance=False):
    print("Checking data...")
    train_ds = DeepfakeDataset('train', img_size=Config.IMG_SIZE, aug_strong=False)
    val_ds = DeepfakeDataset('valid', img_size=Config.IMG_SIZE, aug_strong=False)
    assert len(train_ds) > 0, "Train dataset is empty"
    assert len(val_ds) > 0, "Val dataset is empty"
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    if show_class_balance:
        print("\n=== CLASS BALANCE ANALYSIS ===")
        # Use fast label access
        train_labels = train_ds.labels
        val_labels = val_ds.labels
        # Use all labels for exact count
        from collections import Counter
        train_counter = Counter(train_labels)
        val_counter = Counter(val_labels)
        sample_size_train = len(train_labels)
        sample_size_val = len(val_labels)
        print(f"Train - Real: {train_counter[0]}, Fake: {train_counter[1]} (total {sample_size_train})")
        print(f"Train - Real %: {train_counter[0]/sample_size_train*100:.1f}%, Fake %: {train_counter[1]/sample_size_train*100:.1f}%")
        print(f"Val - Real: {val_counter[0]}, Fake: {val_counter[1]} (total {sample_size_val})")
        print(f"Val - Real %: {val_counter[0]/sample_size_val*100:.1f}%, Fake %: {val_counter[1]/sample_size_val*100:.1f}%")
        # Check balance
        train_balance = abs(train_counter[0] - train_counter[1]) / sample_size_train if sample_size_train else 0
        val_balance = abs(val_counter[0] - val_counter[1]) / sample_size_val if sample_size_val else 0
        print(f"Train balance ratio: {train_balance:.3f} (0=perfect, 1=unbalanced)")
        print(f"Val balance ratio: {val_balance:.3f} (0=perfect, 1=unbalanced)")
        if train_balance > 0.0:
            print("⚠️  WARNING: Train set is not perfectly balanced")
        if val_balance > 0.0:
            print("⚠️  WARNING: Val set is not perfectly balanced")
        print("=" * 40)
    
    img, mask, label = train_ds[0]
    assert img.shape[1:] == (Config.IMG_SIZE, Config.IMG_SIZE), "Image shape mismatch"
    assert mask.shape[1:] == (Config.IMG_SIZE, Config.IMG_SIZE), "Mask shape mismatch"
    assert label in [0, 1], "Label not 0 or 1"

def check_model():
    print("Checking model...")
    model = MultiTaskDeepfakeModel(
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        dropout=Config.DROPOUT,
        segmentation=Config.SEGMENTATION,
        attention=Config.ATTENTION
    ).to(Config.DEVICE)
    x = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    cls_logits, seg_logits = model(x)
    assert cls_logits.shape == (2, Config.NUM_CLASSES), "Classification output shape mismatch"
    assert seg_logits.shape[0] == 2 and seg_logits.shape[2:] == (Config.IMG_SIZE, Config.IMG_SIZE), "Segmentation output shape mismatch"

def check_utils():
    print("Checking utils...")
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    acc = accuracy(y_true, y_pred)
    assert acc == 1.0, "Accuracy calculation error"
    dice = dice_coef(y_true, y_pred)
    assert dice > 0, "Dice calculation error"

def run_tests():
    print("Running tests...")
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Tests failed:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print("All tests passed")

def main():
    parser = argparse.ArgumentParser(description='Preflight check for deepfake detection system')
    parser.add_argument('--class-balance', action='store_true', help='Show detailed class balance analysis')
    args = parser.parse_args()
    
    print("=== PREFLIGHT CHECK ===")
    try:
        check_environment()
        check_data(show_class_balance=args.class_balance)
        check_model()
        check_utils()
        run_tests()
        print("=== ALL CHECKS PASSED ===")
        print("System is ready for training!")
    except Exception as e:
        print(f"=== CHECK FAILED: {e} ===")
        sys.exit(1)

if __name__ == "__main__":
    main() 