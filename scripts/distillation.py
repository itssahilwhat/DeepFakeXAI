import os
import sys
import logging
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

from src.data import get_dataloader
from src.model import EfficientNetLiteTemporal
from src.utils import save_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distill")

def distill(teacher_path, student_out, dataset):
    # Load teacher
    teacher = EfficientNetLiteTemporal(num_classes=1, pretrained=False)
    teacher_ckpt = torch.load(teacher_path, map_location=Config.DEVICE)
    teacher.load_state_dict(teacher_ckpt["state_dict"])
    teacher.to(Config.DEVICE)
    teacher.eval()

    # Student model (same architecture for simplicity)
    student = EfficientNetLiteTemporal(num_classes=1, pretrained=False)
    student.to(Config.DEVICE)
    student.train()
    optimizer = Adam(student.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()

    train_loader = get_dataloader(dataset, "train", batch_size=Config.BATCH_SIZE, shuffle=True)
    criterion = torch.nn.MSELoss()

    for epoch in range(Config.EPOCHS):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}"):
            imgs = batch["image"].to(Config.DEVICE)
            with autocast():
                teacher_out = teacher(imgs)
                student_out_pred = student(imgs)
                loss = criterion(student_out_pred, teacher_out)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        logger.info(f"Epoch {epoch+1}: Distillation loss={epoch_loss/len(train_loader):.4f}")
    save_checkpoint({"state_dict": student.state_dict()}, student_out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Teacher-Student Distillation")
    parser.add_argument("--teacher", required=True, help="Path to teacher checkpoint")
    parser.add_argument("--student_out", required=True, help="Path to save student checkpoint")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    args = parser.parse_args()
    distill(args.teacher, args.student_out, args.dataset)
