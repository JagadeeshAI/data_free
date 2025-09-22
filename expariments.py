# train.py
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm
import sys
sys.path.append('..')

from data import get_loaders
from utils import freeze_n_1, Jag_Reg, kl_to_complement_loss


def train_one_epoch_unlearning(student, forget_loader, optimizer, device, jag_reg=None):
    student.train()
    forget_loss_sum, reg_loss_sum = 0.0, 0.0
    forget_correct, forget_total = 0, 0
    total_samples = 0

    pbar = tqdm(forget_loader, desc="Train (forget)", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = student(images)

        # forgetting loss (this is the actual optimization objective)
        f_loss = kl_to_complement_loss(outputs, labels)

        # compute reg loss (not added into optimization)
        reg_loss = torch.tensor(0.0, device=device)
        if jag_reg is not None:
            reg_loss = jag_reg()

        loss = f_loss
        loss.backward()
        optimizer.step()

        forget_loss_sum += f_loss.item() * batch_size
        reg_loss_sum += reg_loss.item() * batch_size
        total_samples += batch_size

        # track forget accuracy
        _, preds = torch.max(outputs, 1)
        forget_correct += (preds == labels).sum().item()
        forget_total += labels.size(0)

        pbar.set_postfix(f_loss=f_loss.item(), reg=reg_loss.item())

    avg_f_loss = forget_loss_sum / total_samples if total_samples > 0 else 0
    avg_reg_loss = reg_loss_sum / total_samples if total_samples > 0 else 0
    forget_acc = forget_correct / forget_total if forget_total > 0 else 0

    return avg_f_loss, avg_reg_loss, forget_acc


def validate_unlearning(student, val_forget_loader, val_retain_loader, device):
    student.eval()
    retain_loss_sum = 0.0
    retain_correct, retain_total = 0, 0
    forget_correct, forget_total = 0, 0

    with torch.no_grad():
        # Forget validation (accuracy only, no loss)
        for images, labels in tqdm(val_forget_loader, desc="Val (forget)", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            _, preds = torch.max(outputs, 1)
            forget_correct += (preds == labels).sum().item()
            forget_total += labels.size(0)

        # Retain validation (acc + CE loss)
        for images, labels in tqdm(val_retain_loader, desc="Val (retain)", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            retain_loss_sum += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            retain_correct += (preds == labels).sum().item()
            retain_total += labels.size(0)

    retain_loss_avg = retain_loss_sum / retain_total if retain_total > 0 else 0
    forget_acc = forget_correct / forget_total if forget_total > 0 else 0
    retain_acc = retain_correct / retain_total if retain_total > 0 else 0

    return retain_loss_avg, forget_acc, retain_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data
    train_forget_loader, val_forget_loader, _ = get_loaders(batch_size=64, start_range=0, end_range=19)
    _, val_retain_loader, _ = get_loaders(batch_size=64, start_range=20, end_range=99, preserve_indices=True)

    # Teacher & student
    num_classes = 100
    ckpt_path = "/media/jag/volD2/data_free/pre_train/best_vit_tiny.pth"

    teacher = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    teacher.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)

    student = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    student.load_state_dict(teacher.state_dict())
    student = student.to(device)

    student = freeze_n_1(student)

    # Jag_Reg
    jag_reg = Jag_Reg(teacher, student, block_idx=-1, p=2).to(device)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.05)

    num_epochs = 20

    # CSV logging
    csv_file = "loss_tracking.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_f_loss", "retain_loss", "forget_acc", "retain_acc", "reg_loss"])

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        avg_f_loss, avg_reg_loss, train_forget_acc = train_one_epoch_unlearning(
            student, train_forget_loader, optimizer, device, jag_reg=jag_reg
        )

        retain_loss, val_forget_acc, val_retain_acc = validate_unlearning(
            student, val_forget_loader, val_retain_loader, device
        )

        print(f"Train - F Loss: {avg_f_loss:.4f} | Forget Acc: {train_forget_acc*100:.2f}% | Reg Loss: {avg_reg_loss:.6f}")
        print(f"Val   - Forget Acc: {val_forget_acc*100:.2f}% | Retain Loss: {retain_loss:.4f} | Retain Acc: {val_retain_acc*100:.2f}%")

        # Write to CSV
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_f_loss, retain_loss, val_forget_acc*100, val_retain_acc*100, avg_reg_loss])

    print("\nTraining complete! Logged results in loss_tracking.csv")


if __name__ == "__main__":
    main()
