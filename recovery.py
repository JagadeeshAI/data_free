# finetune_student.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm
import sys
sys.path.append('..')

from data import get_loaders


def train_one_epoch(student, train_loader, optimizer, device):
    student.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc="Finetune (train)", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = student(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct / total if total > 0 else 0.0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def validate(student, val_loader, device):
    student.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Finetune (val)", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset (range 0-49)
    train_loader, val_loader, _ = get_loaders(batch_size=32, start_range=0, end_range=20,data_ratio=0.05)

    num_classes = 100
    # Load student model
    student = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=num_classes)
    checkpoint = torch.load("student_forget_retain.pth", map_location="cpu")
    student.load_state_dict(checkpoint["model_state"])
    student = student.to(device)

    # Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=1e-5, weight_decay=0.05)

    num_epochs = 10  # you can increase if needed

    val_loss, val_acc = validate(student, val_loader, device)
    # print(f"Initial Val - Loss: {val_loss:.4f} | Forget Acc (0-49): {val_acc*100:.2f}%")
    # exit()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(student, train_loader, optimizer, device)
        val_loss, val_acc = validate(student, val_loader, device)

        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f} | Forget Acc (0-49): {val_acc*100:.2f}%")


if __name__ == "__main__":
    main()
