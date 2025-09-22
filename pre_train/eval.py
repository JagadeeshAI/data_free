import torch
import torch.nn as nn
from tqdm import tqdm
import timm

from data import get_loaders   # same as in train.py


# ----------------------
# Validation function
# ----------------------
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=True)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ----------------------
# Main Evaluation
# ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (use validation loader as test set)
    _, val_loader, classes = get_loaders(batch_size=64)
    num_classes = len(classes)

    # Model: must match the one trained
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    model = model.to(device)

    # Load best checkpoint
    checkpoint_path = "best_vit_tiny.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    criterion = nn.CrossEntropyLoss()

    # Run evaluation
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print("\n===== Evaluation Results =====")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Acc : {val_acc*100:.2f}%")


if __name__ == "__main__":
    main()
