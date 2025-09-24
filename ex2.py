import os
import sys
import torch
import torch.optim as optim
from tqdm import tqdm
import timm

sys.path.append('..')
from data import get_loaders
from utils import (
    freeze_n_1,
    count_params,
    Jag_Reg,
    retain_loss_fn,
    forget_loss_fn,
    kl_to_complement_loss,
    strong_kl_complement_loss
)


# -----------------------------
# Training
# -----------------------------
def train_one_epoch_unlearning(student, df_loader, dr_loader, optimizer, device,
                               jag_reg=None, lambda_reg=0.0):
    student.train()
    total_forget_loss, total_retain_loss, total_reg_loss = 0.0, 0.0, 0.0
    forget_correct, retain_correct = 0, 0
    forget_total, retain_total = 0, 0

    pbar = tqdm(zip(df_loader, dr_loader),
                desc="Train (forget+retain)",
                leave=False,
                total=min(len(df_loader), len(dr_loader)))
    
    for (f_imgs, f_labels), (r_imgs, r_labels) in pbar:
        f_imgs, f_labels = f_imgs.to(device), f_labels.to(device)
        r_imgs, r_labels = r_imgs.to(device), r_labels.to(device)

        optimizer.zero_grad()

        # --- Forget forward (with strong loss) ---
        f_outputs = student(f_imgs)
        f_loss = strong_kl_complement_loss(f_outputs, f_labels)

        # --- Retain forward (monitor only) ---
        with torch.no_grad():
            r_outputs = student(r_imgs)
            r_loss = retain_loss_fn(r_outputs, r_labels)

        # --- Regularization ---
        reg_loss = torch.tensor(0.0, device=device)
        if jag_reg is not None:
            reg_loss = jag_reg()

        # --- Total loss ---
        loss = f_loss + lambda_reg * reg_loss
        loss.backward()
        optimizer.step()

        # --- Tracking ---
        total_forget_loss += f_loss.item() * f_imgs.size(0)
        total_retain_loss += r_loss.item() * r_imgs.size(0)
        total_reg_loss += reg_loss.item() * f_imgs.size(0)

        # Accs
        _, f_preds = torch.max(f_outputs, 1)
        forget_correct += (f_preds == f_labels).sum().item()
        forget_total += f_labels.size(0)

        _, r_preds = torch.max(r_outputs, 1)
        retain_correct += (r_preds == r_labels).sum().item()
        retain_total += r_labels.size(0)

        pbar.set_postfix(
            f_loss=f_loss.item(),
            reg_raw=reg_loss.item(),
            reg_scaled=lambda_reg * reg_loss.item(),
            total=loss.item()
        )

    # --- Epoch averages ---
    avg_forget_loss = total_forget_loss / forget_total if forget_total > 0 else 0
    avg_retain_loss = total_retain_loss / retain_total if retain_total > 0 else 0
    avg_reg_loss = total_reg_loss / forget_total if forget_total > 0 else 0
    forget_acc = forget_correct / forget_total if forget_total > 0 else 0
    retain_acc = retain_correct / retain_total if retain_total > 0 else 0

    return avg_forget_loss, avg_retain_loss, avg_reg_loss, forget_acc, retain_acc


# -----------------------------
# Validation
# -----------------------------
def validate_unlearning(student, val_forget_loader, val_retain_loader, device):
    student.eval()
    forget_correct, retain_correct = 0, 0
    forget_total, retain_total = 0, 0
    forget_loss_sum, retain_loss_sum = 0.0, 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_forget_loader, desc="Val (forget)", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            loss = forget_loss_fn(outputs, labels)
            forget_loss_sum += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            forget_correct += (preds == labels).sum().item()
            forget_total += labels.size(0)

        for images, labels in tqdm(val_retain_loader, desc="Val (retain)", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            loss = retain_loss_fn(outputs, labels)
            retain_loss_sum += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            retain_correct += (preds == labels).sum().item()
            retain_total += labels.size(0)

    forget_acc = forget_correct / forget_total if forget_total > 0 else 0
    retain_acc = retain_correct / retain_total if retain_total > 0 else 0
    avg_forget_loss = forget_loss_sum / forget_total if forget_total > 0 else 0
    avg_retain_loss = retain_loss_sum / retain_total if retain_total > 0 else 0

    return avg_forget_loss, avg_retain_loss, forget_acc, retain_acc


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Forget = 0-19, Retain = 20-99
    df_train_loader, df_val_loader, _ = get_loaders(batch_size=64, start_range=0, end_range=19, data_ratio=0.1)
    dr_train_loader, dr_val_loader, _ = get_loaders(batch_size=64, start_range=20, end_range=99, preserve_indices=True, data_ratio=0.1)

    num_classes = 100
    ckpt_path = "/media/jag/volD2/data_free/pre_train/best_vit_tiny.pth"

    teacher = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    teacher.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)

    student = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    student.load_state_dict(teacher.state_dict())
    student = student.to(device)

    jag_reg = Jag_Reg(teacher, student, num_blocks=1, p=2).to(device)

    count_params(student)
    student = freeze_n_1(student, num_unfreeze=1)
    count_params(student)

    optimizer = optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=3e-4, weight_decay=0.05)

    num_epochs = 20
    save_path = "student_forget_retain.pth"

    # Initial val
    f_loss, r_loss, f_acc, r_acc = validate_unlearning(student, df_val_loader, dr_val_loader, device)
    print(f"Initial Val - Forget Loss: {f_loss:.4f} | Forget Acc: {f_acc*100:.2f}% | "
          f"Retain Loss: {r_loss:.4f} | Retain Acc: {r_acc*100:.2f}%")

    lambda_reg = 50  

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_f_loss, train_r_loss, train_reg_loss, train_f_acc, train_r_acc = train_one_epoch_unlearning(
            student, df_train_loader, dr_train_loader, optimizer, device, jag_reg=jag_reg, lambda_reg=lambda_reg
        )

        # Validate
        val_f_loss, val_r_loss, val_f_acc, val_r_acc = validate_unlearning(
            student, df_val_loader, dr_val_loader, device
        )

        print(f"Train - Forget Loss: {train_f_loss:.4f} | Retain Loss: {train_r_loss:.4f} "
              f"| Reg Loss(raw): {train_reg_loss:.4f} | Forget Acc: {train_f_acc*100:.2f}% | Retain Acc: {train_r_acc*100:.2f}%")
        print(f"Val   - Forget Loss: {val_f_loss:.4f} | Retain Loss: {val_r_loss:.4f} "
              f"| Forget Acc: {val_f_acc*100:.2f}% | Retain Acc: {val_r_acc*100:.2f}%")

        torch.save({"model_state": student.state_dict()}, save_path)
        print(f"✅ Saved checkpoint at epoch {epoch} with Retain Acc: {val_r_acc*100:.2f}%")


if __name__ == "__main__":
    main()
