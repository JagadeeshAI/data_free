#!/usr/bin/env python3
"""
Compute KL / JS divergence between two models' outputs on forget / retain datasets.

Usage examples:
    python compute_kl_diff.py --student-ckpt student_forget_retain.pth --m1-mode random
    python compute_kl_diff.py --student-ckpt student_forget_retain.pth --m1-mode pretrained --data-ratio 1.0
"""

import argparse
import os
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
import timm
import numpy as np

# adapt this import path to your repo layout (assumes script next to repo root)
from data import get_loaders

def build_model(name="vit_tiny_patch16_224", pretrained=False, num_classes=100, device="cpu"):
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model.to(device)

def load_student(ckpt_path, device, model_name="vit_tiny_patch16_224", num_classes=100):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # support multiple checkpoint formats
    if isinstance(ckpt, dict) and ("model_state" in ckpt or "state_dict" in ckpt):
        state = ckpt.get("model_state", ckpt.get("state_dict"))
        model.load_state_dict(state)
    else:
        model.load_state_dict(ckpt)
    return model.to(device)

def safe_logits_forward(model, images):
    """
    Forward and return logits tensor (B, C). Handle dict outputs from some wrappers.
    """
    out = model(images)
    if isinstance(out, dict) and "logits" in out:
        out = out["logits"]
    return out

def per_loader_kl_js(m2, m1, loader, device, csv_path):
    """
    For each sample from loader compute:
      KL(m2 || m1), KL(m1 || m2), JS
    and save to csv_path.
    Returns mean values (kl_pq_mean, kl_qp_mean, js_mean).
    """
    m2.eval()
    m1.eval()
    rows = []
    kl_pq_sum = 0.0
    kl_qp_sum = 0.0
    js_sum = 0.0
    n = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Collecting ({os.path.basename(csv_path)})", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits2 = safe_logits_forward(m2, imgs)
            logits1 = safe_logits_forward(m1, imgs)

            # ensure tensors
            if not torch.is_tensor(logits2):
                logits2 = torch.tensor(logits2).to(device)
            if not torch.is_tensor(logits1):
                logits1 = torch.tensor(logits1).to(device)

            # probabilities and log-probs
            p = F.softmax(logits2, dim=1)          # p = m2
            logp = F.log_softmax(logits2, dim=1)
            q = F.softmax(logits1, dim=1)          # q = m1
            logq = F.log_softmax(logits1, dim=1)

            # KL(p || q) per sample: sum_i p_i * (log p_i - log q_i)
            # compute per-sample sums
            kl_pq = torch.sum(p * (logp - logq), dim=1)  # shape (B,)
            kl_qp = torch.sum(q * (logq - logp), dim=1)

            # JS: m = 0.5*(p+q), JS = 0.5 KL(p||m) + 0.5 KL(q||m)
            m_mix = 0.5 * (p + q)
            logm = torch.log(m_mix + 1e-12)
            js1 = torch.sum(p * (logp - logm), dim=1)
            js2 = torch.sum(q * (logq - logm), dim=1)
            js = 0.5 * (js1 + js2)

            B = imgs.size(0)
            kl_pq_np = kl_pq.cpu().numpy()
            kl_qp_np = kl_qp.cpu().numpy()
            js_np = js.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(B):
                rows.append([n + i, float(kl_pq_np[i]), float(kl_qp_np[i]), float(js_np[i]), int(labels_np[i])])

            kl_pq_sum += kl_pq.sum().item()
            kl_qp_sum += kl_qp.sum().item()
            js_sum += js.sum().item()
            n += B

    # write CSV
    header = ["index", "KL(m2||m1)", "KL(m1||m2)", "JS", "label"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    if n == 0:
        return 0.0, 0.0, 0.0

    return kl_pq_sum / n, kl_qp_sum / n, js_sum / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-ckpt", type=str, required=True, help="path to trained student checkpoint")
    parser.add_argument("--m1-mode", type=str, default="random", choices=["random", "pretrained"], help="m1 initialization mode")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-ratio", type=float, default=0.1, help="pass to get_loaders (0-1)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--num-classes", type=int, default=100)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Device:", device)

    # Load m1
    if args.m1_mode == "random":
        print("Building m1: random init")
        m1 = build_model(name=args.model_name, pretrained=False, num_classes=args.num_classes, device=device)
    else:
        print("Building m1: pretrained")
        m1 = build_model(name=args.model_name, pretrained=True, num_classes=args.num_classes, device=device)

    # Load m2 (student trained model)
    print("Loading student (m2) from:", args.student_ckpt)
    m2 = load_student(args.student_ckpt, device=device, model_name=args.model_name, num_classes=args.num_classes)

    # get loaders
    print("Loading forget (members) and retain (non-member) loaders ...")
    df_train_loader, df_val_loader, _ = get_loaders(batch_size=args.batch_size, start_range=0, end_range=19, data_ratio=args.data_ratio)
    dr_train_loader, dr_val_loader, _ = get_loaders(batch_size=args.batch_size, start_range=20, end_range=99, data_ratio=args.data_ratio, preserve_indices=True)

    # We'll compare on validation sets (non-shuffled) and on train sets optionally;
    # use underlying datasets to create no-shuffle loaders for deterministic per-sample results.
    from torch.utils.data import DataLoader
    forget_member_loader = DataLoader(df_train_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    forget_nonmember_loader = DataLoader(df_val_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    retain_member_loader = DataLoader(dr_train_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    retain_nonmember_loader = DataLoader(dr_val_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # compute divergences on forget group (train vs val)
    print("\nComputing KL/JS on FORGET group (train vs val)...")
    kl_pq_train_forget, kl_qp_train_forget, js_train_forget = per_loader_kl_js(m2, m1, forget_member_loader, device, csv_path="kl_forget_train.csv")
    kl_pq_val_forget, kl_qp_val_forget, js_val_forget = per_loader_kl_js(m2, m1, forget_nonmember_loader, device, csv_path="kl_forget_val.csv")

    # compute divergences on retain group
    print("\nComputing KL/JS on RETAIN group (train vs val)...")
    kl_pq_train_retain, kl_qp_train_retain, js_train_retain = per_loader_kl_js(m2, m1, retain_member_loader, device, csv_path="kl_retain_train.csv")
    kl_pq_val_retain, kl_qp_val_retain, js_val_retain = per_loader_kl_js(m2, m1, retain_nonmember_loader, device, csv_path="kl_retain_val.csv")

    # print summary
    print("\n=== Summary (means) ===")
    print("FORGET group (train):   KL(m2||m1)={:.6f}, KL(m1||m2)={:.6f}, JS={:.6f}".format(kl_pq_train_forget, kl_qp_train_forget, js_train_forget))
    print("FORGET group (val):     KL(m2||m1)={:.6f}, KL(m1||m2)={:.6f}, JS={:.6f}".format(kl_pq_val_forget, kl_qp_val_forget, js_val_forget))
    print("RETAIN group (train):   KL(m2||m1)={:.6f}, KL(m1||m2)={:.6f}, JS={:.6f}".format(kl_pq_train_retain, kl_qp_train_retain, js_train_retain))
    print("RETAIN group (val):     KL(m2||m1)={:.6f}, KL(m1||m2)={:.6f}, JS={:.6f}".format(kl_pq_val_retain, kl_qp_val_retain, js_val_retain))

    print("\nCSV files written: kl_forget_train.csv, kl_forget_val.csv, kl_retain_train.csv, kl_retain_val.csv")
    print("Interpretation hints:")
    print("- Higher KL/JS on FORGET train vs val indicates m2 differs more from m1 on members -> potential privacy leakage")
    print("- If m1 is random and m2 trained, KL will be large (expected). If m1 is pretrained, values tell you how m2 deviated from pretrained baseline.")
    print("- JS is symmetric and bounded (useful summary).")

if __name__ == "__main__":
    main()
