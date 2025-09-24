# mia.py
"""
Membership Inference Attack (direct / logistic-regression) for your unlearning experiment.

Usage:
    python mia.py --ckpt student_forget_retain.pth
This script will:
  - load the student model (timm ViT tiny, same architecture as your training script)
  - build "member" and "non-member" sets for both Forget (0-19) and Retain (20-99)
    using the same data_ratio and batch_size as your training script
  - extract softmax probabilities and per-sample cross-entropy loss as features
  - train a logistic regression attack classifier and report Accuracy / AUC / Precision / Recall / F1

Only run against models/datasets you own/have permission to test.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

# adapt imports to your repo layout
from data import get_loaders

# ----------------------------
# Utils for collecting features
# ----------------------------
def collect_model_outputs(model, loader, device):
    """
    Collect softmax probs, per-sample CE loss, and true labels for all samples in loader.
    Returns: probs (N,C), losses (N,), labels (N,)
    """
    model.eval()
    probs_list, loss_list, labels_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting outputs", leave=False):
            # loader expected to yield (images, labels, ...)
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                raise ValueError("Loader must return (images, labels, ...)")
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            # handle HF-style outputs
            if isinstance(logits, dict) and 'logits' in logits:
                logits = logits['logits']

            if not torch.is_tensor(logits):
                logits = torch.tensor(logits).to(device)

            probs = F.softmax(logits, dim=1)
            ce = F.cross_entropy(logits, labels, reduction='none')

            probs_list.append(probs.cpu().numpy())
            loss_list.append(ce.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    losses = np.concatenate(loss_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return probs, losses, labels

def prepare_features(probs, losses, use_entropy=True, use_loss=True):
    feats = [probs]
    if use_entropy:
        eps = 1e-12
        entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1, keepdims=True)
        feats.append(entropy)
    if use_loss:
        feats.append(losses.reshape(-1, 1))
    X = np.concatenate(feats, axis=1)
    return X

def train_attack_classifier(X_train, y_train, X_test=None, y_test=None):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    clf.fit(Xs, y_train)
    results = {'clf': clf, 'scaler': scaler}
    if X_test is not None:
        Xv = scaler.transform(X_test)
        y_pred = clf.predict(Xv)
        y_score = clf.predict_proba(Xv)[:, 1]
        results['acc'] = accuracy_score(y_test, y_pred)
        try:
            results['auc'] = roc_auc_score(y_test, y_score)
        except ValueError:
            results['auc'] = float('nan')
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        results.update({'precision': prec, 'recall': rec, 'f1': f1})
    return results

# ----------------------------
# Attack runner for one group
# ----------------------------
def run_direct_mia_for_group(student, member_loader, nonmember_loader, device, attack_train_frac=0.7, random_seed=42):
    print("Collecting member (train) outputs...")
    mem_probs, mem_losses, _ = collect_model_outputs(student, member_loader, device)
    print("Collecting non-member (val) outputs...")
    non_probs, non_losses, _ = collect_model_outputs(student, nonmember_loader, device)

    X_mem = prepare_features(mem_probs, mem_losses)
    X_non = prepare_features(non_probs, non_losses)

    X = np.concatenate([X_mem, X_non], axis=0)
    y = np.concatenate([np.ones(len(X_mem)), np.zeros(len(X_non))], axis=0)

    # shuffle
    rng = np.random.RandomState(random_seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n_train = int(len(X) * attack_train_frac)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    attack = train_attack_classifier(X_train, y_train, X_test, y_test)
    print("=== Attack results ===")
    print(f"Test samples: {len(y_test)}")
    print(f"Attack Acc: {attack.get('acc', float('nan'))*100:.2f}%")
    print(f"Attack AUC: {attack.get('auc', float('nan')):.4f}")
    print(f"Precision: {attack.get('precision', float('nan')):.4f}, Recall: {attack.get('recall', float('nan')):.4f}, F1: {attack.get('f1', float('nan')):.4f}")
    return attack

# ----------------------------
# Main script
# ----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load model architecture (same as training script)
    num_classes = args.num_classes
    student = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    # load provided checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # support both {"model_state": state_dict} and plain state_dict or {"state_dict":...}
    if isinstance(ckpt, dict) and ("model_state" in ckpt or "state_dict" in ckpt):
        state = ckpt.get("model_state", ckpt.get("state_dict"))
        student.load_state_dict(state)
    else:
        student.load_state_dict(ckpt)
    student = student.to(device)
    student.eval()
    print("Loaded student model from:", args.ckpt)

    # Build loaders for forget (members) and their val (non-members)
    # NOTE: get_loaders returns DataLoaders that might shuffle; we create new DataLoader with shuffle=False
    df_train_loader, df_val_loader, _ = get_loaders(batch_size=args.batch_size, start_range=args.forget_start, end_range=args.forget_end, data_ratio=args.data_ratio, preserve_indices=False)
    dr_train_loader, dr_val_loader, _ = get_loaders(batch_size=args.batch_size, start_range=args.retain_start, end_range=args.retain_end, data_ratio=args.data_ratio, preserve_indices=True)

    # Recreate deterministic loaders from underlying datasets for attack (no shuffle)
    mem_forget_loader = DataLoader(df_train_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    nonmem_forget_loader = DataLoader(df_val_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    mem_retain_loader = DataLoader(dr_train_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    nonmem_retain_loader = DataLoader(dr_val_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("\nRunning MIA on FORGET group (members = train 0-19, non-members = val 0-19):")
    run_direct_mia_for_group(student, mem_forget_loader, nonmem_forget_loader, device,
                              attack_train_frac=args.attack_train_frac, random_seed=args.seed)

    print("\nRunning MIA on RETAIN group (members = train 20-99, non-members = val 20-99):")
    run_direct_mia_for_group(student, mem_retain_loader, nonmem_retain_loader, device,
                              attack_train_frac=args.attack_train_frac, random_seed=args.seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct MIA for unlearning experiment")
    parser.add_argument("--ckpt", type=str, required=True, help="path to student checkpoint (student_forget_retain.pth)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--forget_start", type=int, default=0)
    parser.add_argument("--forget_end", type=int, default=19)
    parser.add_argument("--retain_start", type=int, default=20)
    parser.add_argument("--retain_end", type=int, default=99)
    parser.add_argument("--data_ratio", type=float, default=0.1, help="same data_ratio used during training (0-1)")
    parser.add_argument("--attack_train_frac", type=float, default=0.7, help="fraction of attack dataset used for training attacker")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
