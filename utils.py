import torch
import torch.nn as nn

# -------------------------
# Param counting helper
# -------------------------
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total / 1e6, trainable / 1e6  # raw + Millions


# -------------------------
# Freeze N-1 blocks, unfreeze only FFN (fc1, fc2) of last block
# -------------------------
def freeze_n_1(model):
    """
    Freeze all transformer blocks except the last one.
    Inside the last block, only unfreeze the FFN (mlp.fc1, mlp.fc2).
    Keep classifier head frozen.
    Works for timm ViT models like vit_tiny_patch16_224.
    """
    print("Before freezing:")
    total, trainable, total_m, trainable_m = count_params(model)
    print(f"  Total params: {total_m:.2f}M | Trainable params: {trainable_m:.2f}M "
          f"({(trainable/total)*100:.2f}%)")

    # Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze mlp.fc1 and mlp.fc2 of last block
    if hasattr(model, "blocks"):
        last_block = model.blocks[-1]
        if hasattr(last_block, "mlp"):
            for name in ["fc1", "fc2"]:
                if hasattr(last_block.mlp, name):
                    for p in getattr(last_block.mlp, name).parameters():
                        p.requires_grad = True
        else:
            raise AttributeError("Last block has no mlp submodule")
    else:
        raise AttributeError("Model does not have `blocks` attribute. Check architecture.")

    print("After freezing:")
    total, trainable, total_m, trainable_m = count_params(model)
    print(f"  Total params: {total_m:.2f}M | Trainable params: {trainable_m:.2f}M "
          f"({(trainable/total)*100:.2f}%)")

    return model

def kl_to_uniform_loss(logits):
    """
    KL(p || U) where U is uniform. Minimizing this increases entropy of p.
    Analytic expression: KL = sum p log p + log C
    We return mean KL over batch (torch scalar).
    """
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    C = logits.shape[1]
    kl = (probs * log_probs).sum(dim=1) + torch.log(torch.tensor(float(C), device=logits.device))
    return kl.mean()


def forget_loss(outputs, labels):
    """Custom forgetting loss: ReLU(110 - CrossEntropy)"""
    ce_loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
    return torch.relu(110 - ce_loss).mean()


def kl_to_complement_loss(logits, labels):
    """
    KL(p || q) where q is the complement distribution:
    q = 0 for true class, 1/(C-1) for all other classes.
    This encourages the model to avoid predicting the true label.
    """
    probs = torch.softmax(logits, dim=1)          # p
    log_probs = torch.log_softmax(logits, dim=1)  # log p
    C = logits.shape[1]

    # Build q (batch_size x C)
    q = torch.full_like(probs, 1.0 / (C - 1))     # fill with 1/(C-1)
    q.scatter_(1, labels.unsqueeze(1), 0.0)       # set true class prob = 0

    # KL(p || q) = sum_i p_i (log p_i - log q_i)
    log_q = torch.log(q + 1e-12)                  # numerical stability
    kl = (probs * (log_probs - log_q)).sum(dim=1)

    return kl.mean()



class Jag_Reg(nn.Module):
    """
    Regularizer to penalize deviation between student and teacher FFN (mlp.fc1, mlp.fc2)
    of the last transformer block.
    """
    def __init__(self, teacher, student, block_idx=-1, p=2):
        super(Jag_Reg, self).__init__()
        self.block_idx = block_idx
        self.p = p  # norm: 1 = L1, 2 = L2
        # store references to teacher + student last block FFN
        self.teacher_ffn = teacher.blocks[block_idx].mlp
        self.student_ffn = student.blocks[block_idx].mlp

        # Freeze teacher parameters (safety)
        for p in self.teacher_ffn.parameters():
            p.requires_grad = False

    def forward(self):
        reg_loss = 0.0
        # compare fc1 and fc2
        for name in ["fc1", "fc2"]:
            if hasattr(self.teacher_ffn, name) and hasattr(self.student_ffn, name):
                t_w = getattr(self.teacher_ffn, name).weight
                s_w = getattr(self.student_ffn, name).weight
                reg_loss = reg_loss + torch.norm(s_w - t_w, p=self.p)

                # also include biases if they exist
                if getattr(self.teacher_ffn, name).bias is not None:
                    t_b = getattr(self.teacher_ffn, name).bias
                    s_b = getattr(self.student_ffn, name).bias
                    reg_loss = reg_loss + torch.norm(s_b - t_b, p=self.p)
        return reg_loss
