import torch
import torch.nn as nn

# -------------------------
# Param counting helper
# -------------------------
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M "
          f"({(trainable/total)*100:.2f}%)")
    return total, trainable


# -------------------------
# Freeze N-1 blocks, unfreeze only FFN of last block
# -------------------------

def freeze_n_1(model, num_unfreeze: int = 1):
    """
    Freeze all transformer blocks except the last `num_unfreeze` blocks.
    Unfreeze both attention + MLP inside those blocks.
    Keep classifier head frozen.
    """
    print("Before freezing:")
    count_params(model)

    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "blocks"):
        if num_unfreeze > len(model.blocks):
            raise ValueError(f"Model has {len(model.blocks)} blocks, requested {num_unfreeze}.")
        for i in range(-num_unfreeze, 0):
            for p in model.blocks[i].parameters():
                p.requires_grad = True
    else:
        raise AttributeError("Model missing `blocks` attr (not a ViT).")

    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = False
    if hasattr(model, "fc"):  # fallback
        for p in model.fc.parameters():
            p.requires_grad = False

    print("After freezing:")
    count_params(model)

    return model


# -------------------------
# Loss functions
# -------------------------
def retain_loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def forget_loss_fn(outputs, labels):
    ce_loss = nn.CrossEntropyLoss(reduction="none")(outputs, labels)
    return torch.relu(8.0 - ce_loss).mean()


# -------------------------
# Jag_Reg with normalization
# -------------------------
class Jag_Reg(nn.Module):
    """
    Penalize deviation between student and teacher across last `num_blocks`.
    Normalized so scale of reg_loss is stable.
    """
    def __init__(self, teacher, student, num_blocks=1, p=2):
        super(Jag_Reg, self).__init__()
        self.p = p
        self.teacher_blocks = teacher.blocks[-num_blocks:]
        self.student_blocks = student.blocks[-num_blocks:]

        for block in self.teacher_blocks:
            for param in block.parameters():
                param.requires_grad = False

    def forward(self):
        reg_loss = 0.0
        count = 0
        for t_block, s_block in zip(self.teacher_blocks, self.student_blocks):
            for (_, t_param), (_, s_param) in zip(t_block.named_parameters(),
                                                 s_block.named_parameters()):
                reg_loss += torch.norm(s_param - t_param, p=self.p)
                count += 1
        return reg_loss / (count + 1e-8)

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


import torch
import torch.nn.functional as F

def strong_kl_complement_loss(logits, labels, alpha=0.01, beta=5.0, gamma=0.1, use_reverse=False):
    """
    Stronger forgetting loss:
    Combines KL-to-complement, margin penalty, squared KL, and entropy.
    
    Args:
        logits: [batch_size, num_classes]
        labels: [batch_size]
        alpha: margin threshold for true class probability
        beta: weight for margin penalty
        gamma: weight for entropy reward
        use_reverse: if True, also adds reverse KL(q||p)
    """
    probs = F.softmax(logits, dim=1)          # p
    log_probs = F.log_softmax(logits, dim=1)  # log p
    C = logits.shape[1]

    # Complement distribution q
    q = torch.full_like(probs, 1.0 / (C - 1))     # uniform over wrong classes
    q.scatter_(1, labels.unsqueeze(1), 0.0)       # zero on true class

    # Forward KL(p||q)
    log_q = torch.log(q + 1e-12)                  # stability
    kl_forward = (probs * (log_probs - log_q)).sum(dim=1)  # [batch]

    # (Optional) Reverse KL(q||p)
    if use_reverse:
        kl_reverse = (q * (torch.log(q + 1e-12) - log_probs)).sum(dim=1)
        kl = kl_forward + kl_reverse
    else:
        kl = kl_forward

    # Margin penalty: force p_true < alpha
    p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    margin_penalty = F.relu(p_true - alpha)

    # Entropy term (maximize uncertainty)
    entropy = -(probs * log_probs).sum(dim=1)

    # Final loss (mean over batch)
    loss = kl.mean() + beta * margin_penalty.mean()  + (kl ** 2).mean() - gamma * entropy.mean()

    return loss
