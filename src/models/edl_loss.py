import torch
import torch.nn.functional as F


def kl_divergence(alpha, num_classes):
    beta = torch.ones((1, num_classes)).to(alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def edl_loss(evidence, target, num_classes, epoch, annealing_step):

    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)

    probs = alpha / S
    one_hot = F.one_hot(target, num_classes=num_classes).float()

    mse = torch.sum((one_hot - probs) ** 2, dim=1, keepdim=True)
    var = torch.sum(probs * (1 - probs) / (S + 1), dim=1, keepdim=True)

    loss = mse + var

    annealing_coef = min(1.0, epoch / annealing_step)

    kl = kl_divergence(alpha, num_classes)

    return torch.mean(loss + annealing_coef * kl)