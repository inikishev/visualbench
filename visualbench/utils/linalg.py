import torch


def sinkhorn(logits, num_iters=10):
    """Applies Sinkhorn normalization to logits to generate a doubly stochastic matrix."""
    log_alpha = logits
    for _ in range(num_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)