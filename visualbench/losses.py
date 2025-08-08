import torch

def _reduce(loss: torch.Tensor, reduction):
    if reduction is None or reduction == 'none': return loss
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum': return loss.sum()
    raise ValueError(reduction)

def linf_loss(x: torch.Tensor, y: torch.Tensor, reduction='mean'):
    loss = torch.linalg.vector_norm(x-y, ord=float('inf'), dim=list(range(1, x.ndim))) # pylint:disable=not-callable
    return _reduce(loss, reduction)

def median_loss(x: torch.Tensor, y: torch.Tensor, reduction='mean'):
    loss, _ = (x-y).abs().flatten(1, -1).median(1)
    return _reduce(loss, reduction)

def quartic_loss(x: torch.Tensor, y: torch.Tensor, p=4, reduction='mean'):
    loss = ((x-y)**p).mean(list(range(1, x.ndim)))
    return _reduce(loss, reduction)

def rmse_loss(x: torch.Tensor, y: torch.Tensor, reduction='mean'):
    loss = ((x-y).abs().sqrt()).mean(list(range(1, x.ndim)))
    return _reduce(loss, reduction)

def qrmse_loss(x: torch.Tensor, y: torch.Tensor, p=1/4, reduction='mean'):
    loss = ((x-y).abs().pow(p)).mean(list(range(1, x.ndim)))
    return _reduce(loss, reduction)

def norm_loss(x: torch.Tensor, y: torch.Tensor, ord=2, reduction='mean'):
    loss = torch.linalg.vector_norm(x-y, ord=ord, dim=list(range(1, x.ndim))) # pylint:disable=not-callable
    return _reduce(loss, reduction)

def mape_loss(x: torch.Tensor, y: torch.Tensor, epsilon=1e-8, reduction='mean'):
    """mean absolute percentage error"""
    loss = ((x - y).abs() / (y.abs().clip(min=epsilon)))
    loss = loss.mean(list(range(1, x.ndim)))
    return _reduce(loss, reduction)

# def mutual_information(x1, x2, bins=20):
#     # https://matthew-brett.github.io/teaching/mutual_information.html
#     data = torch.stack([x1, x2], -1)
#     hgram, edges = torch.histogramdd(data, bins=[bins, bins])

#     pxy = hgram / torch.sum(hgram)
#     px = torch.sum(pxy, dim=1) # marginal for x over y
#     py = torch.sum(pxy, dim=0) # marginal for y over x
#     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
#     # Now we can do the calculation using the pxy, px_py 2D arrays
#     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
#     return torch.sum(pxy[nzs] * torch.log(pxy[nzs] / px_py[nzs]))