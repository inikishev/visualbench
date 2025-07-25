import torch

from ..utils import totensor
from ..utils.linalg import sinkhorn
from ..benchmark import Benchmark


def _zeros(size, generator):
    return torch.zeros(size, dtype=torch.float32)

class Sorting(Benchmark):
    """differentiable sorting objective via sinkhorn iteration.

    Renders:
        raw logits and permutatation matrix generated from them via sinkhorn iteration.

    Args:
        vec (Any): vector to sort.
        sinkhorn_iters (int, optional): sinkhorn iterations. Defaults to 10.
        binary_weight (float, optional): weight for loss to nudge values to 0 or 1. Defaults to 0.2.
        ortho_weight (float, optional): weight for loss to nudge sums of rows and columns to 0 or 1. Defaults to 0.2.
        p (int, optional): power for loss (1 for L1, 2 for L2). Defaults to 2.
        init (_type_, optional): _description_. Defaults to torch.randn.
    """

    def __init__(
        self,
        vec=torch.randint(0, 100, (100,), generator=torch.Generator().manual_seed(0)),
        sinkhorn_iters: int | None = 10,
        binary_weight=0.2,
        ortho_weight=0.2,
        p=2,
        init=_zeros,
    ):
        super().__init__()

        self.sinkhorn_iters = sinkhorn_iters
        self.binary_weight = binary_weight
        self.ortho_weight = ortho_weight
        self.p = p

        self.vec = torch.nn.Buffer(totensor(vec).ravel().float().contiguous())
        self.P_logits = torch.nn.Parameter(init((self.vec.numel(), self.vec.numel()), generator = self.rng.torch()).contiguous())
        self.identity = torch.nn.Buffer(torch.eye(self.vec.numel()).contiguous())

        self.true_argsort = torch.nn.Buffer(torch.argsort(self.vec).float().contiguous())

    def sorted_vec(self):
        """vec sorted by predicted argsort"""
        if self.sinkhorn_iters is not None: P = sinkhorn(self.P_logits, self.sinkhorn_iters)
        else: P = self.P_logits.softmax(0).softmax(1)
        return self.vec[P.argmax(0)]

    def get_loss(self):
        if self.sinkhorn_iters is not None: P = sinkhorn(self.P_logits, self.sinkhorn_iters)
        else: P = self.P_logits.softmax(0).softmax(1)
        sorted = self.vec @ P

        loss = torch.nn.functional.softplus(sorted[:-1] - sorted[1:]).pow(self.p).mean() # pylint:disable=not-callable

        # encourage 0s and 1s
        binary_loss = torch.sum(P * (1 - P))

        # permutation penalty
        PPt = P @ P.transpose(-2, -1)
        ortho_loss = torch.norm(PPt - self.identity, p='fro')

        if self._make_images:
            self.log_image('image logits', self.P_logits, to_uint8=True)
            self.log_image('image permutation', P, to_uint8=True)

        with torch.no_grad():
            predicted_argsort = P.argmax(0).float()
            self.log('accuracy', (predicted_argsort == self.true_argsort).sum() / P.size(0), plot=True)
            self.log('distance', torch.nn.functional.l1_loss(predicted_argsort, self.true_argsort), plot=True)

        return loss + binary_loss*self.binary_weight + ortho_loss*self.ortho_weight
