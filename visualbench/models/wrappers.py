from collections.abc import Sequence
from functools import partial
import torch

from ..utils import totensor
from ..rng import RNG

def _get_lr(opt):
    return next(iter(opt.param_groups))["lr"]

def _decay_lr_(opt):
    for p in opt.param_groups:
        p["lr"] *= 0.99

class ConstantInput(torch.nn.Module):
    """wraps another model and passes it constant input, optionally optimizes input to output x0 under norm constraint"""
    @torch.no_grad
    def __init__(self, model: torch.nn.Module, input: int | Sequence[int] | torch.Tensor, x0=None, noise:float=0, maxiter=1000, tol=1e-4, progress:bool=True, device='cuda', seed: int | None =0):
        super().__init__()

        generator = torch.Generator(device)
        if seed is not None: generator = generator.manual_seed(seed)
        if not isinstance(input, torch.Tensor): input = torch.randn(input, device=device, generator=generator)
        input = input.to(device)
        model = model.to(device)

        if x0 is not None:
            x0 = totensor(x0).to(device=device, dtype=torch.float32)
            xx = (x0 * x0).mean()
            input = input.requires_grad_(True)

            params = [input, *model.parameters()]
            opt = torch.optim.LBFGS(params, line_search_fn='strong_wolfe')

            def closure(backward=True, w_norm=1.):
                with torch.enable_grad():
                    out = model(input)
                    loss1 = torch.nn.functional.mse_loss(out, x0)
                    loss2 = torch.nn.functional.softplus((input * input).mean() - xx, beta=5) # pylint:disable=not-callable
                    loss = loss1 + w_norm*loss2
                    if backward:
                        opt.zero_grad()
                        loss.backward()
                    return loss

            pbar = range(maxiter)
            if progress:
                import importlib.util
                if importlib.util.find_spec("tqdm") is not None:
                    import tqdm
                    pbar = tqdm.trange(maxiter)

            prev_loss = torch.inf
            w_norm = 1
            div = 2
            lr = 1

            prev_fail = False
            for _ in pbar:
                loss = opt.step(partial(closure, w_norm=w_norm))

                assert loss is not None
                if loss < tol: break
                if loss < prev_loss - 1e-8:
                    prev_loss = loss
                    prev_fail = False

                else:
                    if prev_fail:
                        if isinstance(opt, torch.optim.LBFGS):
                            import importlib.util
                            if importlib.util.find_spec('torchzero') is not None:
                                import torchzero as tz
                                opt = tz.Optimizer(
                                    params,
                                    tz.m.SOAP(beta1=0.9), tz.m.NormalizeByEMA(), tz.m.WeightDecay(0.01), tz.m.LR(lr)
                                )
                            else:
                                opt = torch.optim.AdamW(params, lr, (0.9, 0.95))
                        else:
                            _decay_lr_(opt)
                    w_norm /= div
                    div *= 2
                    if div > 1e10: div = 1e10
                    prev_fail = True

                if hasattr(pbar, "set_postfix_str"):
                    pbar.set_postfix_str(f"{opt.__class__.__name__} {_get_lr(opt):.5f}: {loss.item():.5f}") # type:ignore

        self.model = model
        self.input = torch.nn.Buffer(input.requires_grad_(False))
        self.input_mad = torch.nn.Buffer(input.abs().mean())
        self.noise = noise
        self.generator = generator
        self.to(device)

    def forward(self):
        input = self.input
        if self.noise != 0:
            noise = torch.randn(input.size(), device=input.device, dtype=input.dtype, generator=self.generator)
            input = input + noise * self.noise * self.input_mad
        return self.model(input)


class RandomInput(torch.nn.Module):
    """wraps another model and passes it random input"""
    @torch.no_grad
    def __init__(self, model: torch.nn.Module, input_size: int | Sequence[int], seed: int | None =0):
        super().__init__()

        self.model = model
        self.input_size = input_size
        self.rng = RNG(seed)

    def forward(self):
        p = next(iter(self.model.parameters()))
        input = torch.randn(self.input_size, device=p.device, dtype=p.dtype, generator=self.rng.torch(p.device))
        return self.model(input)
