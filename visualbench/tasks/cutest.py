from typing import TYPE_CHECKING

import numpy as np
import torch

from ..benchmark import Benchmark

if TYPE_CHECKING:
    import pycutest

# tutorial of how to install pycutest on an immutable distro
# so just install vscode in ubuntu distrobox and follow
# this in it https://jfowkes.github.io/pycutest/_build/html/install.html
# no need to even install miniconda in distrobox it can use host one

class _CUTEstGrad(torch.autograd.Function): # pylint:disable=abstract-method
    """gradient as autograd func for hvps"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, problem: "pycutest.CUTEstProblem", r, J): # type:ignore
        assert problem.m == 0

        ctx.save_for_backward(x)
        ctx.problem = problem

        x_np = x.numpy(force=True)
        ctx.x_np = x_np

        grad_np = problem.grad(x_np)
        ctx.r = ctx.J = None

        return torch.tensor(grad_np, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor): # type:ignore
        x, = ctx.saved_tensors
        x_np = ctx.x_np
        problem = ctx.problem

        v_np = grad_output.numpy(force=True)
        hvp_np = problem.hprod(p=v_np, x=x_np)

        # grads for (x, problem)
        return torch.tensor(hvp_np, device=x.device, dtype=x.dtype), None, None, None


class _CUTEstObj(torch.autograd.Function): # pylint:disable=abstract-method
    @staticmethod
    def forward(ctx, x: torch.Tensor, problem: "pycutest.CUTEstProblem"): # type:ignore
        ctx.save_for_backward(x)
        ctx.problem = problem

        if problem.m == 0:
            f = problem.obj(x.numpy(force=True))
            ctx.r = ctx.J = None

        else:
            # for least squares cons returns residuals and jacobian
            f, J = problem.cons(x.numpy(force=True), gradient=True) # type:ignore
            ctx.r = f
            ctx.J = J

        return torch.tensor(f, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor): # type:ignore
        problem = ctx.problem
        x = ctx.saved_tensors[0]

        if problem.m == 0:
            grad = _CUTEstGrad.apply(x, problem, ctx.r, ctx.J)
            return grad_output * grad, None # type:ignore

        # least squares
        J = torch.tensor(ctx.J, device=x.device, dtype=x.dtype)
        return J.T @ grad_output, None

def cutest_obj(x: torch.Tensor, problem: "pycutest.CUTEstProblem") -> torch.Tensor:
    return _CUTEstObj.apply(x, problem) # type:ignore

class CUTEst(Benchmark):
    """pycytest wrapper"""
    def __init__(self, problem: "pycutest.CUTEstProblem | str", dtype=torch.float32):
        super().__init__()

        if isinstance(problem, str):
            import pycutest
            problem = pycutest.import_problem(problem)

        self.problem = problem

        x = torch.tensor(problem.x0, dtype=dtype)
        self.x = torch.nn.Parameter(x)

    def get_loss(self):
        return cutest_obj(self.x, self.problem)
