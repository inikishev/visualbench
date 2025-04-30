from collections.abc import Iterable, Sequence
from typing import Literal

import torch


def jacobian(input: Sequence[torch.Tensor], wrt: Sequence[torch.Tensor], create_graph=False):
    flat_input = torch.cat([i.reshape(-1) for i in input])
    return torch.autograd.grad(
        flat_input,
        wrt,
        torch.eye(len(flat_input), device=input[0].device, dtype=input[0].dtype),
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
        is_grads_batched=True,
    )


def make_newton_loss(loss_fn, tik_l: float | Literal['eig'] = 1e-2, use_torch_func = False):

    class NewtonLoss(torch.autograd.Function):

        @staticmethod
        def forward(ctx, preds: torch.Tensor, targets: torch.Tensor):
            with torch.enable_grad():
                # necessary to flatten preds FIRST so they are part of the graph
                preds_flat = preds.ravel()
                value = loss_fn(preds_flat.view_as(preds), targets)

                # caluclate gradient and hessian
                if use_torch_func:
                    H: torch.Tensor = torch.func.hessian(loss_fn, 0)(preds_flat, targets) # pyright:ignore[reportAssignmentType]
                    g = torch.autograd.grad(value, preds)[0]

                else:
                    g = torch.autograd.grad(value, preds_flat, create_graph=True)[0]
                    H: torch.Tensor = jacobian([g], [preds_flat])[0]

            # apply regularization
            if tik_l == 'eig':
                reg = torch.linalg.eigvalsh(H).neg().clip(min=0).max()
            else:
                reg = tik_l

            if reg != 0:
                H.add_(torch.eye(H.size(0), device=H.device, dtype=H.dtype).mul_(reg))

            # newton step
            newton_step, success = torch.linalg.solve_ex(H, g)
            ctx.save_for_backward(newton_step.view_as(preds))

            return value

        @staticmethod
        def backward(ctx, *grad_outputs):
            newton_step = ctx.saved_tensors[0] # inputs to loss
            return newton_step, None

    return NewtonLoss.apply