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


def make_newton_loss(loss_fn, tik_l: float = 1e-2):

    class NewtonLoss(torch.autograd.Function):

        @staticmethod
        def forward(ctx, preds: torch.Tensor, targets: torch.Tensor):
            with torch.enable_grad():
                # necessary to flatten preds FIRST so they are part of the graph
                preds_flat = preds.ravel()
                value = loss_fn(preds_flat.view_as(preds), targets)

                # caluclate gradient and hessian
                g = torch.autograd.grad(value, preds_flat, create_graph=True)[0]
                H: torch.Tensor = jacobian([g], [preds_flat])[0]

            # apply regularization
            if tik_l != 0:
                H.add_(torch.eye(H.size(0), device=H.device, dtype=H.dtype).mul_(tik_l))

            # newton step
            newton_step, success = torch.linalg.solve_ex(H, g)
            ctx.save_for_backward(newton_step.view_as(preds))

            return value

        @staticmethod
        def backward(ctx, *grad_outputs):
            newton_step = ctx.saved_tensors[0] # inputs to loss
            return newton_step, None

    return NewtonLoss.apply


def make_batched_newton_loss(loss_fn, tik_l: float = 1e-2):

    class BatchedNewtonLoss(torch.autograd.Function):

        @staticmethod
        def forward(ctx, preds: torch.Tensor, targets: torch.Tensor):
            with torch.enable_grad():
                # necessary to flatten and unbind preds FIRST and then re-stack so they are part of the graph
                preds_flat = preds.view(preds.size(0), -1)
                samples = preds_flat.unbind(0)
                value = loss_fn(torch.stack(samples).view_as(preds), targets)

                # caluclate gradient and hessian
                per_sample_H = []
                per_sample_g = []
                for sample in samples:
                    g = torch.autograd.grad(value, sample, create_graph=True,)[0]
                    H: torch.Tensor = jacobian([g], [sample])[0]
                    per_sample_g.append(g)
                    per_sample_H.append(H)

            # apply regularization
            if tik_l != 0:
                I = torch.eye(per_sample_H[0].size(0), device=per_sample_H[0].device, dtype=per_sample_H[0].dtype).mul_(tik_l)
                for H in per_sample_H: H.add_(I)

            # newton step
            H = torch.stack(per_sample_H)
            g = torch.stack(per_sample_g)
            newton_step, success = torch.linalg.solve_ex(H, g)
            ctx.save_for_backward(newton_step.view_as(preds))

            return value

        @staticmethod
        def backward(ctx, *grad_outputs):
            newton_step = ctx.saved_tensors[0] # inputs to loss
            return newton_step, None

    return BatchedNewtonLoss.apply


if __name__ == "__main__":
    dice = torch.nn.MSELoss()

    input = torch.randn(32,100, requires_grad=True, device='cuda')
    target = (torch.rand(32,100, device='cuda') > 0.5).float()
    opt = torch.optim.SGD([input], 1)

    print('normal dice')
    for i in range(100):
        loss = dice(input, target)
        mse = (input-target).pow(2).mean()
        print(f'{i}, {loss = }, {mse = }')
        opt.zero_grad()
        loss.backward()
        opt.step()


    newton_dice = make_batched_newton_loss(torch.nn.MSELoss(), tik_l=1e-4)

    input = torch.randn(32,100, requires_grad=True, device='cuda')
    target = (torch.rand(32,100, device='cuda') > 0.5).float()
    opt = torch.optim.SGD([input], 1)

    print('newton dice')
    for i in range(100):
        loss = newton_dice(input, target)
        mse = (input-target).pow(2).mean()
        print(f'{i}, {loss = }, {mse = }')
        opt.zero_grad()
        loss.backward()
        opt.step()

