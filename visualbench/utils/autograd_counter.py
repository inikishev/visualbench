import torch

class AutogradCounter(torch.autograd.Function):
    """counts autograd calls except it doesn't work"""
    @staticmethod
    def forward(x, counter, train):
        if train: counter.num_forwards += 1
        return x.view_as(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, counter, train = inputs
        ctx.counter = counter
        ctx.train = train

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.train
        ctx.counter.num_backwards += 1
        return grad_output, None, None

    @staticmethod
    def jvp(ctx, tangent_x, _tangent_counter, _tangent_train):
        assert ctx.train
        ctx.counter.num_jvps += 1
        return tangent_x.view_as(tangent_x)

    @staticmethod
    def vmap(info, in_dims, x, counter, train):
        return x.view_as(x), in_dims[0]


def autograd_counter(x: torch.Tensor, counter, train:bool) -> torch.Tensor:
    return AutogradCounter.apply(x, counter, train)