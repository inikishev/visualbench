"""SSM models from https://github.com/lixilinx/Fully-Trainable-SSM/tree/main"""
from collections.abc import Iterable
import numpy as np
from torch import nn
import torch

# from https://github.com/lixilinx/Fully-Trainable-SSM/blob/main/state_space_models.pyhttps://github.com/lixilinx/Fully-Trainable-SSM/blob/main/state_space_models.py
########################### Complex State 1D SSM #################################
def fast_conv_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Inputs:
        a: tensor with shape [len1, n] representing len1 n x n diagonal matrices.
        b: another torch tensor with shape [batch, len2, n].

    Outputs:
        c = conv(a, b) with shape [batch, len1 + len2 - 1, n].
    """
    len1, _ = a.shape
    _, len2, _ = b.shape
    T = 2**int(np.ceil(np.log2(len1 + len2 - 1)))
    A = torch.fft.fft(a, n=T, dim=0) # pylint:disable=not-callable
    B = torch.fft.fft(b, n=T, dim=1) # pylint:disable=not-callable
    C = A * B
    c = torch.fft.ifft(C, n=T, dim=1) # pylint:disable=not-callable
    return c[:, :(len1 + len2 - 1)]

# from https://github.com/lixilinx/Fully-Trainable-SSM/blob/main/state_space_models.pyhttps://github.com/lixilinx/Fully-Trainable-SSM/blob/main/state_space_models.py
class _ComplexStateSpaceModel(torch.nn.Module):
    """
    It returns a fully trainable complex state SSM defined as:

        x_t = A @ x_{t-1} + B @ u_t,
        y_t = C @ x_t + D @ u_t + b,

    where:

        u_t, 1 <= t <= T, are the sequences of (real) inputs,
        x_t, 1 <= t <= T, are the sequence of (complex) states,
        y_t, 1 <= t <= T, are the sequence of (real) outputs,
        matrices A (diagonal complex), B (complex) and C (complex) are mandatory,
        matrix D (real) and bias b (real) are optional,
        and x_0 is the initial (complex) state.

    Note that we use the tranposed version of these equations in the Python code
    by following the 'row major' convention.
    """
    def __init__(self, input_size: int, state_size: int, output_size: int,
                 has_matrixD: bool=False, has_bias: bool=False,
                 resample_up: int=1, resample_down: int=1,
                 enforce_stability: bool=False) -> None:
        """
        Inputs:
            input_size, state_size and output_size: sizes of u, x, and y, respectively.
            has_matrixD: matrix D is None if setting to False, otherwise not.
            has_bias: bias b is None if setting to False, otherwise not.
            resample_up, resample_down: resample the sequence with ratio resample_up/resample_down.
            enforce_stability: set to True to enforce the poles to stay inside unit disc,
                        otherwise poles can be outside of unit disc (unstable for long sequences).
        """
        super(_ComplexStateSpaceModel, self).__init__()
        A = 2*torch.pi*torch.rand(state_size)
        A = torch.complex(torch.cos(A), torch.sin(A))
        self.enforce_stability = enforce_stability
        if enforce_stability:
            self.A = torch.nn.Parameter(10 * A)
        else:
            self.A = torch.nn.Parameter(A)
        self.B = torch.nn.Parameter(
            torch.randn(input_size, state_size, dtype=torch.complex64)/(state_size + input_size)**0.5)
        self.C = torch.nn.Parameter(
            torch.randn(state_size, output_size, dtype=torch.complex64)/(state_size + output_size)**0.5)
        self.has_matrixD = has_matrixD
        if has_matrixD:
            self.D = torch.nn.Parameter(torch.zeros(input_size, output_size))
        self.has_bias = has_bias
        if has_bias:
            self.b = torch.nn.Parameter(torch.zeros(output_size))
        self.resample_up = resample_up
        self.resample_down = resample_down


    def forward(self, u: torch.Tensor, x0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            u: the real input tensor with shape [batch, length, input_size].
            x0: the complex initial state with shape [batch, state_size].

        Outputs:
            y: the real output tensor with shape [batch, resample_up*length//resample_down, output_size].
            x0: the complex final state with shape [batch, state_size].
        """
        if self.resample_up > 1:
            u = u.repeat_interleave(self.resample_up, 1)
        _, length, _ = u.shape
        if self.enforce_stability:
            A = self.A * torch.rsqrt(self.A * self.A.conj() + 1) # pull inside the unit disc
        else:
            A = self.A
        Aps = torch.pow(A, torch.arange(length + 1, device=self.A.device)[:, None])
        uB = u.to(torch.complex64) @ self.B # tranpose of math eq B*u
        x = fast_conv_complex(Aps[:-1], uB)[:, :length]
        if x0 is not None:
            x = x + Aps[1:] * x0[:,None,:]
        if self.resample_down > 1:
            x = x[:, self.resample_down-1::self.resample_down]
            if self.has_matrixD:
                u = u[:, self.resample_down-1::self.resample_down]
        y = torch.real(x @ self.C) # tranpose of math eq C * x
        if self.has_matrixD:
            y = y + u @ self.D # transpose of math eq D * u
        if self.has_bias:
            y = y + self.b
        return (y, x[:, -1])

# from https://github.com/lixilinx/Fully-Trainable-SSM/blob/main/demo.py
class SSMNet(torch.nn.Module):
    """
    has_bias: bias b is None if setting to False, otherwise not.
    resample_up, resample_down: resample the sequence with ratio resample_up/resample_down.
    enforce_stability:
        set to True to enforce the poles to stay inside unit disc,
        otherwise poles can be outside of unit disc (unstable for long sequences).
    """
    def __init__(self, in_channels, out_channels, hidden: int | Iterable[int] | None = (16, 128), has_bias=False, resample_up=1, resample_down=4, enforce_stability=False):
        super().__init__()
        self.in_channels = in_channels

        if isinstance(hidden, int): hidden = [hidden]
        if hidden is None: hidden = []
        channels = [in_channels] + list(hidden) + [out_channels]

        layers = []
        for i,o in zip(channels[:-2], channels[1:-1]):
            layers.append(_ComplexStateSpaceModel(i, o, o, has_matrixD=False, has_bias=has_bias, resample_up=resample_up, resample_down=resample_down, enforce_stability=enforce_stability))

        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-2], channels[-1])

    def forward(self, x):
        if x.ndim == 2 and self.in_channels == 1: x = x.unsqueeze(2)

        for i, ssm in enumerate(self.layers):
            x, _ = ssm(x)

            if i == len(self.layers) - 1:
                x = x[:, -1]

            x = x * torch.rsqrt(1 + x*x)

        x = self.head(x)
        return x

