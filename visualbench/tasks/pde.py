from collections.abc import Callable

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..benchmark import Benchmark


class SinAct(nn.Module):
    def forward(self, x): return torch.sin(x)

# https://github.com/AdityaLab/pinnsformer/blob/main/demo/navier_stokes/naiver_stoke_pinnsformer.ipynb
class WaveAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

# https://github.com/AdityaLab/pinnsformer/blob/main/demo/navier_stokes/naiver_stoke_qres.ipynb
class QResBblock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.H1 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.H2 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.act = nn.Sigmoid()
    def forward(self, x):
        x1 = self.H1(x)
        x2 = self.H2(x)
        return self.act(x1*x2 + x1)

@torch.no_grad
def _initialize_(net: nn.Module):
    generator = torch.Generator().manual_seed(0)
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight, generator=generator)
            layer.bias.fill_(0.01)

class _PINNs(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=256, n_hidden=5, act1:Callable=nn.Tanh, acth:Callable = nn.Tanh):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(act1())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(acth())
        layers.append(nn.Linear(hidden_size, out_size))
        self.net = nn.Sequential(*layers)

        _initialize_(self)

    def forward(self, x):
        return self.net(x)


class _FLS(_PINNs):
    def __init__(self, in_size, out_size,hidden_size=256, n_hidden=5, act1:Callable=WaveAct, acth:Callable = nn.Tanh):
        super().__init__(in_size=in_size, hidden_size=hidden_size, out_size=out_size, n_hidden=n_hidden, act1=act1, acth=acth)

# https://github.com/AdityaLab/pinnsformer/blob/main/demo/navier_stokes/naiver_stoke_qres.ipynb
class _QRes(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=256, n_hidden=5):
        super().__init__()
        self.N = n_hidden-1
        self.inlayer = QResBblock(in_size, hidden_size)

        layers = []
        for _ in range(n_hidden - 1):
            layers.append(QResBblock(hidden_size, hidden_size))
        self.hidden = nn.Sequential(*layers)

        self.outlayer = nn.Linear(in_features=hidden_size, out_features=out_size)
        _initialize_(self)

    def forward(self, x):
        x = self.inlayer(x)
        x = self.hidden(x)
        x = self.outlayer(x)
        return x


class WavePINN(Benchmark):
    """Solving a wave PDE with a PINN.

    Renders:
        predicted solution, true solution and the error.

    Args:
        net (nn.Module): model, input size is 2, output size is 1.
        c (float): wave speed.
        n_pde (int): number of collocation points for the PDE residual.
        n_ic (int): number of points for the initial condition loss.
        n_bc (int): number of points for the boundary condition loss.
        render_res (int): resolution for rendering.
        criterion (Callable): loss func.
    """
    PINNs = _PINNs
    QRes = _QRes
    FLS = _FLS
    def __init__(self, net, c=2.0, n_pde=1024, n_ic=512, n_bc=512, render_res=100, criterion = F.mse_loss):
        super().__init__()
        rng = self.rng.torch()
        self.c = c
        self.c_squared = c**2
        self.render_res = render_res

        # net
        self.net = net

        # 1. PDE points (domain interior)
        pde_t = torch.rand(n_pde, 1, generator=rng) # t in (0, 1)
        pde_x = torch.rand(n_pde, 1, generator=rng) # x in (0, 1)
        self.pde_t = nn.Buffer(pde_t)
        self.pde_x = nn.Buffer(pde_x)

        # 2. initial Condition (IC) points (t=0)
        ic_x = torch.rand(n_ic, 1, generator=rng)
        ic_t = torch.zeros_like(ic_x)
        u_ic_true = torch.sin(np.pi * ic_x) + 0.5 * torch.sin(4 * np.pi * ic_x)
        ut_ic_true = torch.zeros_like(ic_x)

        self.ic_t = nn.Buffer(ic_t)
        self.ic_x = nn.Buffer(ic_x)
        self.u_ic_true = nn.Buffer(u_ic_true)
        self.ut_ic_true = nn.Buffer(ut_ic_true)

        # 3. boundary Condition (BC) points (x=0 and x=1)
        bc_t = torch.rand(n_bc, 1, generator=rng)
        bc_x0 = torch.zeros_like(bc_t)
        bc_x1 = torch.ones_like(bc_t)
        u_bc_true = torch.zeros_like(bc_t)

        self.bc_t = nn.Buffer(bc_t)
        self.bc_x0 = nn.Buffer(bc_x0)
        self.bc_x1 = nn.Buffer(bc_x1)
        self.u_bc_true = nn.Buffer(u_bc_true)

        self.criterion = criterion

        # grid for rendering
        t_render_ax = torch.linspace(0, 1, render_res)
        x_render_ax = torch.linspace(0, 1, render_res)
        t_grid, x_grid = torch.meshgrid(t_render_ax, x_render_ax, indexing='ij')
        self.t_grid = nn.Buffer(t_grid)
        self.x_grid = nn.Buffer(x_grid)
        render_points = torch.stack([t_grid.flatten(), x_grid.flatten()], dim=1)
        self.render_points = nn.Buffer(render_points)
        self.vmin = -1.5
        self.vmax = 1.5
        self.vmax_error = 0.5

        u_exact_grid = (torch.sin(torch.pi * self.x_grid) * torch.cos(self.c * np.pi * self.t_grid) +
                        0.5 * torch.sin(4 * np.pi * self.x_grid) * torch.cos(4 * self.c * np.pi * self.t_grid))
        self.u_exact_grid = nn.Buffer(u_exact_grid)
        self.add_reference_image('u exact', self._touin8(self.u_exact_grid), to_uint8=False)

    def _touin8(self, grid_data):
        normalized_grid = 255 * (grid_data - self.vmin) / (self.vmax - self.vmin)
        normalized_grid.clip_(0, 255)

        if isinstance(normalized_grid, torch.Tensor):
            normalized_grid = normalized_grid.detach().cpu().numpy()

        img = normalized_grid.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)[:,:,::-1] # pylint:disable=no-member
        return img


    def get_loss(self):
        # PDE Loss
        t_pde, x_pde = self.pde_t.clone(), self.pde_x.clone()
        t_pde.requires_grad = True
        x_pde.requires_grad = True

        u_pde = self.net(torch.cat([t_pde, x_pde], dim=1))

        grads = torch.autograd.grad(u_pde, [t_pde, x_pde], grad_outputs=torch.ones_like(u_pde), create_graph=True)
        u_t, u_x = grads[0], grads[1]
        u_tt = torch.autograd.grad(u_t, t_pde, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        pde_residual = u_tt - self.c_squared * u_xx
        loss_pde = self.criterion(pde_residual, torch.zeros_like(pde_residual))

        # initial condition loss
        t_ic, x_ic = self.ic_t.clone(), self.ic_x.clone()
        t_ic.requires_grad = True # Required for u_t calculation

        u_ic_pred = self.net(torch.cat([t_ic, x_ic], dim=1))
        u_t_ic_pred = torch.autograd.grad(u_ic_pred, t_ic, grad_outputs=torch.ones_like(u_ic_pred), create_graph=True)[0]

        loss_ic_u = self.criterion(u_ic_pred, self.u_ic_true)
        loss_ic_ut = self.criterion(u_t_ic_pred, self.ut_ic_true)

        # boundary condition loss
        u_bc_pred_0 = self.net(torch.cat([self.bc_t, self.bc_x0], dim=1))
        u_bc_pred_1 = self.net(torch.cat([self.bc_t, self.bc_x1], dim=1))
        loss_bc = self.criterion(u_bc_pred_0, self.u_bc_true) + self.criterion(u_bc_pred_1, self.u_bc_true)

        total_loss = loss_pde + 100*(loss_ic_u + loss_ic_ut + loss_bc)

        if self._make_images:
            self.net.eval()
            with torch.no_grad():
                u_render = self.net(self.render_points)
                u_render_grid = u_render.view(self.render_res, self.render_res)
                self.log_image("u predicted", self._touin8(u_render_grid), to_uint8=False, show_best=True)

                error_grid = torch.abs(u_render_grid - self.u_exact_grid)
                error_grid /= self.vmax_error / 255
                error_grid.clip_(0, 255)
                img = error_grid.detach().cpu().numpy().astype(np.uint8)
                self.log_image("error", cv2.applyColorMap(img, cv2.COLORMAP_JET)[:,:,::-1], to_uint8=False) # pylint:disable=no-member

            self.net.train()

        return total_loss
