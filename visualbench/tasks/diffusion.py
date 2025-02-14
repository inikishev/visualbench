import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from ..benchmark import Benchmark
from ..dataset_tools import make_dataset_from_tensor
from ..utils import CUDA_IF_AVAILABLE
from .mnist1d import get_mnist1d


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def sinusoidal_embedding(t, dim):
    # Sinusoidal positional embeddings
    half_dim = dim // 2
    embeddings = np.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
    embeddings = t[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings


def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """Forward diffusion process (noisy sample)"""
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, noise=None,  sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None):
    """Loss function for training"""
    noise = torch.randn_like(x_start) if noise is None else noise
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod)
    noise_pred = denoise_model(x_noisy, t)
    loss = F.mse_loss(noise_pred.squeeze(), noise) # Simple MSE loss
    return loss

def extract(a, t, x_shape):
    """Extract coefficients from diffusion schedules"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)



class LinearDiffusion(nn.Module):
    def __init__(self, channels, image_size, timesteps, base_dim=32):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.timesteps = timesteps
        self.base_dim = base_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(timesteps, base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim),
        )

        self.down_blocks = nn.ModuleList([
            nn.Linear(image_size, base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim * 2),
            nn.ReLU(),
        ])

        self.up_blocks = nn.ModuleList([
            nn.Linear(base_dim * 2 + base_dim, base_dim * 2), # + base_dim for time embedding
            nn.ReLU(),
            nn.Linear(base_dim * 2, base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, image_size),
        ])

    def forward(self, x, time):
        # Time embedding
        time_emb = self.time_mlp(sinusoidal_embedding(time, self.timesteps))

        # Downsample path
        h = x.squeeze(1) # Remove channel dim for MLP
        for block in self.down_blocks:
            h = block(h)

        # Bottleneck (optional, can remove for even simpler model)
        # h = nn.Linear(base_dim * 2, base_dim * 2)(h)
        # h = nn.ReLU()(h)

        # Upsample path
        for block in self.up_blocks:
            if isinstance(block, nn.Linear) and block.in_features == self.up_blocks[0].in_features: # Inject time embedding in first layer
                h = block(torch.cat((h, time_emb), dim=-1))
            else:
                h = block(h)


        return h.unsqueeze(1) # Add channel dimension back

class UNet(nn.Module):
    def __init__(self, channels, image_size, timesteps, base_dim=32):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.timesteps = timesteps
        self.base_dim = base_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(timesteps, base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim),
        )

        # --- Downsampling ---
        self.down_conv1 = nn.Conv1d(channels, base_dim, kernel_size=3, padding=1)
        self.down_relu1 = nn.ReLU()
        self.down_conv2 = nn.Conv1d(base_dim, base_dim * 2, kernel_size=3, padding=1, stride=2) # Stride for downsample
        self.down_relu2 = nn.ReLU()

        # --- Bottleneck ---
        self.bottleneck_conv = nn.Conv1d(base_dim * 2, base_dim * 2, kernel_size=3, padding=1)
        self.bottleneck_relu = nn.ReLU()
        self.bottleneck_conv2 = nn.Conv1d(base_dim * 2, base_dim * 2, kernel_size=3, padding=1)
        self.bottleneck_relu2 = nn.ReLU()

        # --- Upsampling ---
        self.up_tconv1 = nn.ConvTranspose1d(base_dim * 2, base_dim, kernel_size=2, stride=2) # Transposed conv for upsample
        self.up_relu1 = nn.ReLU()
        self.up_conv1 = nn.Conv1d(base_dim * 2, base_dim, kernel_size=3, padding=1) # *2 because of skip connection
        self.up_relu2 = nn.ReLU()
        self.up_conv2 = nn.Conv1d(base_dim, channels, kernel_size=3, padding=1) # To output channels


    def forward(self, x:torch.Tensor, time):
        x = x.unsqueeze(1)
        # Time embedding
        time_emb = self.time_mlp(sinusoidal_embedding(time, self.timesteps))
        time_emb = time_emb.unsqueeze(-1) # [B, base_dim, 1] to add to feature maps

        # --- Downsampling ---
        h1 = self.down_relu1(self.down_conv1(x)) # [B, base_dim, L]
        h2 = self.down_relu2(self.down_conv2(h1)) # [B, base_dim*2, L/2]
        # --- Bottleneck ---
        h_bottleneck = self.bottleneck_relu(self.bottleneck_conv(h2))
        h_bottleneck = self.bottleneck_relu2(self.bottleneck_conv2(h_bottleneck))
        # --- Upsampling ---
        h3 = self.up_tconv1(h_bottleneck) # [B, base_dim, L] (upsample)

        # --- Skip Connection & Upsample Block ---
        h_skip = torch.cat((h3, h1), dim=1) # Skip connection from downsample path
        h4 = self.up_relu1(self.up_conv1(h_skip)) # [B, base_dim, L]
        output = self.up_conv2(h4) # [B, channels, L]


        return output



class DiffusionMNIST1D(Benchmark):
    def __init__(self, base_dim = 32, batch_size=128, timesteps = 200, num_samples = 5000, seed = 0, model_cls = UNet, device = CUDA_IF_AVAILABLE):
        super().__init__()

        train, test = get_mnist1d(
            num_samples=int(num_samples * 1/0.9),
            train_split=0.9,
            seed=seed,
            device=device,
        )
        train_data, _ = make_dataset_from_tensor(
            dataset = train,
            batch_size = batch_size,
            seed = seed,
        )

        super().__init__(
            train_data = train_data,
            seed = seed
        )

        self.model = model_cls(1, 40, timesteps, base_dim=base_dim)
        self.model.train()
        self.timesteps = timesteps

        betas = linear_beta_schedule(timesteps=timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0) # type:ignore
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.nn.Buffer(torch.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = torch.nn.Buffer(torch.sqrt(1. - alphas_cumprod))
        # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.to(device)

    def reset(self, model):
        super().reset()
        self.model = model.to(self.device)
        self.train()

    def get_loss(self):
        images = self.batch[0]
        t = torch.randint(0, self.timesteps, (images.shape[0],), device=self.sqrt_alphas_cumprod.device).long() # Random timesteps

        loss = p_losses(
            self.model,
            images,
            t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod
        )

        return loss


# bench = DiffusionMNIST1D()
# bench.run(torch.optim.LBFGS(bench.parameters(), 1), 1000)