import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from myai.transforms import normalize, znormalize
from torchvision import models
from torchvision.transforms import v2

from ..benchmark import Benchmark
from .._utils import _make_float_hw3_tensor


# --- VGG19 Feature Extraction Model ---
class VGG(nn.Module):
    def __init__(self, content_layers, style_layers):
        super(VGG, self).__init__()
        self.content_features = content_layers
        self.style_features = style_layers
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29] # Up to layer 28 (index 28 is layer 29 actually) # type:ignore

    def forward(self, x):
        # print(f'{x.device = }')
        content = []
        style = []

        for layer_num, layer in enumerate(self.vgg):
            x = layer(x)

            if layer_num in self.content_features: content.append(x)
            if layer_num in self.style_features: style.append(x)

        return content, style

def grams_style_loss(gen_features, style_features):
    batch_size, channel, height, width = gen_features.shape
    G = torch.matmul(gen_features.view(channel, height * width), gen_features.view(channel, height * width).T) # Gram matrix of generated
    A = torch.matmul(style_features.view(channel, height * width), style_features.view(channel, height * width).T) # Gram matrix of style
    return torch.mean((G - A)**2) / (channel * height * width)


def _vgg_normalize(x):
    return v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(normalize(x))

class StyleTransfer(Benchmark):
    """VGG style transfer"""
    def __init__(
        self,
        content: Any,
        style: Any,
        image_size = 128,
        content_loss = F.mse_loss,
        style_loss = grams_style_loss,
        content_layers = (19,),
        content_weights = (1.,),
        style_layers = (0, 5, 10, 19, 28),
        style_weights = (200,200,200,200,200),
        make_images = True,
        use_vgg_norm = False,
    ):
        super().__init__(log_params=True)
        #
        if use_vgg_norm: content = _vgg_normalize(_make_float_hw3_tensor(content).moveaxis(-1,0))
        else: content = znormalize(_make_float_hw3_tensor(content)).moveaxis(-1,0)
        self.content = torch.nn.Buffer(v2.Resize((image_size, image_size))(content).unsqueeze(0).contiguous())

        #style = znormalize(_make_float_hw3_tensor(style)).moveaxis(-1,0)
        if use_vgg_norm: style = _vgg_normalize(_make_float_hw3_tensor(style).moveaxis(-1,0))
        else: style = znormalize(_make_float_hw3_tensor(style)).moveaxis(-1,0)
        self.style = torch.nn.Buffer(v2.Resize((image_size, image_size))(style).unsqueeze(0).contiguous())

        self.content_loss = content_loss
        self.style_loss = style_loss

        self.generated = torch.nn.Parameter(self.content.clone().requires_grad_(True).contiguous())

        self._ignore_vgg = VGG(content_layers=content_layers, style_layers=style_layers) # parameters wont return it
        for param in self._ignore_vgg.parameters(): param.requires_grad_(False)

        content_features, _ = self._ignore_vgg(self.content) # Content features from layer 19
        for i,f in enumerate(content_features):
            self.register_buffer(f'content_features_{i}', f)

        _, style_features = self._ignore_vgg(self.style)
        for i,f in enumerate(style_features):
            self.register_buffer(f'style_feature_{i}', f)

        self.style_weights = style_weights
        self.content_weights = content_weights

        self._make_images = make_images
        if make_images:
            self.set_display_best('image generated')
            self.add_reference_image('content', content, to_uint8=True)
            self.add_reference_image('style', style, to_uint8=True)

    def get_loss(self):
        self._ignore_vgg.eval()
        content, style = self._ignore_vgg(self.generated)

        content_loss = 0
        for i, (f, w) in enumerate(zip(content, self.content_weights)): # Layers 19
            content_loss += self.content_loss(f, getattr(self, f'content_features_{i}')) * w

        style_loss = 0
        for i, (f, w) in enumerate(zip(style, self.style_weights)): # Layers 19
            style_loss += self.style_loss(f, getattr(self, f'style_feature_{i}')) * w

        if self._make_images:
            self.log('image', self.generated, log_test=False, to_uint8=True)
            self.log_difference('image update', self.generated, to_uint8=True)

        return content_loss + style_loss

