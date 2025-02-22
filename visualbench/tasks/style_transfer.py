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
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28'] # Content layer (layer 19), Style layers (layers 0, 5, 10, 28)
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29] # Up to layer 28 (index 28 is layer 29 actually) # type:ignore

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.vgg):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

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
        style_weights = (200,200,200,200,200),
        content_weight = 1.,
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

        self._ignore_vgg = VGG() # parameters wont return it
        for param in self._ignore_vgg.parameters(): param.requires_grad_(False)

        self.content_features_orig = torch.nn.Buffer(self._ignore_vgg(self.content)[3]) # Content features from layer 19
        style_features = self._ignore_vgg(self.style)
        self.n_style_features = len(style_features)
        # self.style_features_orig = torch.nn.Buffer(torch.nested.nested_tensor(style_features)) # nested can be buffer but cant be deepcopied
        for i,sf in enumerate(style_features):
            self.register_buffer(f'style_feature_{i}', sf)
        self.style_weights = style_weights
        self.content_weight = content_weight

        self._make_images = make_images
        if make_images:
            self.set_display_best('image generated')
            self.add_reference_image('content', content, to_uint8=True)
            self.add_reference_image('style', style, to_uint8=True)

    def reset(self):
        super().reset()
        self.generated = torch.nn.Parameter(self.content.clone().requires_grad_(True).contiguous())

    def get_loss(self):
        self._ignore_vgg.eval()
        gen_features = self._ignore_vgg(self.generated)

        content_loss = self.content_loss(gen_features[3], self.content_features_orig) * self.content_weight # Layer 19

        style_loss = 0
        for i in range(self.n_style_features): # Layers 0, 5, 10, 28
            style_loss += self.style_loss(gen_features[i], getattr(self, f'style_feature_{i}')) * self.style_weights[i]

        if self._make_images:
            self.log('image', self.generated, log_test=False, to_uint8=True)
            self.log_difference('image update', self.generated, to_uint8=True)

        return content_loss + style_loss

