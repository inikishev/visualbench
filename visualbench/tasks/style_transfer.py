
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from ..benchmark import Benchmark
from ..utils import normalize, to_HW3, znormalize


class VGG(nn.Module):
    def __init__(self, content_layers, style_layers):
        super().__init__()
        self.content_features = content_layers
        self.style_features = style_layers
        from torchvision import models
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

def _grams_style_loss(gen, style):
    b, c, h, w = gen.shape
    G = torch.matmul(gen.view(c, h * w), gen.view(c, h * w).T) # Gram matrix of generated
    A = torch.matmul(style.view(c, h * w), style.view(c, h * w).T) # Gram matrix of style
    return torch.mean((G - A)**2) / (c * h * w)
    #return criterion(G, A) / ((h*w)**2)


def _vgg_normalize(x):
    from torchvision.transforms import v2
    return v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(znormalize(x))

class StyleTransfer(Benchmark):
    """VGG style transfer.

    Renders:
        style transferred image.
    """
    def __init__(
        self,
        content: Any,
        style: Any,
        image_size = 128,
        content_loss = F.mse_loss,
        style_loss = _grams_style_loss,
        content_layers = (21,),
        content_weights = (1.,),
        style_layers = (1, 6, 11, 19, 28),
        style_weights = (200,200,200,200,200),
        use_vgg_norm = True,
    ):
        super().__init__()
        #
        from torchvision.transforms import v2
        if use_vgg_norm: content = _vgg_normalize(to_HW3(content).float().moveaxis(-1,0))
        else: content = znormalize(to_HW3(content)).float().moveaxis(-1,0)
        self.content = torch.nn.Buffer(v2.Resize((image_size, image_size))(content).unsqueeze(0).contiguous())

        #style = znormalize(_make_float_hw3_tensor(style)).moveaxis(-1,0)
        if use_vgg_norm: style = _vgg_normalize(to_HW3(style).moveaxis(-1,0))
        else: style = znormalize(to_HW3(style)).moveaxis(-1,0)
        self.style = torch.nn.Buffer(v2.Resize((image_size, image_size))(style).unsqueeze(0).contiguous())

        self.content_loss = content_loss
        self.style_loss = style_loss

        self.generated = torch.nn.Parameter(self.content.clone().requires_grad_(True).contiguous())

        if isinstance(content_layers, int): content_layers = (content_layers, )
        self.vgg = VGG(content_layers=content_layers, style_layers=style_layers).requires_grad_(False)
        for param in self.vgg.parameters(): param.requires_grad_(False)

        content_features, _ = self.vgg(self.content) # Content features from layer 19
        for i,f in enumerate(content_features):
            self.register_buffer(f'content_features_{i}', f)

        _, style_features = self.vgg(self.style)
        for i,f in enumerate(style_features):
            self.register_buffer(f'style_feature_{i}', f)

        self.style_weights = style_weights
        self.content_weights = content_weights

        self.add_reference_image('content', content, to_uint8=True)
        self.add_reference_image('style', style, to_uint8=True)

        self._show_titles_on_video = False

    def get_loss(self):
        self.vgg.eval()
        content, style = self.vgg(self.generated)

        content_loss = cast(torch.Tensor, 0)
        for i, (f, w) in enumerate(zip(content, self.content_weights)): # Layers 19
            content_loss += self.content_loss(f, getattr(self, f'content_features_{i}')) * w

        style_loss = 0
        for i, (f, w) in enumerate(zip(style, self.style_weights)): # Layers 19
            style_loss += self.style_loss(f, getattr(self, f'style_feature_{i}')) * w

        if self._make_images:
            self.log_image('generated', self.generated, to_uint8=True, log_difference=True)


        return content_loss + style_loss

