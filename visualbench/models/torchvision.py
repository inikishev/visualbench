from torch import nn

def pretrained_squeezenet(in_channels: int = 3, out_channels: int = 1000):
    """for 2D inputs. if in_channels isn't 3 or out_channels isn't 1000, some layers will be replaced with untrained"""
    from torchvision import models
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT) # or resnet34, resnet50, etc.

    if in_channels != 3:
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2)

    if out_channels != 1000:
        final_conv = model.classifier[1] = nn.Conv2d(512, out_channels, 1)

        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, 0)

    return model

