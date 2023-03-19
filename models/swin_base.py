import torch
from torch import nn
from torchvision import models


class Swin_base(nn.Module):

    def __init__(self, pretrained=True, device='cpu'):
        super(Swin_base, self).__init__()
        if pretrained:
            base_model = models.swin_b(
                weights=models.Swin_B_Weights.IMAGENET1K_V1)
        else:
            base_model = models.swin_b()
        self.nets = nn.Sequential(*(list(base_model.children())[:-2]))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.model_out_feature = 1024
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.nets(x).permute(0, 3, 1, 2)
        out = self.avg_pool(x)
        return out
