import torch
from torch import nn
from torchvision import models


class Resnet18(nn.Module):

    def __init__(self, pretrained=True, device='cpu'):
        super(Resnet18, self).__init__()
        base_model = models.resnet18(pretrained=pretrained)
        self.nets = nn.Sequential(*(list(base_model.children())[:-1]))
        self.device = device
        self.model_out_feature = 512

    def forward(self, x):
        x = x.to(self.device)
        return self.nets(x)
