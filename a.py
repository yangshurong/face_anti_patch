import torch
import argparse

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os



train_ds = datasets.FashionMNIST(
                                root = './data',
                                train=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                                download=True)