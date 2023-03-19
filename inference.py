import torch
import argparse

import torch.nn as nn
import torch.nn.functional as F
from models.resnet18 import FeatureExtractor
from metrics.losses import PatchLoss

from torchvision import datasets, transforms
import numpy as np
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/mnt/f/home/Implementation-patchnet/runs/save/swin_base_best.pth'
model_state = torch.load(model_path)
model = FeatureExtractor(device=device)
model.load_state_dict(model_state['state_dict'])
model.cuda()
loss = PatchLoss().to(device)
loss.load_state_dict(model_state['loss'])
img = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    feature = model(img)
    ans = loss.amsm_loss.fc(feature.squeeze(3).squeeze(2))
    print(torch.argmax(ans, dim=1))
