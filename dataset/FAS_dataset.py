import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
import json


class FASDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None, is_train=True, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        with open(csv_file, 'r', encoding='utf-8') as f:
            self.data = json.loads(f.read())
        self.transform = transform
        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99
        self.img_list = []
        self.img_label = []
        for k, x in self.data.items():
            self.img_list.append(k)
            label = x[40]
            if label >= 1 and label <= 3:
                label = 1
            elif label >= 4 and label <= 6:
                label = 2
            elif label >= 7 and label <= 8:
                label = 3
            else:
                label = 4
            t_label=np.zeros(5)
            t_label[label]=1.0   
            self.img_label.append(t_label)
            # self.img_label.append(np.zeros(2))    

    def __getitem__(self, index):

        img_name = os.path.join(self.root_dir, self.img_list[index])

        img = Image.open(img_name)
        label = self.img_label[index].astype(np.float32)
        # label = np.expand_dims(label, axis=0)

        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)

        return img1, img2, label

    def __len__(self):
        return len(self.img_label)
