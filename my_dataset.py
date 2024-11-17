import os
import numpy as np

import torch
from torch.utils.data import Dataset

# Custom Dataset 생성
class gen_dataset(Dataset):
    def __init__(self, num_classes=2, sampling_rate=34.13, train_or_test='default'):

        if 'train' in train_or_test.lower():
            path_load = 'Y:/MESA_0.7.0/ppg_preprocessed_npy/train'
        elif 'test' in train_or_test.lower():
            path_load = 'Y:/MESA_0.7.0/ppg_preprocessed_npy/test'
        else:
            path_load = 'Y:/MESA_0.7.0/ppg_preprocessed_npy'
                    
        self.data = np.load(os.path.join(path_load, "ppg_ins_hc.npy"))
        self.labels = np.load(os.path.join(path_load, "label_ins_hc.npy"))
        self.mask = np.load(os.path.join(path_load, "mask_ins_hc.npy"))

        self.num_samples = self.data.shape[0]
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate 
        self.train_or_test = train_or_test

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        # PyTorch 텐서로 변환
        return torch.tensor(signal, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)