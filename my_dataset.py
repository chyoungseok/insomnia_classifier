import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Custom Dataset 생성
class gen_dataset(Dataset):
    def __init__(self, num_classes=2, sampling_rate=34.13, path_load = 'Y:/MESA_0.7.0/ppg_preprocessed_epoch_npy'):
                    
        self.data = torch.from_numpy(np.load(os.path.join(path_load, "ppg_epochs.npy")))
        self.label = torch.from_numpy(np.load(os.path.join(path_load, "label_epochs.npy")))

        self.num_samples = self.data.shape[0]
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        return data, label