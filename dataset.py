import numpy as np
import torch
import scipy.io
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images, masks=None, train=True):
        self.images = images
        self.masks = masks
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.train:
            return img, self.masks[idx]
        return img

def load_data(data_path, total_num, density=1000, num_cell=32):
    x, y = [], []
    for i in range(total_num):
        data = scipy.io.loadmat(data_path.format(i))
        m = data['m'] / density
        d = np.nan_to_num(data['d'])
        x.append(d.reshape(1, num_cell, num_cell))
        y.append(m.reshape(16, num_cell, num_cell))
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
