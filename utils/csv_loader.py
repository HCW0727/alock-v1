import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class csv_dataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
def create_csv_loader(x_data:np.ndarray,
                  y_data:np.ndarray,
                  batch_size:int = 16,
                  shuffle=True):
    dataset = csv_dataset(x_data,y_data)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return loader