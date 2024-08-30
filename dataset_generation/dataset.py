import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class NNUEDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        white_features = torch.tensor(row['white_features'], dtype=torch.float32)
        black_features = torch.tensor(row['black_features'], dtype=torch.float32)
        stm = torch.tensor(row['stm'], dtype=torch.float32)
        eval = torch.tensor(row['eval'], dtype=torch.float32)

        return white_features, black_features, stm, eval


def create_dataloader(dataframe, batch_size=128, shuffle=True, num_workers=4):
    dataset = NNUEDataset(dataframe)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
