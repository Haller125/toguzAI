import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Assuming the last column is the target and the rest are features
        features = torch.tensor(row[:-1].values, dtype=torch.float32)
        target = torch.tensor(row[-1], dtype=torch.long)
        return features, target


class DSLoader:
    def __init__(self, csv_file, batch_size=32):
        self.dataset = CSVDataset(csv_file)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        return self.loader.dataset[idx]

    def __repr__(self):
        return f"DSLoader with {len(self.loader)} batches"
