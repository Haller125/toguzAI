import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import pandas as pd
import numpy as np


class ChunkedNNUEDataset(Dataset):
    def __init__(self, file_path, chunk_size=1000000):
        self.file_path = file_path
        self.chunk_size = chunk_size

        # Read the entire Parquet file
        self.data = pq.read_table(file_path)
        self.total_rows = len(self.data)

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        # Convert PyArrow record to pandas Series
        row = self.data[idx].to_pandas().iloc[0]

        white_features = torch.tensor(row['white_features'], dtype=torch.int8)
        black_features = torch.tensor(row['black_features'], dtype=torch.int8)
        stm = torch.tensor(row['stm'], dtype=torch.int8)
        eval = torch.tensor(row['eval'], dtype=torch.float32)

        return white_features, black_features, stm, eval

def create_dataloader(file_path, batch_size=512, shuffle=True, num_workers=4, chunk_size=100000):
    dataset = ChunkedNNUEDataset(file_path, chunk_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)