from game_logic import toguz
from dataset_generation import CONSTANTS
from dataset_generation.dataset import create_dataloader
from simple_nnue.model import NNUE, train_nnue

import pandas as pd
import ast
import torch


def main():
    board = toguz.Board([1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18], [-1,-1], [0,0])
    mmx = toguz.MinimaxH(CONSTANTS.heuristic_weights)

    print(board.rotate().toString())
    board.atsyrauFunction(False)
    board.rotate()
    print(board.rotate().toString())

    return 0

def check_dataset():
    file_path = "dataset_generation/dataset.parquet"

    df = pd.read_parquet(file_path)

    print("DF loaded")
    print(df.head())

    meta_epoch = 10

    nnue = NNUE()

    for i in range(meta_epoch):
        df1 = pd.concat([df.iloc[:len(df) // 2].sample(frac=0.1), df.iloc[len(df) // 2:].sample(frac=0.1)],
                        ignore_index=True).reset_index()

        train_loader = create_dataloader(df1, batch_size=1024)
        val_loader = create_dataloader(df.sample(frac=0.1).reset_index())

        nnue = train_nnue(nnue, train_loader, val_loader, num_epochs=50)

        torch.save(nnue.state_dict(), f"simple_nnue/models/nnue_epoch_{i}.pt")

if __name__ == '__main__':
    check_dataset()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
