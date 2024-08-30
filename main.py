from game_logic import toguz
from dataset_generation import CONSTANTS
from dataset_generation.dataset import create_dataloader
from simple_nnue.model import NNUE, train_nnue

import pandas as pd
import ast


def main():
    board = toguz.Board([1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18], [-1,-1], [0,0])
    mmx = toguz.MinimaxH(CONSTANTS.heuristic_weights)

    print(board.rotate().toString())
    print(board.toString())

    return 0

def check_dataset():
    file_path = "dataset_generation/flat_porn.parquet"

    df = pd.read_parquet(file_path)

    print("DF loaded")
    print(df.head())

    meta_epoch = 10

    nnue = NNUE()

    for i in range(meta_epoch):
        train_loader = create_dataloader(df.sample(frac=0.05).reset_index())
        val_loader = create_dataloader(df.sample(frac=0.01).reset_index())

        train_nnue(nnue, train_loader, val_loader)

if __name__ == '__main__':
    check_dataset()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
