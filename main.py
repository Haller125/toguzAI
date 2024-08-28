from game_logic import toguz
from dataset_generation import CONSTANTS
from dataset_generation.dataset import create_dataloader

import pandas as pd
import ast


def main():
    board = toguz.Board([1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18], [-1,-1], [0,0])
    mmx = toguz.MinimaxH(CONSTANTS.heuristic_weights)

    print(board.rotate().toString())
    print(board.toString())

    return 0

def check_dataset():
    file_path = "dataset_generation/porn.parquet"
    dataloader = create_dataloader(file_path)

    for i, (white_features, black_features, stm, targets) in enumerate(dataloader):
        print(f"Batch {i}:")
        print("White features:", white_features.shape)
        print("Black features:", black_features.shape)
        print("STM:", stm.shape)
        print("Targets:", targets.shape)
        print("\n")

        if i == 9:  # Print first 10 batches
            break

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    check_dataset()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
