import os, sys
from tqdm import tqdm
import ast

from CONSTANTS import EVAL_DEPTH, heuristic_weights, PLAY_DEPTH, NUMBER_OF_GAMES, DFS_DEPTH
from toguz import Board, MinimaxH

import pandas as pd

def create_rotated_dataset(checkpoint_interval=1000):
    df = pd.read_csv("gayporn.csv")
    df = df.reset_index()

    df['sockets'] = df['sockets'].apply(ast.literal_eval)
    df['tuzdeks'] = df['tuzdeks'].apply(ast.literal_eval)
    df['kaznas'] = df['kaznas'].apply(ast.literal_eval)

    checkpoint_file = "checkpoint.csv"
    if os.path.exists(checkpoint_file):
        print(f"Checkpoint file found. Loading from {checkpoint_file}...")

        new_df = pd.read_csv(checkpoint_file)

        new_df['sockets'] = new_df['sockets'].apply(ast.literal_eval)
        new_df['tuzdeks'] = new_df['tuzdeks'].apply(ast.literal_eval)
        new_df['kaznas'] = new_df['kaznas'].apply(ast.literal_eval)

        start_index = len(new_df)

        print(f"Resuming from index {start_index}")
    else:
        new_df = pd.DataFrame(columns=["sockets", "kaznas", "tuzdeks", "eval"])
        start_index = 0

    mmh = MinimaxH(heuristic_weights)

    for index in tqdm(range(start_index, len(df)), initial=start_index, total=len(df)):
        row = df.iloc[index]
        board = Board(row["sockets"], row["tuzdeks"], row["kaznas"])
        board = board.rotate()
        evaluation, _ = mmh.minimaxWithABWithHeuristics(board, EVAL_DEPTH, -1000000, 1000000, False, -1)
        new_row = pd.DataFrame({
            "sockets": [board.sockets],
            "kaznas": [board.kaznas],
            "tuzdeks": [board.tuzdeks],
            "eval": [evaluation]})
        new_df = pd.concat([new_df, new_row], ignore_index=True)

        if (index + 1) % checkpoint_interval == 0:
            new_df.to_csv(checkpoint_file, index=False)
            print(f"\nCheckpoint saved at index {index + 1}")
            sys.stdout.flush()

    new_df.to_csv("hetero_porn.csv", index=False)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print("Dataset creation completed successfully.")

if __name__ == '__main__':
    create_rotated_dataset()