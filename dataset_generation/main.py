import random

from CONSTANTS import EVAL_DEPTH, heuristic_weights, PLAY_DEPTH, NUMBER_OF_GAMES, DFS_DEPTH
from toguz import Board, MinimaxH

import pandas as pd

dataset = pd.DataFrame(columns=["sockets", "kaznas", "tuzdeks", "eval"])


def eval_play(board: Board, current_player: bool = False):
    lb = Board(board.sockets, board.tuzdeks, board.kaznas)
    game_finished = False
    current_player = current_player
    evaluation, _ = MinimaxH(heuristic_weights).minimaxWithABWithHeuristics(lb, EVAL_DEPTH, -1000000, 1000000,
                                                                            current_player, -1)

    mmh = MinimaxH(heuristic_weights)
    while True:
        if game_finished:
            return evaluation, lb.kaznas[0] >= 81
        if not lb.isMovePossible(current_player):
            lb.atsyrauFunction(not current_player)
            game_finished = True
            continue
        _, move = mmh.minimaxWithABWithHeuristics(lb, PLAY_DEPTH, -1000000, 1000000, current_player, -1)
        lb.makeMove(move, current_player)
        if lb.kaznas[0] >= 81 or lb.kaznas[1] >= 81:
            game_finished = True
        current_player = not current_player


def create_dataset_v1(board: Board, depth: int):
    """ Create dataset by playing games by DFS """
    global dataset
    mmh = MinimaxH(heuristic_weights)
    evaluation, _ = mmh.minimaxWithABWithHeuristics(board, EVAL_DEPTH, -1000000, 1000000, False, -1)
    new_row = pd.DataFrame({
        "sockets": [board.sockets],
        "kaznas": [board.kaznas],
        "tuzdeks": [board.tuzdeks],
        "eval": [evaluation]})
    dataset = pd.concat([dataset, new_row], ignore_index=True)
    if depth == 0:
        return dataset
    for i in range(9):
        lb = Board(board.sockets, board.tuzdeks, board.kaznas)
        if lb.sockets[i] == 0:
            continue
        lb.makeMove(i, False)
        create_dataset_v1(lb.rotate(), depth - 1)


def random_move(board: Board, current_player: bool):
    cp = 1 if current_player else 0
    """ Make random move """
    move = random.randint(cp * 9, (cp * 9) + 8)
    while board.sockets[move] == 0:
        move = random.randint(cp * 9, (cp * 9) + 8)
    return move


def create_dataset_v2():
    """ Create dataset by playing eventual random moves """
    mmh = MinimaxH(heuristic_weights)
    df = pd.DataFrame(columns=["sockets", "kaznas", "tuzdeks", "eval", "player"])
    i = 0
    j = 0  # first moves to ignore counter
    while True:
        lb = Board()
        game_finished = False
        current_player = False
        switch = True
        while switch:
            if game_finished:
                switch = False
                continue
            if not lb.isMovePossible(current_player):
                lb.atsyrauFunction(not current_player)
                game_finished = True
                continue

            evaluation, minmax_move = mmh.minimaxWithABWithHeuristics(lb, EVAL_DEPTH, -1000000, 1000000, False, -1)
            if j >= DFS_DEPTH:
                new_row = pd.DataFrame({
                    "sockets": [lb.sockets],
                    "kaznas": [lb.kaznas],
                    "tuzdeks": [lb.tuzdeks],
                    "eval": [evaluation]})
                df = pd.concat([df, new_row], ignore_index=True)

            if random.random() < 0.5:
                move = minmax_move
            else:
                move = random_move(lb, current_player)

            lb.makeMove(move, current_player)
            if lb.kaznas[0] >= 81 or lb.kaznas[1] >= 81:
                game_finished = True
            j += 1
            lb = lb.rotate()
        print(f"Game {i} finished")
        i += 1
        j = 0
        df.to_csv("4.csv")


def main():
    # board = Board()
    # create_dataset_v1(board, DFS_DEPTH)
    # dataset.to_csv("datasetDFS.csv")
    # #
    # # create_dataset_v2()

    import numpy as np
    import torch

    def convert_feature_column(feature_column):
        converted = [np.array([np.array(x, dtype=np.int8) for x in i]).flatten() for i in feature_column]
        return converted

    df1 = pd.read_parquet("porn.parquet")
    df1['white_features'] = convert_feature_column(df1['white_features'])
    df1['black_features'] = convert_feature_column(df1['black_features'])

    df1.to_parquet("flat_porn.parquet")

if __name__ == '__main__':
    main()
