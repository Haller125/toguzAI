from toguz import Board, MinimaxH
from dataset_generation.CONSTANTS import heuristic_weights, EVAL_DEPTH

import pandas as pd
import numpy as np
import json

mmh = MinimaxH(heuristic_weights)
def play_game(board: Board, hs: dict, hs_lookup: dict):
    local_board = Board(board.sockets, board.tuzdeks, board.kaznas)
    game_finished = False
    list_of_pos = []
    cp = False
    i = 0

    arr = [local_board.sockets, local_board.tuzdeks, local_board.kaznas]
    arr_key = json.dumps(arr)
    if arr_key not in hs_lookup:
        list_of_pos.append(str(arr))

    while not game_finished:
        if not local_board.isMovePossible(False):
            local_board.atsyrauFunction(True)
            game_finished = True
            continue
        if i > 1000:
            print("Infinite loop")
            return list_of_pos, 0.5

        _, move = mmh.minimaxWithABWithHeuristics(local_board, EVAL_DEPTH, -1000000, 1000000, False, -1)
        local_board.makeMove(move, False)

        if not cp:
            arr = [local_board.sockets, local_board.tuzdeks, local_board.kaznas]
        else:
            temp_board = local_board.rotate()
            arr = [temp_board.sockets, temp_board.tuzdeks, temp_board.kaznas]
        arr_key = json.dumps(arr)
        if arr_key not in hs_lookup:
            list_of_pos.append(arr_key)
        else:
            return list_of_pos, hs[arr_key]

        if local_board.kaznas[0] >= 81 or local_board.kaznas[1] >= 81:
            game_finished = True
            continue

        local_board = local_board.rotate()
        cp = not cp
        i += 1
    if not cp:
        return list_of_pos, 0.5 if local_board.kaznas[0] == local_board.kaznas[1] else (1 if local_board.kaznas[0] > 81 else 0)
    return list_of_pos, 0.5 if local_board.kaznas[0] == local_board.kaznas[1] else (0 if local_board.kaznas[0] > 81 else 1)


def main():
    import pickle

    df = pd.read_parquet("ultimate_porn.parquet")
    hs_lookup = {}

    with open('saved_dictionary.pkl', 'rb') as f:
        try:
            hs = pickle.load(f)
            for key in hs.keys():
                hs_lookup[key] = True
        except EOFError:
            hs = {}
            hs_lookup = {}
        f.close()

    for i in range(500001):
        row = df.sample(n=1).iloc[0]
        sockets = np.concatenate([row['white'], row["black"]], axis=None) if not row["stm"] else np.concatenate([row["black"], row['white']], axis=None)
        store = [sockets.tolist(), row['tuzdeks'].tolist(), row['kaznas'].tolist()]
        store_key = json.dumps(store)
        if store_key in hs_lookup:
            continue
        else:
            board = Board(sockets.tolist(), row['tuzdeks'].tolist(), row['kaznas'].tolist())
            list_of_pos, result = play_game(board, hs, hs_lookup)
            if row['stm']:
                result = 1 - result
            for pos in list_of_pos:
                hs[pos] = result
                hs_lookup[pos] = True

        if i % 1000 == 0:
            print(f"Iteration {i} finished")
            print(len(hs))
            print("Saved")

            if i == 500000:
                break
    print("Finished")
    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(hs, f)


if __name__ == '__main__':
    main()
