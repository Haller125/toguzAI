import toguz


def performance(n=10000000):
    import time
    # accum = 0
    # start = time.time()
    # for i in range(n):
    #     board = toguz.Board()
    #     board.makeMove(1, 0)
    #
    # end = time.time()
    # accum += end - start
    # print(f"{accum} seconds for {n} moves")

    heuristic_weights = (0.754946, 0.572817, 0.951618, -0.416603, -0.316876, -0.0844536, 0.513683, 0.347318, 0.11385)

    mmh = toguz.MinimaxH(heuristic_weights)

    board = toguz.Board([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0], [-1, -1], [80, 79])

    eval = mmh.minimaxWithABWithHeuristics(board, 3, -1000000, 1000000, True, -1)

    print(eval)

    board.makeMove(8, False)

    print(board.toString())

    print(board.tuzdeks)


if __name__ == '__main__':
    performance()
