from game_logic import toguz
from dataset_generation import CONSTANTS


def main():
    board = toguz.Board()
    mmx = toguz.MinimaxH(CONSTANTS.heuristic_weights)
    lb = toguz.Board(board.sockets, board.tuzdeks, board.kaznas)
    lb.makeMove(1, False)
    print(board.toString())
    print(lb.toString())

    

    return 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
