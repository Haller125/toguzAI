from game_logic import toguz
from dataset_generation import CONSTANTS


def main():
    board = toguz.Board([1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18], [-1,-1], [0,0])
    mmx = toguz.MinimaxH(CONSTANTS.heuristic_weights)

    print(board.rotate().toString())
    print(board.toString())

    

    return 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
