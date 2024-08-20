//
// Created by qwerty on 06.03.2024.
//

#ifndef DIPLOMA_TOGUZ_H
#define DIPLOMA_TOGUZ_H

#include <iostream>
#include <array>
#include "CONSTANTS.h"
#include <iostream>
#include <limits>
#include <array>
#include <sstream>

struct Board{
public:
    std::array<int_fast8_t, K*2> sockets{};
    std::array<int_fast8_t, 2> tuzdeks{};
    std::array<int_fast8_t, 2> kaznas{};
    Board();
    Board(std::array<int_fast8_t, 2*K>& inSocket, std::array<int_fast8_t, 2>& inTuzdek, std::array<int_fast8_t, 2>& inKaznas);
    int16_t getSumOfOtausOfPlayer(const bool p);
    int8_t getNumOfOddCells(const bool p);
    int8_t getNumOfEvenCells(const bool p);
    float heurestic1(const bool p);
    int8_t playSocket(const int8_t s);
    bool isMovePossible(const bool p);
    int16_t atsyrauFunction(const bool p);
    bool tuzdekPossible(const int8_t s, const bool player) const;
    void accountSocket(const int8_t s, const bool player);
    void pli(const int8_t s, const bool tuzdek, const bool player);
    Board rotate() const;
    static int8_t idx(const int8_t s);
    void makeMove(const int8_t s, const bool player);
    std::string toString() const;
};

Board::Board() {
    sockets.fill(N);
    tuzdeks.fill(-1);
    kaznas.fill(0);
}

Board::Board(std::array<int8_t, 2 * K>& inSocket, std::array<int8_t, 2>& inTuzdek, std::array<int8_t, 2>& inKaznas) {
    sockets = inSocket;
    tuzdeks = inTuzdek;
    kaznas = inKaznas;
}

int16_t Board::getSumOfOtausOfPlayer(const bool p) {
    int16_t sum = 0;
    for (int8_t i = p * K; i < K * (1 + p); ++i) {
        sum += sockets[i];
    }
    return sum;
}

int8_t Board::getNumOfOddCells(const bool p){
    int8_t sum = 0;
    for (int8_t i = p * K; i < K * (1 + p); ++i) {
        if ((sockets[i] & 1) == 1) sum++;
    }
    return sum;
}

int8_t Board::getNumOfEvenCells(const bool p){
    int8_t sum = 0;
    for (int8_t i = p * K; i < K * (1 + p); ++i) {
        if (!(sockets[i] & 1)) sum++;
    }
    return sum;
}

float Board::heurestic1(const bool p = 0) {
    return (kaznas[p] - kaznas[1 - p]) * 0.9f +
           (getSumOfOtausOfPlayer(p)) * 0.2f +
           (tuzdeks[p]) * 0.8f +
           (tuzdeks[1-p]) * -0.8f +
           (getNumOfOddCells(1-p)) * 0.6f +
           (getNumOfEvenCells(p)) * 0.6f;
}

int8_t Board::playSocket(const int8_t s) {
    if (sockets[s] == 1) {
        sockets[s] = 0;
        sockets[idx(s + 1)] += 1;
        return idx(s + 1);
    }
    int8_t tempSocket = sockets[s];
    sockets[s] = 1;
    for (int8_t i = s + 1; i < s + tempSocket; ++i) {
        int8_t id = idx(i);
        sockets[id] += 1;
    }
    int8_t result = idx(s + tempSocket - 1);
    return result;
}

bool Board::isMovePossible(const bool p) {
    for (int8_t i = p * K; i < K * (1 + p); ++i) {
        if (sockets[i] != 0) {
            return true;
        }
    }
    return false;
}

int16_t Board::atsyrauFunction(const bool p) {
    for (int8_t i = 0; i <= K * 2 - 1; ++i) {
        kaznas[(int8_t)i/K] += sockets[i];
        sockets[i] = 0;
    }
    return kaznas[p];
}

[[nodiscard]] bool Board::tuzdekPossible(const int8_t s, const bool player) const {
    return sockets[s] == 3
           && (s / K == (1 - player)) // opposite side
           && ((s + 1) % K) != 0 // not the last
           && tuzdeks[player] == -1 // not having tuzdek
           && (tuzdeks[1 - player] == -1 || tuzdeks[1 - player] % K != (s % K)); // not symmetrical tuzdek
}

void Board::accountSocket(const int8_t s, const bool player) {
    if (s / K == (1 - player) // correct side
        && sockets[s] % 2 == 0) { // and even
        kaznas[player] += sockets[s];
        sockets[s] = 0;
    }
    for (int8_t playerIt = 0; playerIt < 2; ++playerIt) {
        if (tuzdeks[playerIt] != -1) {
            kaznas[playerIt] += sockets[tuzdeks[playerIt]];
            sockets[tuzdeks[playerIt]] = 0;
        }
    }
}

void Board::pli(const int8_t s, const bool tuzdek, const bool player) {
    auto target = playSocket(s);
    if (tuzdek) {
        tuzdeks[player] = target;
    }
    accountSocket(target, player);
}

void Board::makeMove(const int8_t s, const bool player) {
    int8_t target = this->playSocket(s);
    if (this->tuzdekPossible(target, player)) {
        this->tuzdeks[player] = target;
    }
    this->accountSocket(target, player);
}

Board Board::rotate() const {
    Board result(*this);
    result.kaznas[0] = kaznas[1];
    result.kaznas[1] = kaznas[0];

    result.tuzdeks[0] = tuzdeks[1];
    result.tuzdeks[1] = tuzdeks[0];

    for (int8_t i = 0; i < K; ++i) {
        result.sockets[i] = sockets[K + i];
        result.sockets[K + i] = sockets[i];
    }
    return result;
}


int8_t Board::idx(const int8_t s) {
    return s % (2 * K);
}

std::string Board::toString() const {
    std::stringstream ss;
    for (int player = 1; player >= 0; --player) {
        ss << static_cast<int>(player) << ":\t";
        for (int i = 0; i < K; ++i) {
            int idx = player ? (2 * K - i - 1) : i;
            ss << " " << static_cast<int>(sockets[idx]);
            if (tuzdeks[(1 - player)] == idx) {
                ss << "*";
            }
            ss << "\t";
        }
        ss << "Kazna: " << static_cast<int>(kaznas[player]);
        ss << std::endl;
    }
    ss << "\t";
    for (int i = 0; i < K; ++i) {
        ss << "-" << (i + 1) << "-\t";
    }
    ss << std::endl;
    return ss.str(); // Convert stringstream to string and return it
}

#endif //DIPLOMA_TOGUZ_H

#ifndef DIPLOMA_MINIMAXH_H
#define DIPLOMA_MINIMAXH_H
#include "toguz.h"
#include "CONSTANTS.h"
#include <unordered_set>
#include <array>

struct MinimaxH{
    std::array<float, NUM_OF_HEURISTICS> weights{};
public:
    explicit MinimaxH(std::array<float, NUM_OF_HEURISTICS>& inputWeights);
    std::tuple<float, int> minimaxWithABWithHeuristics(Board &board, int depth, float alpha, float beta, bool player, int &move) const;
    float heuristic1(Board &board, bool p = false) const;
private:
    static int getNumOfLegalMoves(Board &board, bool p);
    static int getNumOfLegalMovesWithDistinctDestination(Board &board, bool p);

};

MinimaxH::MinimaxH(std::array<float, NUM_OF_HEURISTICS>& inputWeights) {
    weights = inputWeights;
}

std::tuple<float, int> MinimaxH::minimaxWithABWithHeuristics(Board &board, int depth, float alpha, float beta, bool player, int &move) const {
    if (depth == 0) {
        move = -1;
        return std::make_tuple(this->heuristic1(board), move) ;
    }else if (board.kaznas[0] > K * N || board.kaznas[1] > K * N){
        move = -1;
        return std::make_tuple(board.kaznas[0] > K * N ? 100000000.f: -100000000.f, move);
    }

    int dummyMove;
    bool dummyTuzdek;

    float bestValue = (player == 0) ? INT_MIN : INT_MAX;

    bool played = false;
    if (!board.isMovePossible(player)) {
        Board copyBoard(board);
        copyBoard.atsyrauFunction(!player);
        return std::make_tuple(board.kaznas[0] > K*N ? 100000000.f : -100000000.f, dummyMove) ;
    }
    for (int i = player * K; i < player * K + K; ++i) {
        Board localBoard(board);
        if (localBoard.sockets[i] == 0) continue;
        played = true;
        auto target = localBoard.playSocket(i);
        bool isTuzdekPossible = localBoard.tuzdekPossible(target, player);

        float value = INT_MIN;
        if (isTuzdekPossible) {
            Board tuzdekBoard(localBoard);
            tuzdekBoard.tuzdeks[player] = target;
            tuzdekBoard.accountSocket(target, player);

            value = get<0>(this->minimaxWithABWithHeuristics(tuzdekBoard, depth - 1, alpha, beta, 1 - player, dummyMove));
        }else {
            localBoard.accountSocket(target, player);
            value = get<0>(this->minimaxWithABWithHeuristics(localBoard, depth - 1, alpha, beta, 1 - player, dummyMove));
        }
        if (player == 0 && value > bestValue) {  // Maximizing player
            bestValue = value;
            move = i;
            alpha = std::max(alpha, bestValue);
        } else if (player == 1 && value < bestValue) {  // Minimizing player
            bestValue = value;
            move = i;
            beta = std::min(beta, bestValue);
        }
        if (beta <= alpha) break;
    }

    if (played) {
        return std::make_tuple(bestValue, move) ;
    } else {
        move = -1;
        return std::make_tuple((float)board.kaznas[0], move) ;
    }
}

int MinimaxH::getNumOfLegalMoves(Board &board, const bool p){
    int result = 0;
    for (int i = p * K; i < K * (1 + p); ++i) {
        if (board.sockets[i]) {
            int a = board.sockets[i] / (2*K);
            int b = board.sockets[i] % (2*K);
            int c = board.sockets[b] + a;
            if ((c == 3) && (b / K == (1 - p)) &&   // opposite side
            (((b + 1) % K) != 0) &&                 // not the last
            (board.tuzdeks[p] == -1) &&             // not having tuzdek
            (board.tuzdeks[1 - p] == -1 || board.tuzdeks[1 - p] % K != b % K))  // not symmetrical tuzdek
            {
                result++;
            }
            result++;
        }
    }
    return result;
}

int MinimaxH::getNumOfLegalMovesWithDistinctDestination(Board &board, bool p){
    int result = 0;
    std::unordered_set<int> destinations = {};
    for (int i = p * K; i < K * (1 + p); ++i) {
        if (board.sockets[i])
        {
            int a = board.sockets[i] / (2*K);
            int b = board.sockets[i] % (2*K);
            int c = board.sockets[b] + a;
            if (destinations.count(b) <= 0){
                destinations.insert(b);
            }
        }
    }
    result = destinations.size();
    return result;
}

float MinimaxH::heuristic1(Board &board, bool p) const {
    return (board.kaznas[p]) * weights[0] +
           (board.getSumOfOtausOfPlayer(p)) * weights[1] +
           (board.tuzdeks[p]) * weights[2] +
           (board.tuzdeks[1-p]) * weights[3] +
           (board.getNumOfOddCells(1-p)) * weights[4] +
           (board.getNumOfEvenCells(p)) * weights[5] +
           (this->getNumOfLegalMoves(board,p)) * weights[6] +
           (this->getNumOfLegalMovesWithDistinctDestination(board, p)) * weights[7];
}

#endif //DIPLOMA_MINIMAXH_H
