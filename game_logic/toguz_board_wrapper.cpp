//
// Created by qwerty on 13.07.2024.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "toguz.h"

namespace py = pybind11;

PYBIND11_MODULE(toguz, m) {
    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def(py::init<std::array<int8_t, 2*K>&, std::array<int8_t, 2>&, std::array<int8_t, 2>&>())
        .def("getSumOfOtausOfPlayer", &Board::getSumOfOtausOfPlayer)
        .def("getNumOfOddCells", &Board::getNumOfOddCells)
        .def("getNumOfEvenCells", &Board::getNumOfEvenCells)
        .def("heurestic1", &Board::heurestic1)
        .def("playSocket", &Board::playSocket)
        .def("isMovePossible", &Board::isMovePossible)
        .def("atsyrauFunction", &Board::atsyrauFunction)
        .def("tuzdekPossible", &Board::tuzdekPossible)
        .def("accountSocket", &Board::accountSocket)
        .def("pli", &Board::pli)
        .def("rotate", &Board::rotate)
        .def_static("idx", &Board::idx)
        .def("makeMove", &Board::makeMove)
        .def("toString", &Board::toString)
        .def_readwrite("sockets", &Board::sockets)
        .def_readwrite("tuzdeks", &Board::tuzdeks)
        .def_readwrite("kaznas", &Board::kaznas);

    py::class_<MinimaxH>(m, "MinimaxH")
        .def(py::init<std::array<float, NUM_OF_HEURISTICS>&>())
        .def("minimaxWithABWithHeuristics", &MinimaxH::minimaxWithABWithHeuristics)
        .def("heuristic1", &MinimaxH::heuristic1);
}