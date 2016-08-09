# fastText C++ interface
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "cpp/src/real.h":
    ctypedef float real

cdef extern from "interface.h":
    cdef cppclass FastTextModel:
        vector[string] getWords()
        vector[vector[real]] getVectors()

    void trainWrapper(int argc, char **argvm, int silent)
    void loadModelWrapper(string filename, FastTextModel& model)
