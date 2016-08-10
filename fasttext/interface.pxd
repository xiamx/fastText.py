# fastText C++ interface
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "cpp/src/real.h":
    ctypedef float real

cdef extern from "interface.h":
    cdef cppclass FastTextModel:
        FastTextModel()
        int dim;
        int ws;
        int epoch;
        int minCount;
        int neg;
        int wordNgrams;
        string lossName;
        string modelName;
        int bucket;
        int minn;
        int maxn;
        int lrUpdateRate;
        double t;

        vector[string] getWords()
        vector[real] getVectorWrapper(string word)

    void trainWrapper(int argc, char **argvm, int silent)
    void loadModelWrapper(string filename, FastTextModel& model)

