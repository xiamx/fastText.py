#ifndef FASTTEXT_INTERFACE_H
#define FASTTEXT_INTERAFCE_H

#include <string>
#include <vector>

#include "cpp/src/real.h"
#include "cpp/src/args.h"
#include "cpp/src/dictionary.h"
#include "cpp/src/matrix.h"

class FastTextModel {
    private:
        std::vector<std::string> _words;
        Dictionary _dict;
        Matrix _input_matrix;
        Matrix _output_matrix;

    public:
        FastTextModel();
        int dim;
        int ws;
        int epoch;
        int minCount;
        int neg;
        int wordNgrams;
        std::string lossName;
        std::string modelName;
        int bucket;
        int minn;
        int maxn;
        double lr;
        int lrUpdateRate;
        double t;

        std::vector<std::string> getWords();
        std::vector<real> getVectorWrapper(std::string word);
        std::vector<double> classifierTest(std::string filename);
        std::string classifierPredict(std::string text);

        void addWord(std::string word);
        void setDict(Dictionary dict);
        void setMatrix(Matrix& input, Matrix& output);
        void setArg(Args arg);

        Dictionary getDictionary();
};

void trainWrapper(int argc, char **argv, int silent);
void loadModelWrapper(std::string filename, FastTextModel& model);

#endif

