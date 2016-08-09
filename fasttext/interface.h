#ifndef FASTTEXT_INTERFACE_H
#define FASTTEXT_INTERAFCE_H

#include <string>
#include <vector>
#include "cpp/src/real.h"

class FastTextModel {
    private:
        std::vector<std::string> _words;
        std::vector<std::vector<real>> _vectors;

    public:
        std::vector<std::string> getWords();
        std::vector<std::vector<real>> getVectors();

        void addWord(std::string word);
        void addVector(std::vector<real> vector);
};

void trainWrapper(int argc, char **argv, int silent);
void loadModelWrapper(std::string filename, FastTextModel& model);

#endif
