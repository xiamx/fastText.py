/* An interface for fastText */
#include <iostream>
#include <string>
#include <vector>

#include "interface.h"
#include "cpp/src/real.h"
#include "cpp/src/dictionary.h"
#include "cpp/src/matrix.h"
#include "cpp/src/vector.h"

std::vector<std::string> FastTextModel::getWords()
{
    return _words;
}

std::vector<std::vector<real>> FastTextModel::getVectors()
{
    return _vectors;
}

void FastTextModel::addWord(std::string word)
{
    _words.push_back(word);
}

void FastTextModel::addVector(std::vector<real> vector)
{
    _vectors.push_back(vector);
}

#include "cpp/src/fasttext.cc"

void trainWrapper(int argc, char **argv, int silent)
{
    /* output file stream to redirect output from fastText library */
    std::string temp_file_name = std::tmpnam(nullptr);
    std::ofstream new_ofs(temp_file_name);
    std::streambuf* old_ofs = std::cout.rdbuf();

    /* if silent > 0, the log from train() function will be supressed */
    if(silent > 0) {
        std::cout.rdbuf(new_ofs.rdbuf());
        train(argc, argv);
        std::cout.rdbuf(old_ofs);
    } else {
        train(argc, argv);
    }

    new_ofs.close();
}

void loadModelWrapper(std::string filename, FastTextModel& model)
{
    Dictionary dict;
    Matrix input, output;
    loadModel(filename, dict, input, output);

    Vector vec(args.dim);
    for(int32_t i = 0; i < dict.nwords(); i++) {
        std::string word  = dict.getWord(i);
        model.addWord(word);

        getVector(dict, input, vec, word);
        std::vector<real> vector(vec.data_, vec.data_ + vec.m_);
        model.addVector(vector);
    }
}
