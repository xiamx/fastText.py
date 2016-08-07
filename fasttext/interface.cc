/* An interface for fastText */
#include <iostream>
#include <string>

#include "cpp/src/fasttext.cc"

void train_wrapper(int argc, char **argv, int silent)
{
    /* output file stream to redirect output from fastText library */
    std::string temp_file_name = std::tmpnam(nullptr);
    std::ofstream new_ofs(temp_file_name);
    std::streambuf* old_ofs = std::cout.rdbuf();

    /* if silent > 0, the log from train() function will be supressed */
    if(silent > 0) {
        std::cout.rdbuf(new_ofs.rdbuf());
    }

    train(argc, argv);

    if(silent > 0) {
        std::cout.rdbuf(old_ofs);
        new_ofs.close();
    }
}
