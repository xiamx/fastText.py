# Train module
# This module is used to create word vectors model or classifier
cimport utils
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy
from cpython.string cimport PyString_AsString

from .model import Model

import os

# Expose train function from cpp/src/fasttext.cc
cdef extern from "cpp/src/fasttext.cc":
    void train(int argc, char **argv)

# skipgram: Learn word representation using skipgram model
def skipgram(input_file, output, lr=0.05, dim=100, ws=5, epoch=5,
        min_count=5, neg=5, word_ngrams=1, loss='ns', bucket=2000000, minn=3,
        maxn=6, thread=12, verbose=10000, t=1e-4):
    """
    The following arguments are mandatory:
      input      training file path
      output     output file path

    The following arguments are optional:
      lr         learning rate [0.05]
      dim        size of word vectors [100]
      ws         size of the context window [5]
      epoch      number of epochs [5]
      minCount   minimal number of word occurences [5]
      neg        number of negatives sampled [5]
      wordNgrams max length of word ngram [1]
      loss       loss function {ns, hs, softmax} [ns]
      bucket     number of buckets [2000000]
      minn       min length of char ngram [3]
      maxn       max length of char ngram [6]
      thread     number of threads [12]
      verbose    how often to print to stdout [10000]
      t          sampling threshold [0.0001]
    """

    # Check if the input_file is valid
    if not os.path.isfile(input_file):
        raise ValueError('fastText: input file cannot be opened!')

    # Check if the output is writeable
    try:
        f = open(output, 'w')
        f.close()
    except IOError:
        raise IOError('fastText: output is not writeable!')

    # Initialize log & sigmoid tables
    utils.init_tables()

    # Setup argv, arguments and their values
    model_name = 'skipgram'
    argv = ['fasttext', model_name]
    args = ['-input', '-output', '-lr', '-dim', '-ws', '-epoch', '-minCount',
            '-neg', '-wordNgrams', '-loss', '-bucket', '-minn', '-maxn',
            '-thread', '-verbose', '-t']
    values = [input_file, output, lr, dim, ws, epoch, min_count, neg,
            word_ngrams, loss, bucket, minn, maxn, thread, verbose, t]

    for arg, value in zip(args, values):
        argv.append(arg)
        argv.append(str(value))
    argc = len(argv)

    # Converting Python object to C++
    cdef int c_argc = argc
    cdef char **c_argv = <char **>malloc(c_argc * sizeof(char *))
    for i, arg in enumerate(argv):
        c_argv[i] = PyString_AsString(arg)

    # Call train function with specified argc & argv
    train(c_argc, c_argv)

    # Free the log & sigmoid tables from the heap
    utils.free_tables()

    # Free the allocated memory
    # The content from PyString_AsString is not deallocated
    # info: https://docs.python.org/2/c-api/string.html#c.PyString_AsString
    free(c_argv)

    # Return fastText model
    input_path = os.path.join(os.getcwd(), input_file)
    output_bin = os.path.join(os.getcwd(), output + '.bin')
    output_vec = os.path.join(os.getcwd(), output + '.vec')
    return Model(model_name, input_path, output_bin, output_vec, lr, dim, ws,
        epoch, min_count, neg, word_ngrams, loss, bucket, minn, maxn, thread,
        verbose, t)
