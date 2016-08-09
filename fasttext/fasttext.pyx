# fastText C++ interface
cimport utils
from interface cimport trainWrapper
from interface cimport loadModelWrapper
from interface cimport FastTextModel

# Python/C++ standart libraries
from libc.stdlib cimport malloc, free
from cpython.string cimport PyString_AsString
from libcpp.string cimport string

# Python module
import os
from model import WordVectorModel

# This class wrap C++ class FastTextModel, so it can be accessed via Python
cdef class FastTextModelWrapper:
    cdef FastTextModel fm

    def __cinit__(self):
        self.fm = FastTextModel()

    def get_words(self):
        return self.fm.getWords()

    def get_vector(self, string word):
        return self.fm.getVectorWrapper(word)

    @property
    def inputFileName(self):
        return self.fm.inputFileName;

    @property
    def testFileName(self):
        return self.fm.testFileName;

    @property
    def outputFileName(self):
        return self.fm.outputFileName;

    @property
    def lr(self):
        return self.fm.lr;

    @property
    def dim(self):
        return self.fm.dim;

    @property
    def ws(self):
        return self.fm.ws;

    @property
    def epoch(self):
        return self.fm.epoch;

    @property
    def minCount(self):
        return self.fm.minCount;

    @property
    def neg(self):
        return self.fm.neg;

    @property
    def wordNgrams(self):
        return self.fm.wordNgrams;

    @property
    def lossName(self):
        return self.fm.lossName;

    @property
    def modelName(self):
        return self.fm.modelName;

    @property
    def bucket(self):
        return self.fm.bucket;

    @property
    def minn(self):
        return self.fm.minn;

    @property
    def maxn(self):
        return self.fm.maxn;

    @property
    def thread(self):
        return self.fm.thread;

    @property
    def verbose(self):
        return self.fm.verbose;

    @property
    def neg(self):
        return self.fm.neg;

    @property
    def t(self):
        return self.fm.t;

    @property
    def label(self):
        return self.fm.label;

def skipgram(string input_file, string output, lr=0.05, dim=100, ws=5, epoch=5,
        min_count=5, neg=5, word_ngrams=1, loss='ns', bucket=2000000, minn=3,
        maxn=6, thread=12, verbose=10000, t=1e-4, silent=1):
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
      silent     supress the output from fastText [1]
    """

    # Check if the input_file is valid
    if not os.path.isfile(input_file):
        raise ValueError('fastText: input file cannot be opened!')

    # Check if the output is writeable
    try:
        f = open(output, 'w')
        os.remove(output)
        f.close()
    except IOError:
        raise IOError('fastText: output is not writeable!')

    # Initialize log & sigmoid tables
    utils.initTables()

    # Setup argv, arguments and their values
    model_name = 'skipgram'
    py_argv = ['fasttext', model_name]
    py_args = ['-input', '-output', '-lr', '-dim', '-ws', '-epoch', '-minCount',
            '-neg', '-wordNgrams', '-loss', '-bucket', '-minn', '-maxn',
            '-thread', '-verbose', '-t']
    values = [input_file, output, lr, dim, ws, epoch, min_count, neg,
            word_ngrams, loss, bucket, minn, maxn, thread, verbose, t]

    for arg, value in zip(py_args, values):
        py_argv.append(arg)
        py_argv.append(str(value))
    argc = len(py_argv)

    # Converting Python object to C++
    cdef int c_argc = argc
    cdef char **c_argv = <char **>malloc(c_argc * sizeof(char *))
    for i, arg in enumerate(py_argv):
        c_argv[i] = PyString_AsString(arg)

    # Run the train wrapper
    trainWrapper(c_argc, c_argv, silent)

    # Load the model
    model = load_model(output + '.bin')

    # Free the log & sigmoid tables from the heap
    utils.freeTables()

    # Free the allocated memory
    # The content from PyString_AsString is not deallocated
    free(c_argv)

    return model

# load_model: load a word vector model
def load_model(string filename):
    # Check if the filename is readable
    if not os.path.isfile(filename):
        raise ValueError('fastText: trained model cannot be opened!')

    model = FastTextModelWrapper()
    loadModelWrapper(filename, model.fm)

    # TODO: handle supervised here
    if model.fm.modelName == 'skipgram':
        return WordVectorModel(model)
    else:
        raise ValueError('fastText: model name not exists!')
