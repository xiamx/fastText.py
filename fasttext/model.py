# fastText Model representation in Python
import numpy as np
from numpy.linalg import norm

class WordVectorModel(object):
    def __init__(self, model):
        self._model = model
        self.words = model.get_words()
        self.input_file_name = model.inputFileName;
        self.test_file_name = model.testFileName;
        self.output_file_name = model.outputFileName;
        self.lr = model.lr;
        self.dim = model.dim;
        self.ws = model.ws;
        self.epoch = model.epoch;
        self.min_count = model.minCount;
        self.neg = model.neg;
        self.word_ngrams = model.wordNgrams;
        self.loss_name = model.lossName;
        self.model_name = model.modelName;
        self.bucket = model.bucket;
        self.minn = model.minn;
        self.maxn = model.maxn;
        self.thread = model.thread;
        self.verbose = model.verbose;
        self.t = model.t;

    def get_vector(self, word):
        return self._model.get_vector(word)

    def cosine_similarity(self, first_word, second_word):
        v1 = self.get_vector(first_word)
        v2 = self.get_vector(second_word)
        dot_product = np.dot(v1,v2)
        cosine_sim = dot_product/(norm(v1)*norm(v2))
        return cosine_sim

