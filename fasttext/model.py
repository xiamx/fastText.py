# fastText Model representation
import numpy as np
from numpy.linalg import norm

class Model(object):
    def __init__(self, words, vectors):
        self.words = words
        self.word_vector = dict(zip(words, vectors))

    def cosine_similarity(self, first_word, second_word):
        vector_dict = self.word_vector
        if first_word in vector_dict and second_word in vector_dict:
            v1 = vector_dict[first_word]
            v2 = vector_dict[second_word]
            dot_product = np.dot(v1,v2)
            cosine_sim = dot_product/(norm(v1)*norm(v2))
            return cosine_sim
