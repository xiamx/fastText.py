# fastText Model representation

class Model(object):
    def __init__(self, name, input_file, output_bin, output_vec, lr, dim, ws,
            epoch, min_count, neg, word_ngrams, loss, bucket, minn, maxn,
            thread, verbose, t):

        # fastText model attributes
        self.name = name
        self.input_file = input_file
        self.output_bin = output_bin
        self.output_vec = output_vec
        self.lr = lr
        self.dim = dim
        self.ws = ws
        self.epoch = epoch
        self.min_count = min_count
        self.neg = neg
        self.word_ngrams = word_ngrams
        self.loss = loss
        self.bucket = bucket
        self.minn = minn
        self.maxn = maxn
        self.thread = thread
        self.verbose = verbose
        self.t = t

        # TODO: load word_vectors here
        # self.word_vectors # dictionary that map word to numpy array

