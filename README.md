# fasttext [![Build Status](https://travis-ci.org/salestock/fastText.py.svg?branch=master)](https://travis-ci.org/salestock/fastText.py)

fasttext is a Python interface for
[Facebook fastText](https://github.com/facebookresearch/fastText).

## Requirements

fasttext support Python 2.6 or newer. It requires
[Cython](https://pypi.python.org/pypi/Cython/) in order to build the C++ extension.

## Installation

```shell
pip install fasttext
```

## Example usage

This package has two main use cases: word representation learning and
text classification.

These were described in the two papers
[1](#enriching-word-vectors-with-subword-information)
and [2](#bag-of-tricks-for-efficient-text-classification).

### Word representation learning

In order to learn word vectors, as described in
[1](#enriching-word-vectors-with-subword-information), we can use
`fasttext.skipgram` and `fasttext.cbow` function like the following:

```python
import fasttext

# Skipgram model
model = fasttext.skipgram('data.txt', 'model')
print model.words # list of words in dictionary

# CBOW model
model = fasttext.cbow('data.txt', 'model')
print model.words # list of words in dictionary
```

where `data.txt` is a training file containing `utf-8` encoded text.
By default the word vectors will take into account character n-grams from
3 to 6 characters.

At the end of optimization the program will save two files:
`model.bin` and `model.vec`.

`model.vec` is a text file containing the word vectors, one per line.
`model.bin` is a binary file containing the parameters of the model
along with the dictionary and all hyper parameters.

The binary file can be used later to compute word vectors or
to restart the optimization.

The following `fasttext(1)` command is equivalent

```shell
# Skipgram model
./fasttext skipgram -input data.txt -output model

# CBOW model
./fasttext cbow -input data.txt -output model
```

### Obtaining word vectors for out-of-vocabulary words

The previously trained model can be used to compute word vectors for
out-of-vocabulary words.

```python
print model.get_vector('king') # get the vector of the word 'king'
```

the following `fasttext(1)` command is equivalent:

```shell
echo "king" | ./fasttext print-vectors model.bin
```

This will output the vector of word `king` to the standard output.

### Load pre-trained model

We can use `fasttext.load_model` to load pre-trained model:

```python
model = fasttext.load_model('model.bin')
print model.words # list of words in dictionary
print model.get_vector('king') # get the vector of the word 'king'
```

### Text classification

_Works in progress_


## API documentation

### Word vector model

```python
import fasttext

model = fasttext.skipgram(params)
model.words
model.get_vector(word)

model = fasttext.cbow(params)
model.words
model.get_vector(word)

model = fasttext.load_model('model.bin')
model.words
model.get_vector(word)
```

List of params and their default value:

```
input          training file path
output         output file path
lr             learning rate [0.05]
lr_update_rate change the rate of updates for the learning rate [100]
dim            size of word vectors [100]
ws             size of the context window [5]
epoch          number of epochs [5]
min_count      minimal number of word occurences [1]
neg            number of negatives sampled [5]
word_ngrams    max length of word ngram [1]
loss           loss function {ns, hs, softmax} [ns]
bucket         number of buckets [2000000]
minn           min length of char ngram [3]
maxn           max length of char ngram [6]
thread         number of threads [12]
t              sampling threshold [0.0001]
silent         disable the log output from the C++ extension [1]
```

## References

### Enriching Word Vectors with Subword Information

[1] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, [*Enriching Word Vectors with Subword Information*](https://arxiv.org/pdf/1607.04606v1.pdf)

```
@article{bojanowski2016enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.04606},
  year={2016}
}
```

### Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/pdf/1607.01759v2.pdf)

```
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```

(\* These authors contributed equally.)

## Join the fastText community

* Facebook page: https://www.facebook.com/groups/1174547215919768
* Google group: https://groups.google.com/forum/#!forum/fasttext-library


