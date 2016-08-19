# Set encoding to support Python 2
# -*- coding: utf-8 -*-

# We use unicode_literals to generalize unicode syntax in plain string ''
# instead of u''. (to support python 3.2)
from __future__ import unicode_literals
import unittest
from os import path

import fasttext as ft

current_dir = path.dirname(__file__)
classifier_bin = path.join(current_dir, 'classifier.bin')
input_file = path.join(current_dir, 'dbpedia.train')
pred_file  = path.join(current_dir, 'classifier_pred_test.txt')
output = path.join(current_dir, 'generated_classifier')
test_result = path.join(current_dir, 'classifier_test_result.txt')
pred_result = path.join(current_dir, 'classifier_pred_result.txt')
test_file = path.join(current_dir, 'classifier_test.txt')

def read_labels(filename, label_prefix, unique=True):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            # Python 2 read file in ASCII encoding by default
            # so we need to decode the str to UTF-8 first.
            # But, in Python 3, str doesn't have decode method
            # so this decoding step make the test fails.
            # Python 3 read file in UTF-8 encoding by default so
            # we wrap this in the try-except to support both Python 2
            # and Python 3
            try:
                line = line.decode('utf-8')
            except:
                line = line

            label = line.split(',', 1)[0].strip()
            label = label.replace(label_prefix, '')
            if unique:
                if label in labels:
                    continue
                else:
                    labels.append(label)
            else:
                labels.append(label)
    return labels

# Test to make sure that classifier interface run correctly
class TestClassifierModel(unittest.TestCase):
    def test_load_classifier_model(self):
        label_prefix='__label__'
        model = ft.load_model(classifier_bin, label_prefix=label_prefix)

        # Make sure the model is returned correctly
        self.assertEqual(model.model_name, 'supervised')

        # Make sure all params loaded correctly
        # see Makefile on target test-supervised for the params
        self.assertEqual(model.dim, 100)
        self.assertEqual(model.word_ngrams, 2)
        self.assertEqual(model.min_count, 1)
        self.assertEqual(model.epoch, 5)
        self.assertEqual(model.bucket, 2000000)

        # Read labels from the the input_file
        labels = read_labels(input_file, label_prefix)

        # Make sure labels are loaded correctly
        self.assertTrue(sorted(model.labels) == sorted(labels))

    def test_train_classifier(self):
        # set params
        dim=10
        lr=0.005
        epoch=1
        min_count=1
        word_ngrams=3
        bucket=2000000
        thread=4
        silent=0
        label_prefix='__label__'

        # Train the classifier
        model = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
                min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
                thread=thread, silent=silent, label_prefix=label_prefix)

        # Make sure the model is generated correctly
        self.assertEqual(model.dim, dim)
        self.assertEqual(model.epoch, epoch)
        self.assertEqual(model.min_count, min_count)
        self.assertEqual(model.word_ngrams, word_ngrams)
        self.assertEqual(model.bucket, bucket)

        # Read labels from the the input_file
        labels = read_labels(input_file, label_prefix)

        # Make sure .bin and .vec are generated
        self.assertTrue(path.isfile(output + '.bin'))
        self.assertTrue(path.isfile(output + '.vec'))

    def test_classifier_test(self):
        # Read the test result from fasttext(1) using the same classifier model
        precision_at_one = 0.0
        num_examples = 0
        with open(test_result) as f:
            lines = f.readlines()
            precision_at_one = float(lines[0][5:].strip())
            num_examples = int(lines[1][20:].strip())

        # Load and test using the same model and test set
        classifier = ft.load_model(classifier_bin, label_prefix='__label__')
        p_at_1, num_ex = classifier.test(test_file)

        # Make sure that the test result is the same as the result generated
        # by fasttext(1)
        p_at_1 = float("{0:.2f}".format(p_at_1))
        self.assertEqual(p_at_1, precision_at_one)
        self.assertEqual(num_ex, num_examples)

    def test_classifier_predict(self):
        label_prefix = '__label__'
        # Load the pre-trained classifier
        classifier = ft.load_model(classifier_bin, label_prefix=label_prefix)

        # Read texts from the pred_file, prediction made by fasttext(1)
        texts = []
        with open(pred_file, 'r') as f:
            for line in f:
                try:
                    line = line.decode('utf-8')
                except:
                    line = line
                texts.append(line)

        # Predict the labels
        fasttext_labels = read_labels(pred_result, label_prefix=label_prefix,
                unique=False)
        labels = classifier.predict(texts)

        # Make sure the returned labels are the same as predicted by
        # fasttext(1)
        self.assertTrue(labels == fasttext_labels)

if __name__ == '__main__':
    unittest.main()
