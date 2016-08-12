# Set encoding to support Python 2
# -*- coding: utf-8 -*-

# We use unicode_literals to generalize unicode syntax in plain string ''
# instead of u''. (to support python 3.2)
from __future__ import unicode_literals
import unittest
from os import path

import fasttext as ft

supervised_file = path.join(path.dirname(__file__), 'supervised_params_test.bin')
input_file = path.join(path.dirname(__file__), 'supervised_params_test.txt')
output = path.join(path.dirname(__file__), 'generated_supervised')

# Test to make sure that supervised interface run correctly
class TestSupervisedModel(unittest.TestCase):
    def test_load_supervised_model(self):
        label_prefix='__label__'
        model = ft.load_model(supervised_file, label_prefix=label_prefix)

        # Make sure the model is returned correctly
        self.assertEqual(model.model_name, 'supervised')

        # Make sure all params loaded correctly
        # see Makefile on target test-supervised for the params
        self.assertEqual(model.dim, 10)
        self.assertEqual(model.word_ngrams, 2)
        self.assertEqual(model.min_count, 1)
        self.assertEqual(model.epoch, 5)
        self.assertEqual(model.bucket, 2000000)

        # Count how many labels are in the input_file
        labels = []
        with open(input_file, 'r') as f:
            for line in f:
                # str in python 3 doesn't have decode method
                try:
                    line = line.decode('utf-8')
                except:
                    line = line
                label = line.split(',')[0].strip()
                label = label.replace(label_prefix, '')
                if label in labels:
                    continue
                else:
                    labels.append(label)

        # Make sure labels are loaded correctly
        self.assertTrue(sorted(model.labels) == sorted(labels))

    def test_create_supervised_model(self):
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

        # train supervised model
        model = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
                min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
                thread=thread, silent=silent, label_prefix=label_prefix)

        # Make sure the model is generated correctly
        self.assertEqual(model.dim, dim)
        self.assertEqual(model.epoch, epoch)
        self.assertEqual(model.min_count, min_count)
        self.assertEqual(model.word_ngrams, word_ngrams)
        self.assertEqual(model.bucket, bucket)

        # Count how many labels are in the input_file
        labels = []
        with open(input_file, 'r') as f:
            for line in f:
                # str in python 3 doesn't have decode method
                try:
                    line = line.decode('utf-8')
                except:
                    line = line
                label = line.split(',')[0].strip()
                label = label.replace(label_prefix, '')
                if label in labels:
                    continue
                else:
                    labels.append(label)

        # Make sure .bin and .vec are generated
        self.assertTrue(path.isfile(output + '.bin'))
        self.assertTrue(path.isfile(output + '.vec'))

if __name__ == '__main__':
    unittest.main()
