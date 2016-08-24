# Set encoding to support Python 2
# -*- coding: utf-8 -*-

import unittest
from os import path

import fasttext as ft

current_dir = path.dirname(__file__)
params_txt = path.join(current_dir, 'default_params_result.txt')

# Test to make sure that default params is equivalent as fastetxt(1)
class TestDefaultParams(unittest.TestCase):
    def test_default_params(self):
        default_args = {}
        with open(params_txt, 'r') as f:
            for line in f:
                try:
                    line = line.decode('utf-8')
                except:
                    line = line

                raw = line.split(' ')
                key = raw[0]
                value = raw[1].strip()
                default_args[key] = value

        # Make sure the default value of learning rate is correct
        self.assertEqual(ft.default_args['lr'], float(default_args['lr']))

        # Make sure the default value of the dimension is correct
        self.assertEqual(ft.default_args['dim'], int(default_args['dim']))

        # Make sure the default value of ws is correct
        self.assertEqual(ft.default_args['ws'], int(default_args['ws']))

        # Make sure the default value of epoch is correct
        self.assertEqual(ft.default_args['epoch'], int(default_args['epoch']))

        # Make sure the default value of minCount is correct
        self.assertEqual(ft.default_args['minCount'],
                int(default_args['minCount']))

        # Make sure the default value of neg is correct
        self.assertEqual(ft.default_args['neg'], int(default_args['neg']))

        # Make sure the default value of wordNgrams is correct
        self.assertEqual(ft.default_args['wordNgrams'],
                int(default_args['wordNgrams']))

        # Make sure the default value of loss is correct
        self.assertEqual(ft.default_args['loss'], default_args['loss'])

        # Make sure the default value of bucket is correct
        self.assertEqual(ft.default_args['bucket'],
                int(default_args['bucket']))

        # Make sure the default value of minn is correct
        self.assertEqual(ft.default_args['minn'], int(default_args['minn']))

        # Make sure the default value of maxn is correct
        self.assertEqual(ft.default_args['maxn'], int(default_args['maxn']))

        # Make sure the default value of thread is correct
        self.assertEqual(ft.default_args['thread'], int(default_args['thread']))

        # Make sure the default value of lrUpdateRate is correct
        self.assertEqual(ft.default_args['lrUpdateRate'],
                float(default_args['lrUpdateRate']))

        # Make sure the default value of t is correct
        self.assertEqual(ft.default_args['t'], float(default_args['t']))

        # Make sure the default value of label is correct
        self.assertEqual(ft.default_args['label'], default_args['label'])

if __name__ == '__main__':
    unittest.main()
