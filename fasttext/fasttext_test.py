import unittest
from os import path

from fasttext import load_model
from model import WordVectorModel

skipgram_file = path.join(path.dirname(__file__),
    '../result/skipgram_text9.bin')

class TestLoadModel(unittest.TestCase):
    def test_load_skipgram_model(self):
        model = load_model(skipgram_file)
        self.assertIsInstance(model, WordVectorModel)
        self.assertEqual(model.model_name, 'skipgram')

if __name__ == '__main__':
    unittest.main()
