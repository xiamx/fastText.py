from .fasttext import skipgram
from .fasttext import cbow
from .fasttext import load_model
from .fasttext import supervised
from .fasttext import default_args

@property
def __VERSION__():
    with open('VERSION') as f:
        return f.read().strip()
