from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

# Define the C++ extension
extensions = [
    Extension('*',
        sources=['fasttext/train.pyx',
            "fasttext/cpp/src/args.cc",
            "fasttext/cpp/src/dictionary.cc",
            "fasttext/cpp/src/matrix.cc",
            "fasttext/cpp/src/model.cc",
            "fasttext/cpp/src/utils.cc",
            "fasttext/cpp/src/vector.cc",
            ],
        language='c++',
        extra_compile_args=['-pthread', '-funroll-loops', '-std=c++0x'])
]

# Package details
setup(
    name='fasttext',
    version='0.2.0',
    author='Bayu Aldi Yansyah',
    author_email='bayualdiyansyah@gmail.com',
    url='https://github.com/pyk/fastText.py',
    description='A Python wrapper for Facebook fastText',
    license='BSD 3-Clause License',
    packages=find_packages(),
    ext_modules = cythonize(extensions)
)
