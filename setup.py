from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import unittest

# Read the fastText.py version
def read_version():
    with open('fasttext/VERSION') as f:
        return f.read().strip()

# Define the C++ extension
extensions = [
    Extension('*',
        sources=[
            'fasttext/fasttext.pyx',
            'fasttext/interface.cc',
            'fasttext/cpp/src/args.cc',
            'fasttext/cpp/src/dictionary.cc',
            'fasttext/cpp/src/matrix.cc',
            'fasttext/cpp/src/model.cc',
            'fasttext/cpp/src/utils.cc',
            'fasttext/cpp/src/vector.cc'
        ],
        language='c++',
        extra_compile_args=['-pthread', '-funroll-loops', '-std=c++0x'])
]

# Package details
setup(
    name='fasttext',
    version=read_version(),
    author='Bayu Aldi Yansyah',
    author_email='bayualdiyansyah@gmail.com',
    url='https://github.com/pyk/fastText.py',
    description='A Python interface for Facebook fastText library',
    long_description=open('README.rst', 'r').read(),
    license='BSD 3-Clause License',
    packages=['fasttext'],
    ext_modules = cythonize(extensions),
    install_requires=[
        'numpy>=1',
        'future'
    ],
    classifiers= [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
