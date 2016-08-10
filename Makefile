
test:
	python fasttext/fasttext_test.py
.PHONY: test

buildext:
	python setup.py build_ext --inplace
.PHONY: buildext

install:
	python setup.py develop
.PHONY: install

fasttext/cpp/fasttext:
	make --directory fasttext/cpp/

# Test for skipgram model
test/skipgram_params_test.bin:
	./fasttext/cpp/fasttext skipgram -input test/params_test.txt -output \
		test/skipgram_params_test -lr 0.025 -dim 100 -ws 5 -epoch 1 \
		-minCount 5 -neg 5 -loss ns -bucket 2000000 -minn 3 -maxn 6 \
		-thread 4 -lrUpdateRate 100 -t 1e-4

test-skipgram: fasttext/cpp/fasttext test/skipgram_params_test.bin
	python test/skipgram_test.py --failfast --verbose

# Test for cbow model
test/cbow_params_test.bin:
	./fasttext/cpp/fasttext cbow -input test/params_test.txt -output \
		test/cbow_params_test -lr 0.005 -dim 50 -ws 5 -epoch 1 \
		-minCount 3 -neg 5 -loss ns -bucket 2000000 -minn 3 -maxn 6 \
		-thread 4 -lrUpdateRate 100 -t 1e-4

test-cbow: fasttext/cpp/fasttext test/cbow_params_test.bin
	python test/cbow_test.py --failfast --verbose
