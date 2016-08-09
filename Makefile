
test:
	python fasttext/fasttext_test.py
.PHONY: test

buildext:
	python setup.py build_ext --inplace
.PHONY: buildext

