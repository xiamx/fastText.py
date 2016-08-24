# Makefile for building test written in C++
CXX = c++
CXXFLAGS = -pthread -std=c++0x

default: default_params_test default_params_result.txt

args.o: ../fasttext/cpp/args.o
.PHONY: args.o

../fasttext/cpp/args.o:
	$(MAKE) --directory ../fasttext/cpp/

default_params_test: args.o default_params_test.cc
	$(CXX) $(CXXFLAGS) ../fasttext/cpp/args.o \
		default_params_test.cc -o default_params_test

default_params_result.txt: default_params_test
	./default_params_test > default_params_result.txt
