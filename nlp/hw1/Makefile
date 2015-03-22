CC = g++
FLAGS += -std=c++11 -I. -I./utils

all: main.out rare_converter.out gene.rare.counts

main.out: main.cc utils.o
	$(CC) $(FLAGS) -o main.out main.cc utils.o

utils.o:
	$(CC) $(FLAGS) -c utils/utils.cc

rare_converter.out: rare_word_converter.cc
	$(CC) $(FLAGS) -o rare_converter.out rare_word_converter.cc

gene.rare.counts:
	python provided/script/count_freqs.py provided/data/gene.rare.train >> provided/data/gene.rare.counts

eval_gene_tagger:
	python provided/script/eval_gene_tagger.py provided/data/gene.key provided/data/gene_dev.p1.out

clean:
	rm -rf *.out *.o *.stackdump *~ *# provided/data/gene.rare.counts provided/data/gene_dev.p1.out