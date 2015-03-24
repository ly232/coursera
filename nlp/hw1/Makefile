CC = g++
FLAGS += -std=c++11 -I.

all: main.out eval_gene_tagger_dev_p1 eval_gene_tagger_dev_p2 gene_test.p1.out gene_test.p2.out

main.out: hmm.o tagger.o emission_decoder.o viterbi_decoder.o
	$(CC) $(FLAGS) -o main.out main.cc hmm.o tagger.o emission_decoder.o viterbi_decoder.o

hmm.o:
	$(CC) $(FLAGS) -o hmm.o -c hmm/hmm.cc

emission_decoder.o:
	$(CC) $(FLAGS) -o emission_decoder.o -c decoder/emission_decoder.cc

viterbi_decoder.o:
	$(CC) $(FLAGS) -o viterbi_decoder.o -c decoder/viterbi_decoder.cc

tagger.o:
	$(CC) $(FLAGS) -o tagger.o -c tagger/tagger.cc

rare_converter.out:
	$(CC) $(FLAGS) -o rare_converter.out rare_word_converter.cc

gene.rare.train: rare_converter.out
	./rare_converter.out 5 provided/data/gene.train provided/data/gene.rare.train

gene.rare.counts: gene.rare.train
	python provided/script/count_freqs.py provided/data/gene.rare.train >> \
	provided/data/gene.rare.counts

gene_dev.p1.out: main.out gene.rare.counts
	./main.out --count_file provided/data/gene.rare.counts --dev_file provided/data/gene.dev \
	--out_file provided/data/gene_dev.p1.out --decoder emission

gene_test.p1.out: main.out gene.rare.counts
	./main.out --count_file provided/data/gene.rare.counts --dev_file provided/data/gene.test \
	--out_file provided/script/gene_test.p1.out --decoder emission

gene_dev.p2.out: main.out gene.rare.counts
	./main.out --count_file provided/data/gene.rare.counts --dev_file provided/data/gene.dev \
	--out_file provided/data/gene_dev.p2.out --decoder viterbi

gene_test.p2.out: main.out gene.rare.counts
	./main.out --count_file provided/data/gene.rare.counts --dev_file provided/data/gene.test \
	--out_file provided/script/gene_test.p2.out --decoder viterbi

eval_gene_tagger_dev_p1: gene_dev.p1.out
	python provided/script/eval_gene_tagger.py provided/data/gene.key provided/data/gene_dev.p1.out

eval_gene_tagger_dev_p2: gene_dev.p2.out
	python provided/script/eval_gene_tagger.py provided/data/gene.key provided/data/gene_dev.p2.out

clean:
	rm -rf *.out *.o *.stackdump *~ *# provided/data/*.rare.* provided/data/*.out