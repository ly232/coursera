/*
 * Coursera NLP HW1
 * Main driver program.
 * Sample invocation:
 * ./main.out --count_file provided/data/gene.rare.counts \
 *            --dev_file provided/data/gene.dev \
 *            --out_file provided/data/gene_dev.p1.out
 */

#include <cstdio>
#include <cstring>
#include <iostream>
 
#include "hmm/hmm.h"
#include "tagger/tagger.h"

int main(int argc, char** argv) {
  try {
    char* count_file;
    char* dev_file;
    char* out_file;
    bool rare = false;
    for (int i=1; i<argc; i++) {
      char* option = argv[i];
      if (!strcmp(option,"--count_file")) {
        count_file = argv[++i];
        continue;
      }
      if (!strcmp(option,"--dev_file")) {
        dev_file = argv[++i];
        continue;
      }
      if (!strcmp(option,"--out_file")) {
        out_file = argv[++i];
        continue;
      }
      if (!strcmp(option,"--help")) {
        std::cout << "Example invocation:" << std::endl;
        std::cout << "main.out --count_file gene.rare.counts "
                  << "--dev_file gene.dev" << std::endl;
        std::cout << "Flags:" << std::endl;
        std::cout << "  --count_file: count file from training set" << std::endl;
        std::cout << "  --dev_file: dev file to verify training" << std::endl;
        return 0;
      }
    }
    coursera_nlp::Tagger emission_tagger("emission");
    emission_tagger.Train(count_file);
    emission_tagger.Tag(dev_file, out_file);
    return 0;
  } catch (std::string e) {
    std::cerr << "Exception: " << e << std::endl;
	  return 1;
  }
}