/*
 * Coursera NLP HW1
 * Simple tagger implementation.
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "tagger/simple_tagger.h"

namespace coursera_nlp {

void SimpleTagger::Load(char* counts_file, char* dev_file) {
  hmm_.Load(counts_file);
  std::string word;
  std::ifstream input(dev_file);
  if (!input.is_open()) {
    throw "SimpleTagger: unable to open file " +
        std::string(dev_file);
  }
  while (getline(input, word)) {
    if (word.size() == 0) continue;  // Skip newline.
    // Compute y* = argmax_y { e(x | y) } by linear scan.
    std::string opt_tag = *hmm_.tags().begin();
    double opt_emission = 0;
    for (const auto& tag : hmm_.tags()) {
      double emission = hmm_.Emission(word, tag);
      if (opt_emission < emission) {
        opt_tag = tag;
        opt_emission = emission;
      }
    }
    cache_[word] = opt_tag;
  }
  input.close();
}

void SimpleTagger::GenerateTaggerOutput(
    char* dev_file, char* out_file) {
  std::ifstream input(dev_file);
  std::ofstream output(out_file);
  if (!input.is_open()) {
    throw "SimpleTagger: unable to open file " +
        std::string(dev_file);
  }
  if (!output.is_open()) {
    throw "SimpleTagger: unable to open file " +
        std::string(out_file);
  }
  std::string word;
  while (getline(input, word)) {
    if (word.size() == 0) {
      output << std::endl;
      continue;
    }
    output << word << " " << GetTag(word) << std::endl;
  }
  input.close();
  output.close();
}

const std::string& SimpleTagger::GetTag(const std::string& word) {
  std::string key = word;
  if (cache_.find(word) == cache_.end()) key = "_RARE_";
  return cache_[key];
}

}  // namespace coursera_nlp

