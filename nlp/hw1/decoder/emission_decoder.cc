/*
 * Coursera NLP HW1
 * Emission decoder implementation.
 */

#include "decoder/emission_decoder.h"

namespace coursera_nlp {

void EmissionDecoder::Train(char* count_file) {
  hmm_.reset(new HiddenMarkovModel(count_file));
  for (const auto& word: hmm_->words()) {
    // Compute y* = argmax_y { e(x | y) } by linear scan.
    std::string opt_tag = *(hmm_->tags().begin());
    double opt_emission = 0;
    for (const auto& tag : hmm_->tags()) {
      double emission = hmm_->Emission(word, tag);
      if (opt_emission < emission) {
        opt_tag = tag;
        opt_emission = emission;
      }
    }
    cache_[word] = opt_tag;
  }
}

void EmissionDecoder::Decode(
    const Sentence& sentence, Tags* tags) {
  for (const auto& word : sentence) {
    std::string key = word;
    if (cache_.find(word) == cache_.end()) key = "_RARE_";
    tags->push_back(cache_.at(key));
  }
}
}  // namespace coursera_nlp

