/*
 * Coursera NLP HW1
 * Simple tagger only considering emission parameters.
 */

#ifndef COURSERA_NLP_HW1_SIMPLE_TAGGER_H_
#define COURSERA_NLP_HW1_SIMPLE_TAGGER_H_

#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hmm/hmm.h"

namespace coursera_nlp {

/*
 * A simple tagger that always prouduces the tag:
 * y* = argmax_y { e(x | y) }
 */
class SimpleTagger {
public:
  /* 
   * Load counts_file with HiddenMarkovModel,
   * then run simple tagger over dev_file.
   */
  void Load(char* counts_file, char* dev_file);

  /*
   * Writes tagged results to output file. dev_file is needed
   * to re-read the whole file to preserve word ordering.
   */
  void GenerateTaggerOutput(char* dev_file, char* out_file);

  /*
   * Gets optimal tag derived from simple tagger.
   */
  const std::string& GetTag(const std::string& word);
private:
  // HMM for training set.
  HiddenMarkovModel hmm_;

  // Caches simple tagger results (word->tag).
  std::unordered_map<std::string, std::string>  cache_;
};
}  // namespace coursera_nlp

#endif  // COURSERA_NLP_HW1_SIMPLE_TAGGER_H_