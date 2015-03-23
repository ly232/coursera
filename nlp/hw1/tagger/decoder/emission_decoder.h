/*
 * Coursera NLP HW1
 * Simple decoder only considering emission parameters.
 */

#ifndef COURSERA_NLP_HW1_EMISSION_DECODER_H_
#define COURSERA_NLP_HW1_EMISSION_DECODER_H_

#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hmm/hmm.h"
#include "tagger/decoder/decoder.h"

namespace coursera_nlp {

/*
 * A simple decoder that always prouduces the tag:
 * y* = argmax_y { e(x | y) }
 */
class EmissionDecoder : public Decoder {
public:
  EmissionDecoder() {};
  virtual ~EmissionDecoder() {};
  virtual void Train(char* count_file) override;
  virtual void Decode(const Sentence& sentence, Tags* tags) const override;
private:
  // HMM for training set.
  std::unique_ptr<HiddenMarkovModel> hmm_;

  // Caches simple tagger results (word->tag).
  std::unordered_map<std::string, std::string>  cache_;
};
}  // namespace coursera_nlp

#endif  // COURSERA_NLP_HW1_EMISSION_DECODER_H_