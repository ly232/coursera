/*
 * Coursera NLP HW1
 * Viterbi decoder.
 */

#ifndef COURSERA_NLP_HW1_VITERBI_DECODER_H_
#define COURSERA_NLP_HW1_VITERBI_DECODER_H_

#include <string>
#include <map>
#include <vector>

#include "hmm/hmm.h"
#include "decoder/decoder.h"

namespace coursera_nlp {

typedef std::vector<std::string> Ngram;
typedef std::map<std::pair<int, Ngram>, double> DynamicProgrammingTable;
typedef std::map<std::pair<int, Ngram>, std::string> BackTrackingTable;

class ViterbiDecoder : public Decoder {
public:
  ViterbiDecoder() {};
  virtual ~ViterbiDecoder() {};
  virtual void Train(char* count_file) override;
  virtual void Decode(const Sentence& sentence, Tags* tags) override;
private:
  // HMM for training set.
  std::unique_ptr<HiddenMarkovModel> hmm_;

  // Dynamic programming table.
  DynamicProgrammingTable  pi_;

  // Backtracking pointer table.
  BackTrackingTable bp_;
};
}  // namespace coursera_nlp

#endif  // COURSERA_NLP_HW1_VITERBI_DECODER_H_