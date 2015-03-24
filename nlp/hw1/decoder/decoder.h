/*
 * Coursera NLP HW1
 * Generic tagger interface.
 */

#ifndef COURSERA_NLP_HW1_DECODER_H_
#define COURSERA_NLP_HW1_DECODER_H_

#include <vector>

namespace coursera_nlp {

typedef std::vector<std::string> Sentence;
typedef std::vector<std::string> Tags;

/*
 * Decoder interface.
 */
class Decoder {
public:
  Decoder() {};
  virtual ~Decoder() {};
  virtual void Train(char* count_file) = 0;
  virtual void Decode(const Sentence& sentence, Tags* tags) = 0;
};
}  // namespace coursera_nlp

#endif  // COURSERA_NLP_HW1_SIMPLE_TAGGER_H_