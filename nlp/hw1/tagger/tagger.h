/*
 * Coursera NLP HW1
 * Generic tagger interface.
 */

#ifndef COURSERA_NLP_HW1_TAGGER_H_
#define COURSERA_NLP_HW1_TAGGER_H_

#include "decoder/decoder.h"

namespace coursera_nlp {

/*
 * Tagger interface.
 */
class Tagger {
public:
  // Constructs tagger with a decoder.
  Tagger(const std::string& decoder);

  // Trains tagger with given trainig file.
  void Train(char* training_file);

  /*
   * Tags each word in input_file using internal decoder.
   * input_file has 1 word per line, an empty newline indicates
   * end of a sentence. Tagged results are written to output_file,
   * with 2 words per line, 1st is original word, 2nd is tag.
   */
  void Tag(char* input_file, char* output_file) const;
private:
  // Writes (word, tag) pairs to output file.
  void WriteDecodedTagsResult(
      const Sentence& sentence, const Tags& tags, std::ofstream& output) const;

  std::unique_ptr<Decoder> decoder_;

  std::string decoder_type_;
};
}  // namespace coursera_nlp

#endif  // COURSERA_NLP_HW1_SIMPLE_TAGGER_H_