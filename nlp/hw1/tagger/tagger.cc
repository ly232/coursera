/*
 * Coursera NLP HW1
 * Tagger implementation.
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "tagger/tagger.h"
#include "tagger/decoder/emission_decoder.h"

namespace coursera_nlp {

Tagger::Tagger(const std::string& decoder) : decoder_type_ (decoder) {} 

void Tagger::Train(char* training_file) {
  if (decoder_type_ == "emission") decoder_.reset(new EmissionDecoder);
  else throw "Tagger::Train: unrecognized decoder " + decoder_type_;
  decoder_->Train(training_file);
}

void Tagger::Tag(char* input_file, char* output_file) const {
  std::ifstream input(input_file);
  std::ofstream output(output_file);
  if (!input.is_open()) {
    throw "Tagger: unable to open file " + std::string(input_file);
  }
  if (!output.is_open()) {
    throw "Tagger: unable to open file " + std::string(output_file);
  }
  std::string word;
  Sentence sentence;
  Tags tags;
  while (getline(input, word)) {
    if (word.size() == 0) {
      decoder_->Decode(sentence, &tags);
      WriteDecodedTagsResult(sentence, tags, output);
      sentence.clear();
      tags.clear();
      output << std::endl;
      continue;
    }
    sentence.push_back(word);
  }
  input.close();
  output.close();
}

void Tagger::WriteDecodedTagsResult(
    const Sentence& sentence, const Tags& tags, std::ofstream& output) const {
  if (sentence.size() != tags.size()) {
    throw "Tagger::WriteDecodedTagsResult: sentence and tags have different sizes.";
  }
  for (int i = 0; i < sentence.size(); i++) {
    output << sentence[i] << " " << tags[i] << std::endl;
  }
}
}  // namespace coursera_nlp

