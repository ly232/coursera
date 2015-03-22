/*
 * Coursera NLP HW1
 * Utilities interface for preprocessing tasks.
 */

#ifndef COURSERA_NLP_HW1_UTILS_H_
#define COURSERA_NLP_HW1_UTILS_H_

#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace coursera_nlp {

/*
 * Parser to build model from training set and compute
 * emission probability.
 */
class InputParser {
public:
  /*
   * Reads input count file generated from count_freqs.py.
   * Each line is assumed with format:
   *   1 WORDTAG I-GENE SNE
   * or format:
   *   6706 3-GRAM O I-GENE O
   */
  void Load(char* filename);
  
  /*
   * Returns emission parameter
   * e(x|y)=Count(word x tagged with y)/Count(y).
   */
  double Emission(std::string x, std::string y);

  /*
   * Returns all possible tag names.
   */
  const std::unordered_set<std::string>& tags();
private:
  /* 
   * Updates  tag_count and word_tag_count.
   * tokens must have this format:
   *   1 WORDTAG I-GENE SNE
   */
  void UpdateWordTagCount(
      const std::vector<std::string>& tokens);

  // Counts tag occurrences.
  std::unordered_map<std::string, int> tag_count_;

  // Counts word-tag occurrences.
  std::map<std::pair<std::string, std::string>, int>
      word_tag_count_;

  // All tags seen in training set.
  std::unordered_set<std::string> tags_;
};

/*
 * A simple tagger that always prouduces the tag:
 * y* = argmax_y { e(x | y) }
 */
class SimpleTagger {
public:
  /* 
   * Load counts_file with InputParser,
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
  // Input parser to build model from training set.
  InputParser input_parser_;

  // Caches simple tagger results (word->tag).
  std::unordered_map<std::string, std::string>  cache_;
};
}  // namespace coursera_nlp

#endif  //COURSERA_NLP_HW1_UTILS_H_