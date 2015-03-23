/*
 * Coursera NLP HW1
 * Hidden Markov Model for tri-gram language model.
 */

#ifndef COURSERA_NLP_HW1_HMM_H_
#define COURSERA_NLP_HW1_HMM_H_

#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace coursera_nlp {

/*
 * HMM to build model from training set and compute
 * emission probability.
 */
class HiddenMarkovModel {
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
   * Returns q probability.
   * q(y(i)|y(i-2),y(i-1)) =
   *     Count(y(i-2),y(i-1),y(i))/Count(y(i-2),y(i-1)).
   */
  double QProb(std::string yi, std::string yi2, std::string yi1);

  /*
   * Returns all possible tag names.
   */
  const std::unordered_set<std::string>& tags();
private:
  /* 
   * Updates tag_count_ and word_tag_count_.
   * tokens must have this format:
   *   1 WORDTAG I-GENE SNE
   */
  void UpdateWordTagCount(
      const std::vector<std::string>& tokens);

  /* 
   * Updates n_gram_count_.
   * tokens must have this format:
   *   1 3-GRAM O I-GENE STOP
   */
  void UpdateNGramCount(
      const std::vector<std::string>& tokens);

  // Counts tag occurrences.
  std::unordered_map<std::string, int> tag_count_;

  // Counts word occurrences.
  std::unordered_map<std::string, int> word_count_;

  // Counts word-tag occurrences.
  std::map<std::pair<std::string, std::string>, int>
      word_tag_count_;

  // Counts n-gram occurrences.
  std::map<std::vector<std::string>, int> n_gram_count_;

  // All tags seen in training set.
  std::unordered_set<std::string> tags_;
};
}  // namespace coursera_nlp

#endif  // COURSERA_NLP_HW1_HMM_H_