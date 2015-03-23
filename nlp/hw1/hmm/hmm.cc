/*
 * Coursera NLP HW1
 * HMM implementation for training trigram language model.
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>    // std::pair, std::make_pair
#include <vector>

#include "hmm/hmm.h"

namespace coursera_nlp {

void HiddenMarkovModel::Load(char* filename) {
  std::string line;
  std::ifstream input(filename);
  if (!input.is_open()) {
    throw "HiddenMarkovModel: unable to open file " +
        std::string(filename);
  }
  while (getline(input, line)) {
    /*
     * Each line is assumed with format:
     *   1 WORDTAG I-GENE SNE
     * or format:
     *   6706 3-GRAM O I-GENE O
     */
    std::string buf;
    std::stringstream ss(line);
    std::vector<std::string> tokens;
    while (ss >> buf) {
      tokens.push_back(buf);
    }
    if (tokens[1] == "WORDTAG") {
      UpdateWordTagCount(tokens);
    } else {
      UpdateNGramCount(tokens);
    }
  }
  input.close();
}

double HiddenMarkovModel::Emission(std::string x, std::string y) {
  std::pair<std::string, std::string> word_tag_key =
    std::make_pair(x, y);
  if (tag_count_.find(y) == tag_count_.end()) {
    // No tag y appeared in training set.
    return 0;
  } else if (word_count_.find(x) == word_count_.end()) {
    // No word x appeared in training set. Assume x is "_RARE_".
    word_tag_key.first = "_RARE_";
    if (word_tag_count_.find(word_tag_key) == word_tag_count_.end())
      return 0;
  }
  return double(word_tag_count_[word_tag_key]) /
      double(tag_count_[y]);
}

double HiddenMarkovModel::QProb(
    std::string yi, std::string yi2, std::string yi1) {
  std::vector<std::string> trigram_key = {yi2, yi1, yi};
  std::vector<std::string> bigram_key = {yi2, yi1};
  if (n_gram_count_.find(trigram_key) == n_gram_count_.end() ||
      n_gram_count_.find(bigram_key) == n_gram_count_.end()) return 0;
  return double(n_gram_count_[trigram_key]) / double(n_gram_count_[bigram_key]);
}

const std::unordered_set<std::string>& HiddenMarkovModel::tags() {
  return tags_;
}

void HiddenMarkovModel::UpdateWordTagCount(
    const std::vector<std::string>& tokens) {
  /*
   * tokens must have this format:
   *   1 WORDTAG I-GENE SNE
   */
  int count;
  std::istringstream iss(tokens[0]);
  iss >> count;
  const std::string& tag = tokens[2];
  const std::string& word = tokens[3];

  tags_.insert(tag);

  // Updates tag_count_ map:
  if (tag_count_.find(tag) == tag_count_.end()) {
    tag_count_[tag] = count;
  } else {
    tag_count_[tag] += count;
  }

  // Updates word_count_ map:
  if (word_count_.find(word) == word_count_.end()) {
    word_count_[word] = count;
  } else {
    word_count_[word] += count;
  }
  
  // Updates word_tag_count_ map:
  std::pair<std::string, std::string> word_tag_key =
      std::make_pair(word, tag);
  if (word_tag_count_.find(word_tag_key) ==
      word_tag_count_.end()) {
    word_tag_count_[word_tag_key] = count;
  } else {
    word_tag_count_[word_tag_key] += count;
  }
}

void HiddenMarkovModel::UpdateNGramCount(
      const std::vector<std::string>& tokens) {
  /*
   * tokens must have this format:
   *   1 3-GRAM O I-GENE STOP
   */
  int count;
  std::istringstream iss(tokens[0]);
  iss >> count;
  const std::vector<std::string> ngrams(tokens.begin() + 2, tokens.end());
  if (n_gram_count_.find(ngrams) == n_gram_count_.end()) {
    n_gram_count_[ngrams] = count;
  } else {
    n_gram_count_[ngrams] += count;
  }
}
}  // namespace coursera_nlp

