/*
 * Coursera NLP HW1
 * Utilities implementation for preprocessing tasks.
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>    // std::pair, std::make_pair
#include <vector>

#include "utils.h"

namespace coursera_nlp {

void InputParser::Load(char* filename) {
  std::string line;
  std::ifstream input(filename);
  if (!input.is_open()) {
    throw "InputParser: unable to open file " +
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
    }
  }
  input.close();
}

double InputParser::Emission(std::string x, std::string y) {
  std::pair<std::string, std::string> word_tag_key =
    std::make_pair(x, y);
  if (tag_count_.find(y) == tag_count_.end()) {
    // No tag y appeared in training set.
    return 0;
  } else if (word_tag_count_.find(word_tag_key) ==
             word_tag_count_.end()) {
    // No word x appeared in training set. Assume x is "_RARE_".
    word_tag_key.first = "_RARE_";
    if (word_tag_count_.find(word_tag_key) == word_tag_count_.end())
      return 0;
  }
  return double(word_tag_count_[word_tag_key]) /
      double(tag_count_[y]);
  
}

const std::unordered_set<std::string>& InputParser::tags() {
  return tags_;
}

void InputParser::UpdateWordTagCount(
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

void SimpleTagger::Load(char* counts_file, char* dev_file) {
  input_parser_.Load(counts_file);
  std::string word;
  std::ifstream input(dev_file);
  if (!input.is_open()) {
    throw "SimpleTagger: unable to open file " +
        std::string(dev_file);
  }
  while (getline(input, word)) {
    if (word.size() == 0) continue;  // Skip newline.
    // Compute y* = argmax_y { e(x | y) } by linear scan.
    std::string opt_tag;
    double opt_emission = 0;
    for (const auto& tag : input_parser_.tags()) {
      double emission = input_parser_.Emission(word, tag);
      if (opt_emission < emission) {
        opt_tag = tag;
        opt_emission = emission;
      }
    }
    cache_[word] = opt_tag;
  }
  input.close();
}

void SimpleTagger::GenerateTaggerOutput(
    char* dev_file, char* out_file) {
  std::ifstream input(dev_file);
  std::ofstream output(out_file);
  if (!input.is_open()) {
    throw "SimpleTagger: unable to open file " +
        std::string(dev_file);
  }
  if (!output.is_open()) {
    throw "SimpleTagger: unable to open file " +
        std::string(out_file);
  }
  std::string word;
  while (getline(input, word)) {
    if (word.size() == 0) {
      output << std::endl;
      continue;
    }
    output << word << " " << GetTag(word) << std::endl;
  }
  input.close();
  output.close();
}

const std::string& SimpleTagger::GetTag(const std::string& word) {
  std::string key = word;
  if (cache_.find(word) == cache_.end()) key = "_RARE_";
  return cache_[key];
}

}  // namespace coursera_nlp

