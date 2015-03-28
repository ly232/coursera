/*
 * Coursera NLP HW1
 * Utility binary to convert all rare words to "_RARE_".
 * Sample invocation:
 * ./rare_converter.out 5 provided/data/gene.train provided/data/gene.rare.train
 */

#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <unordered_set>

using namespace std;

bool ContainsNumber(const string& word) {
  return std::string::npos != word.find_first_of("0123456789");
}

bool AllCapitals(const string& word) {
  for (const auto& c : word) {
    if (!isupper(c)) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  int threashold = atoi(argv[1]);
  char* inputfilename = argv[2];
  char* outfilename = argv[3];

  ifstream input(inputfilename);
  string line;
  map<string, int> word_count;
  while (getline(input, line)) {
    string word;
    stringstream ss(line);
    ss >> word;
    if (word_count.find(word) == word_count.end()) word_count[word] = 1;
    else word_count[word] += 1;
  }
  input.close();

  unordered_set<string> rares;
  for (const auto& entry : word_count) {
    if (entry.second < threashold) rares.insert(entry.first);
  }
  ofstream output(outfilename);
  input.open(inputfilename);
  while (getline(input, line)) {
    string word;
    stringstream ss(line);
    ss >> word;
    if (rares.find(word) == rares.end()) {
      output << line << endl;
    } else {
      vector<string> tokens;
      string token;
      while (ss >> token) tokens.push_back(token);
      stringstream newss;
      if (ContainsNumber(word)) {
        newss << "_NUMERIC_";
      } else if (AllCapitals(word)) {
        newss << "_ALLCAPITALS_";
      } else if (isupper(word[word.size() - 1])) {
        newss << "_ENDCAPITAL_";
      } else {
        newss << "_RARE_";
      }
      for (const auto& entry : tokens) newss << " " << entry;
      output << newss.str() << endl;
    }
  }
  output.close();
  return 0;
}
