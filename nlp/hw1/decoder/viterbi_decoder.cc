/*
 * Coursera NLP HW1
 * Viterbi decoder.
 */

#include <unordered_set>
#include <utility>      // std::pair, std::make_pair
#include <iostream>

#include "decoder/viterbi_decoder.h"

namespace coursera_nlp {

void ViterbiDecoder::Train(char* count_file) {
  hmm_.reset(new HiddenMarkovModel(count_file));
}

void ViterbiDecoder::Decode(
    const Sentence& sentence, Tags* tags) {
  int n = sentence.size();
  if (n == 0) return;
  pi_[std::make_pair(0, Ngram({"*", "*"}))] = 1.0;
  for (int k = 1; k <= n; k++) {
    std::unordered_set<std::string> K2, K1, K;
    K2 = K1 = K = hmm_->tags();
    if (k - 2 <= 0) K2 = {"*"};
    if (k - 1 <= 0) K1 = {"*"};
    for (const auto& u : K1) {
      for (const auto& v : K) {
        auto key = std::make_pair(k, Ngram({u, v}));
        double opt_pi = 0;
        std::string opt_bp = "";
        for (const auto& w : K2) {
          double pi = pi_[std::make_pair(k-1, Ngram({w, u}))] *
              hmm_->QProb(v, w, u) * hmm_->Emission(sentence[k-1], v);
          if (pi <= opt_pi) continue;
          opt_pi = pi;
          opt_bp = w;
        }
        pi_[key] = opt_pi;
        bp_[key] = opt_bp;
      }
    }
  }
  std::vector<std::string> y(n+1);
  double opt_pi = 0;
  std::string opt_yn1 = "";
  std::string opt_yn = "";
  std::unordered_set<std::string> Kn1, Kn;
  Kn1 = Kn = hmm_->tags();
  if (n - 1 <= 0) Kn1 = {"*"}; else Kn1 = Kn;
  for (const auto& u : Kn1) {
    for (const auto& v : Kn) {
      double pi = pi_[std::make_pair(n, Ngram({u, v}))] *
          hmm_->QProb("STOP", u, v);
      if (pi < opt_pi) continue;
      opt_pi = pi;
      opt_yn1 = u;
      opt_yn = v;
    }
  }
  y[n - 1] = opt_yn1;
  y[n] = opt_yn;
  for (int k = n-2; k >= 1; k--) {
    y[k] = bp_[std::make_pair(k+2, Ngram({y[k+1], y[k+2]}))];
  }
  tags->clear();
  for (int i = 1; i <= n; i++) {
    tags->push_back(y[i]);
  }
  return;
}
}  // namespace coursera_nlp

