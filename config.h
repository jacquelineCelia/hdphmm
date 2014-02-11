#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <string>

#include "gaussian_seed.h"

using namespace std;

class Config {
 public:
  Config();
  ~Config();
  void set_silence(bool mode) {_use_silence = mode;}
  bool Load(string& fn, string& fn_gaussian);
  bool Load(string& fn);
  bool LoadGaussian(string& fn_gaussian);
  bool LoadSeedingMixtures(string&);
  bool parallel() {return _parallel;}
  bool precompute() {return _precompute;} 
  bool UseSilence() {return _use_silence;}
  int prior_for_pi0() {return _initial_0_prior;}
  int state_num() {return _state_num;}
  int max_duration() {return _max_duration;}
  int cluster_num() {return _cluster_num;}
  int mix_num() {return _mix_num;}
  int dim() {return _dim;}
  int weak_limit() {return _weak_limit;}
  int num_sil_mix() {return _num_sil_mix;}
  int num_sil_states() {return _num_sil_states;}
  int num_gaussian_seed() {return _num_mixtures;}
  GaussianSeed mixture(int i) {return _mixtures[i];}
  float sil_self_trans_prob() {return _sil_self_trans_prob;}
  float transition_alpha() {return _transition_alpha;}
  float mix_alpha() {return _mix_alpha;}
  float gaussian_a0() {return _gaussian_a0;}
  float gaussian_k0() {return _gaussian_k0;}
  float alpha_pi() {return _alpha_pi;}
  float cluster_transition_alpha() {return _cluster_transition_alpha;}
  void set_cluster_transition_alpha(float new_a) {_cluster_transition_alpha = new_a;}
  float cluster_transition_gamma() {return _cluster_transition_gamma;}
  vector<float> gaussian_b0() {return _gaussian_b0;}
  vector<float> gaussian_u0() {return _gaussian_u0;}
  void print();
 private:
  int _num_sil_states;
  int _num_sil_mix;
  float _sil_self_trans_prob;
  int _state_num;
  int _max_duration;
  int _cluster_num;
  int _mix_num;
  int _dim;
  int _weak_limit;
  int _num_mixtures;
  int _initial_0_prior;
  float _transition_alpha;
  float _mix_alpha;
  float _gaussian_a0;
  float _gaussian_k0;
  float _alpha_pi; // Alpha for initial probs
  float _cluster_transition_alpha; 
  float _cluster_transition_gamma; // for sampling beta. See Appendix C of [1]
  bool _parallel;
  bool _precompute;
  bool _use_silence;
  vector<float> _gaussian_b0;
  vector<float> _gaussian_u0;
  vector<GaussianSeed> _mixtures;
};

#endif

// Refernces: [1] Emily Fox, Bayesian Nonparametric Learning of Complex Dynamical Phenomena
