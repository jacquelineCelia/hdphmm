#ifndef SAMPLER_H
#define SAMPLER_H

#include <map>
#include <string>
#include "config.h"
#include "datum.h"
#include "uniformer.h"
#include "toolkit.h"
#include "cluster.h"
#include "prob_list.h"
#include "model.h"
#include "counter.h"

class Sampler {
 public:
  Sampler(Config*);
  // Model-wise
  void SampleModelParams(Model*, Counter* = NULL);
  void SampleClusterParams(Cluster*, Cluster* = NULL);
  void SampleInitialProbs(float*, float* = NULL);
  void SampleTransitionBeta(float*, float* = NULL);
  void SampleTransitionA(float*, float**, float** = NULL);
  // Data-wise
  void SampleSegments(Datum*, Model*, Counter*);
  void SampleStateMixtureSeq(Segment*, Cluster*);
  // Data-assist 
  // Note that current implementation still considers words without mapped to any sounds
  void MessageBackward(Datum*, float*, float**, \
          float***, ProbList<int>**, ProbList<int>**);
  void SampleForward(Datum*, ProbList<int>**, ProbList<int>**);
  void ComputeSegProbGivenCluster(Datum*, vector<Cluster*>&, float***);
  bool CheckGammaParams(vector<float>&);
  // Tool-wise
  void RemoveClusterAssignment(Datum*, Counter*);
  void AddClusterAssignment(Datum*, Counter*); 
  int SampleIndexFromLogDistribution(vector<float>);
  int SampleIndexFromDistribution(vector<float>);
  // vector<float> SampleGamma(int, float);
  vector<float> SampleDirFromGamma(int, float*, float = -70000000);
  void SampleDirFromGamma_nonLog(int, float*, float*);
  vector<float> SampleGaussianMean(vector<float>, \
                                   vector<float>, float, int, bool);
  vector<float> SampleGaussianPre(vector<float>, vector<float>, \
                                    float, int, bool);
  float UpdateGammaRate(float, float, float, float, float);
  void SampleAuxiliaryM(float*, float**, float*);
  void SampleAlphaForA(vector<float>, float**);
  ~Sampler();
 private:
  Uniformer* _uniform;
  ToolKit _toolkit;
  Config* _config;
  VSLStreamStatePtr stream; 
};

#endif
