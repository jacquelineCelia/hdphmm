/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./gaussian_seed.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef GAUSSIAN_SEED_H
#define GAUSSIAN_SEED_H

#include<vector>

using namespace std;

class GaussianSeed {
 public:
  GaussianSeed(int dim) {
    _dim = dim;
    _mean.resize(_dim);
    _pre.resize(_dim);
  }
  void set_mean(vector<float> mean) {_mean = mean;}
  void set_pre(vector<float> pre) {_pre = pre;}
  void set_dim(int dim) {_dim = dim;}
  vector<float> mean() {return _mean;}
  vector<float> pre() {return _pre;}
  int dim() {return _dim;}
  ~GaussianSeed() {};
 private:
  vector<float> _mean;
  vector<float> _pre;
  int _dim;
};

#endif
