/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./uniformer.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef UNIFORMER_H
#define UNIFORMER_H

#include <mkl.h>

class Uniformer {
 public:
  Uniformer();
  Uniformer(int batch);
  float GetSample();
  int GetDiscreteSample();
  ~Uniformer();
 private:
  float* _samples;
  int* _int_samples;
  int _batch_size;
  int _index;
  int _int_index;
  VSLStreamStatePtr _stream;
};

#endif
