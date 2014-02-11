/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./toolkit.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef TOOLKIT_H
#define TOOLKIT_H

#include <vector>

using namespace std;

class ToolKit {
 public:
  ToolKit() {};
  ~ToolKit() {};
  float FindLogMax(vector<float>&);
  void MaxRemovedLogDist(vector<float>&);
  int NormalizeDist(vector<float>&);
  float SumLogs(vector<float>&);
  float SumLogs(float, float);
};

#endif
