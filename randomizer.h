/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./randomizer.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef RANDOMIZER_H
#define RANDOMIZER_H

#include <vector>
#include "uniformer.h"
#include "datum.h"

class Randomizer {
 public:
  Randomizer();
  void LoadData(vector<Datum*>& data);
  void Randomize();
  void Shuffle();
  vector<Datum*> GetBatch(int);
  ~Randomizer();
 private:
  vector<Datum*> _data;
  vector<int> _random_index;
  int _index;
  Uniformer* _uniformer;
};

#endif
