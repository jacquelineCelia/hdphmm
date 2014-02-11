/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./segment.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <cstdlib>
#include <iostream>
#include "segment.h"

void Segment::push_back(Bound* bound) {
    vector<float*> data = bound -> data();
    // need to fix the copy thing
    int ptr = _data.size();
    _data.resize(ptr + data.size());
    copy(data.begin(), data.end(), _data.begin() + ptr); 
    for (int i = bound -> start_frame(); i <= bound -> end_frame(); ++i) {
        _data_index.push_back(i);
    } 
}
