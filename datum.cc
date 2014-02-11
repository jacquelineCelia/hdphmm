/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./datum.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <cmath>
#include <fstream>
#include <iostream>
#include "datum.h"

Datum::Datum(Config* config) {
    _config = config;
    _corrupted = false;
}

void Datum::ClearSegs() {
    vector<Segment*>::iterator s_iter = _segments.begin();
    for (; s_iter != _segments.end(); ++s_iter) {
        delete *s_iter;
    }
    _segments.clear();
}

int Datum::FindBoundSearchBoundary(int cur_ptr) {
    int next_ptr = FindNextBoundary(cur_ptr) + 1;
    return FindNextBoundary(next_ptr);
}

int Datum::FindNextBoundary(int cur_ptr) {
    int k = cur_ptr;
    while (k < (int) _bounds.size()) {
        if (_bounds[k] -> is_boundary()) {
           return k; 
        }
        ++k;
    }
    return --k;
}

void Datum::Save(const string& dir) {
    int total_frame = 0;
    string filename = dir + _tag;
    // cout << filename << endl;
    ofstream fout(filename.c_str());
    vector<Segment*>::iterator s_iter = _segments.begin();
    for (; s_iter != _segments.end(); ++s_iter) {
        fout << total_frame << " " << total_frame + (*s_iter) -> frame_num() - 1 << " " << (*s_iter) -> id() << endl; 
        total_frame += (*s_iter) -> frame_num();
    }
    fout.close();
}

Datum::~Datum() {
   // delete memory allocated for segment objects
   vector<Segment*>::iterator s_iter;
   s_iter = _segments.begin();
   for (; s_iter != _segments.end(); ++s_iter) {
       delete *s_iter;
   }
   _segments.clear();
   // delete memory allocated for Bound objects
   vector<Bound*>::iterator b_iter;
   b_iter = _bounds.begin();
   for (; b_iter != _bounds.end(); ++b_iter) {
       delete *b_iter;
   }
   _bounds.clear();
}
