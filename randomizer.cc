/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./randomizer.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include "randomizer.h"

using namespace std;

Randomizer::Randomizer() {
    _uniformer = new Uniformer(3000000);
}

void Randomizer::LoadData(vector<Datum*>& data) {
    _data = data;
    _index = _data.size();
}

void Randomizer::Shuffle() {
    // shuffle _data.begin() and _data.begin() + _index
    for (int i = 0; i < _index; ++i) {
        int j = _uniformer -> GetDiscreteSample() % _index;
        Datum* a = _data[i]; 
        _data[i] = _data[j];
        _data[j] = a; 
    }
}

vector<Datum*> Randomizer::GetBatch(int batch_size) {
    batch_size = batch_size > (int) _data.size() ? _data.size() : batch_size;
    if (_index + batch_size - 1 >= (int) _data.size()) {
        // Randomize();
        _index = 0;
    }
    vector<Datum*> batch(_data.begin() + _index, _data.begin() + _index + batch_size);
    _index += batch_size;
    return batch;
}

void Randomizer::Randomize() {
    vector<Datum*> copy_data;
    // Shuffle();
    int ptr = 0;
    if (_index < (int) _data.size()) {
        copy_data.assign(_data.begin() + _index, _data.end());
        ptr = copy_data.size(); 
    }
    copy_data.resize(_data.size());
    copy(_data.begin(), _data.begin() + _index, copy_data.begin() + ptr); 
    _data = copy_data;
    _index = 0;
}

Randomizer::~Randomizer() {
    if (_uniformer != NULL) {
        delete _uniformer;
    }
}
