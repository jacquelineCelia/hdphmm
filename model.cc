/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./model.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "model.h"

#define MIN_PROB_VALUE -70000000

Model::Model(Config* config) {
    _config = config;
    _weak_limit = _config -> weak_limit();
    _A = NULL;
    _pi = NULL;
    _beta = NULL;
}

Model::Model(Model& rhs) {
    _config = rhs.config();
    _weak_limit = rhs.weak_limit();
    _A = new float* [_weak_limit];
    for (int i = 0; i < _weak_limit; ++i) {
        _A[i] = new float [_weak_limit];
        memcpy(_A[i], rhs.A(i), sizeof(float) * _weak_limit);
    }
    _pi = new float [_weak_limit];
    memcpy(_pi, rhs.pi(), sizeof(float) * _weak_limit);
    _beta = new float [_weak_limit];
    memcpy(_beta, rhs.beta(), sizeof(float) * _weak_limit);
    vector<Cluster*> rhs_clusters = rhs.clusters();
    for (int i = 0 ; i < _weak_limit; ++i) {
        Cluster* c = new Cluster(*rhs_clusters[i]);
        _clusters.push_back(c);
    }
}

Model& Model::operator= (Model& rhs) {
    // First, do some cleaning
    if (_A != NULL) {
        for (int i = 0; i < _weak_limit; ++i) {
            delete[] _A[i];
        }
        delete[] _A;
    }
    if (_pi != NULL) {
        delete[] _pi;
    }
    if (_beta != NULL) {
        delete[] _beta;
    }
    for (int i = 0; i < (int) _clusters.size(); ++i) {
        delete _clusters[i];
    }
    _clusters.clear();
    // Assign values of rhs
    _config = rhs.config();
    _weak_limit = rhs.weak_limit();
    _A = new float* [_weak_limit];
    for (int i = 0; i < _weak_limit; ++i) {
        _A[i] = new float [_weak_limit];
        memcpy(_A[i], rhs.A(i), sizeof(float) * _weak_limit);
    }
    _pi = new float [_weak_limit];
    memcpy(_pi, rhs.pi(), sizeof(float) * _weak_limit);
    _beta = new float [_weak_limit];
    memcpy(_beta, rhs.beta(), sizeof(float) * _weak_limit);
    vector<Cluster*> rhs_clusters = rhs.clusters();
    for (int i = 0 ; i < _weak_limit; ++i) {
        Cluster* c = new Cluster(*rhs_clusters[i]);
        _clusters.push_back(c);
    }
    return *this;
}

void Model::Initialize() {
    if (_A != NULL || _pi != NULL || _beta != NULL) {
        cout << "Model has been set, you cannot initialize it" << endl;
        exit(1);
    }
    else {
        _pi = new float [_weak_limit];
        _A = new float* [_weak_limit];
        _beta = new float [_weak_limit];
        for (int i = 0; i < _weak_limit; ++i) {
            _A[i] = new float [_weak_limit];
            for (int j = 0; j < _weak_limit; ++j) {
                _A[i][j] = 0;
            }
            _pi[i] = 0;
        }
        for (int i = _clusters.size() ; i < _weak_limit; ++i) {
            Cluster* c = new Cluster(_config, i);
            _clusters.push_back(c);
        }
    }
}

void Model::AddSilenceCluster(Cluster* cluster) {
    _clusters.insert(_clusters.begin(), cluster);
}

void Model::Save(const string& path) {
    ofstream fout(path.c_str(), ios::binary);
    int num_clusters = _clusters.size(); 
    fout.write(reinterpret_cast<char*> (&num_clusters), sizeof(int));
    fout.write(reinterpret_cast<char*> (_pi), sizeof(float) * num_clusters);
    fout.write(reinterpret_cast<char*> (_beta), sizeof(float) * num_clusters);
    for(int i = 0; i < num_clusters; ++i) {
        fout.write(reinterpret_cast<char*>(_A[i]), sizeof(float) * num_clusters);
    }
    for (int i = 0; i < num_clusters; ++i) {
        _clusters[i] -> Save(fout);
    }
    fout.close();
}

void Model::LoadSnapshot(const string& fn_snapshot) {
    // Clear things up
    if (_A != NULL) {
        for (int i = 0; i < _weak_limit; ++i) {
            delete[] _A[i];
        }
        delete[] _A;
    }
    if (_pi != NULL) {
        delete[] _pi;
    }
    if (_beta != NULL) {
        delete[] _beta;
    }
    for (int i = 0; i < (int) _clusters.size(); ++i) {
        delete _clusters[i];
    }
    _clusters.clear();
    // Read the new model from snapshot
    ifstream fsnapshot(fn_snapshot.c_str(), ios::binary);
    int num_clusters;
    fsnapshot.read(reinterpret_cast<char*> (&num_clusters), sizeof(int));
    cout << "Number of clusters: " << num_clusters << endl;
    _pi = new float [num_clusters];
    fsnapshot.read(reinterpret_cast<char*> (_pi), sizeof(float) * num_clusters);
    _beta = new float [num_clusters];
    fsnapshot.read(reinterpret_cast<char*> (_beta), sizeof(float) * num_clusters);
    _A = new float* [num_clusters];
    for (int i = 0; i < num_clusters; ++i) {
        _A[i] = new float [num_clusters];
        fsnapshot.read(reinterpret_cast<char*>(_A[i]), sizeof(float) * num_clusters);
    }
    for (int i = 0; i < num_clusters; ++i) {
        Cluster* c = new Cluster(_config);
        _clusters.push_back(c);
        c -> Load(fsnapshot);
    }
    fsnapshot.close();
}

void Model::PreCompute(float** _features, int frame_num) {
    vector<Cluster*>::iterator c_iter = _clusters.begin();
    for (; c_iter != _clusters.end(); ++c_iter) {
        (*c_iter) -> PreCompute(_features, frame_num);
    }
}

void Model::ShowA() {
    for (int i = 0; i < (int) _config -> weak_limit(); ++i) {
        for (int j = 0; j < (int) _config -> weak_limit(); ++j) {
            float value = _A[i][j] == MIN_PROB_VALUE ? 0 : exp(_A[i][j]);
            cout << "A[" << i << "][" << j << "]: " << value << endl; 
        }
        cout << endl;
    }
}

Model::~Model() {
    if (_A != NULL) {
        for (int i = 0; i < _weak_limit; ++i) {
            delete[] _A[i];
        }
        delete[] _A;
    }
    if (_pi != NULL) {
        delete[] _pi;
    }
    if (_beta != NULL) {
        delete[] _beta;
    }
    vector<Cluster*>::iterator c_iter = _clusters.begin();
    for (; c_iter != _clusters.end(); ++c_iter) {
        delete *c_iter;
    }
    _clusters.clear();
}
