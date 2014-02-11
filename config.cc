/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./config.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <fstream>
#include <string>
#include "config.h"
#include "gaussian_seed.h"

Config::Config() {
    _use_silence = false;
    _initial_0_prior = 0;
}

bool Config::Load(string& fn) {
    ifstream fin(fn.c_str(), ios::in);
    if (!fin.good()) {return false;}
    // 1. num silence state
    fin >> _num_sil_states;
    if (!fin.good()) {return false;}
    // 2. num silence mixture
    fin >> _num_sil_mix;
    if (!fin.good()) {return false;}
    // 3. trans prob
    fin >> _sil_self_trans_prob;
    if (!fin.good()) {return false;}
    // 4. num of state of a regular phone
    fin >> _state_num;
    if (!fin.good()) {return false;}
    // 5. max duration
    fin >> _max_duration; // stored in frame number
    if (!fin.good()) {return false;}
    // 6. num of clusters
    fin >> _cluster_num;
    if (!fin.good()) {return false;}
    // 7. num of mixtures
    fin >> _mix_num;
    if (!fin.good()) {return false;}
    // 8. feature dim
    fin >> _dim;
    if (!fin.good()) {return false;}
    // 9. num of clusters
    fin >> _weak_limit;
    if (_weak_limit != _cluster_num) {
        cout << "Unmatched cluster number and weak limit" << endl;
        return false;
    }
    if (!fin.good()) {return false;} 
    // 10. trans prob alpha
    fin >> _transition_alpha; // n_expected / _num_states
    if (!fin.good()) {return false;}
    // 11. mix prob alpha
    fin >> _mix_alpha;
    if (!fin.good()) {return false;}
    // 12. gassian 0
    fin >> _gaussian_a0;
    if (!fin.good()) {return false;}
    // 13. gaussian k0
    fin >> _gaussian_k0;
    if (!fin.good()) {return false;}
    // 14. alpha pi
    fin >> _alpha_pi;
    if (!fin.good()) {return false;}
    fin >> _cluster_transition_alpha;
    // 15. cluster trans alpha
    if (!fin.good()) {return false;}
    fin >> _cluster_transition_gamma;
    if (!fin.good()) {return false;}
    int parallel_type;
    fin >> parallel_type;
    if (parallel_type == 1) {
        _parallel = true;
    }
    else if (parallel_type == 0) {
        _parallel = false;
    }
    else {
        cout << "Undefined parallel type. Must be either 0 or 1." << endl;
        return false;
    }
    if (!fin.good()) {return false;}
    int precompute_type;
    fin >> precompute_type;
    if (precompute_type == 1) {
        _precompute = true;
    }
    else if (precompute_type == 0) {
        _precompute = false;
    }
    else {
        cout << "Undefined precompute type. Must be either 0 or 1." << endl;
        return false;
    }
    if (!fin.good()) {
        cout << "No initial prior for 0 (silence class)!" << endl;
        cout << "So I'm using [0]" << endl;
    }
    else {
        fin >> _initial_0_prior;
    }
    fin.close();
    return true; 
}

bool Config::Load(string& fn, string& fn_gaussian) {
    if (Load(fn)) {
        return LoadGaussian(fn_gaussian);
    }
    else {
        cout << "Cannot load config file..." <<
            " Check " << fn << " to see whether the format is good." << endl;
        return false;
    }
}

bool Config::LoadGaussian(string& fn_gaussian) {
    ifstream fgaussian(fn_gaussian.c_str(), ios::binary);
    if (!fgaussian.good()) {
        cout << "Cannot load Gaussian Prior" << endl;
        return false;
    }
    cout << "Loading Gaussian: " << fn_gaussian << endl;
    float weight;
    fgaussian.read(reinterpret_cast<char*> (&weight), sizeof(float));
    float mean[_dim];
    float pre[_dim];
    fgaussian.read(reinterpret_cast<char*> (mean), sizeof(float) * _dim);
    fgaussian.read(reinterpret_cast<char*> (pre), sizeof(float) * _dim);
    _gaussian_u0.assign(mean, mean + _dim);
    _gaussian_b0.assign(pre, pre + _dim);
    for (int i = 0; i < _dim; ++i) {
        _gaussian_b0[i] = _gaussian_a0 / _gaussian_b0[i];
    }
    fgaussian.close(); 
    return true;
}

bool Config::LoadSeedingMixtures(string& fn_mixtures) {
    cout << "Must load the regular config file before loading gaussians!" << endl;
    ifstream fmixture(fn_mixtures.c_str(), ios::binary);
    if (!fmixture.good()) {
        return false;
    }
    fmixture.seekg(0, fmixture.end);
    int length = fmixture.tellg();
    fmixture.seekg(0, fmixture.beg);
    // mean/pre vectors + weight counted in bytes
    int size_per_mixture = (_dim * 2 + 1) * sizeof(float);
    if (length % size_per_mixture) {
        cout << "Input format may not match" << endl;
        return false;
    }
    _num_mixtures = length / size_per_mixture;
    for (int i = 0; i < _num_mixtures; ++i) {
        // cout << "Seed " << i << ": " << endl;
        GaussianSeed gs(_dim); 
        float weight;
        fmixture.read(reinterpret_cast<char*> (&weight), sizeof(float));
        vector<float> mean(_dim, 0);
        vector<float> pre(_dim, 0);
        fmixture.read(reinterpret_cast<char*> (&mean[0]), sizeof(float) * _dim);
        fmixture.read(reinterpret_cast<char*> (&pre[0]), sizeof(float) * _dim);
        gs.set_mean(mean);
        for (int d = 0; d < _dim; ++d) {
            // cout << "mean[" << d << "]: " << mean[d] << " ";
            // cout << "pre[" << d << "]: " << pre[d] << " " ;
            pre[d] = _gaussian_a0 / pre[d];
        }
        // cout << endl;
        gs.set_pre(pre);
        _mixtures.push_back(gs);
    }
    fmixture.close();
    return true; 
}

void Config::print() {
    cout << "Silence state: " << _num_sil_states << endl;
    cout << "Silence mix: " << _num_sil_mix << endl;
    cout << "Silence self trans: " << _sil_self_trans_prob << endl;
    cout << "State number: " << _state_num << endl;
    cout << "Max duration: " << _max_duration << endl;
    cout << "Cluster num: " << _cluster_num << endl;
    cout << "Mix num: " << _mix_num << endl;
    cout << "Dim: " << _dim << endl;
    cout << "Weak limit: " << _weak_limit << endl;
    cout << "Transition alaph: " << _transition_alpha << endl;
    cout << "Mix alpha: " << _mix_alpha << endl;
    cout << "Gaussian alpha: " << _gaussian_a0 << endl;
    cout << "Gaussian kappa: " << _gaussian_k0 << endl;
    cout << "Parallel: " << _parallel << endl;
    cout << "Precompute: " << _precompute << endl;
    if (_gaussian_b0.size()) {
        cout << "Gaussian mean: " << endl;
        for (int i = 0; i < _dim; ++i) {
            cout << _gaussian_u0[i] << " ";
        }
        cout << endl;
        cout << "Gaussian pre: " << endl;
        for (int i = 0; i < _dim; ++i) {
            cout << _gaussian_b0[i] << " ";
        }
    }
    cout << "Initial prior for pi0: " << _initial_0_prior << endl;
    cout << endl;
}

Config::~Config() {
}
