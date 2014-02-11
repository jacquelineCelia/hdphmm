/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./bound.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include "bound.h"

Bound::Bound(Config* config) {
    _config = config;
    _dim = _config -> dim();
    _is_labeled = false;
    _is_boundary = false;
    _label = -2;
}

void Bound::set_data(float** data, int frame_num) {
    _data = data;
    _frame_num = frame_num;
    for( int i = 0; i < _frame_num; ++i) {
        _data_array.push_back(_data[i]);
    }
}

void Bound::print_data() {
    for (int i = 0; i < (int) _data_array.size(); ++i) {
        for (int j = 0; j < _dim; ++j) {
            cout << _data_array[i][j] << " ";
        }
        cout << endl;
    }
}

Bound::~Bound() {
    for (int i = 0; i < _frame_num; ++i) {
        delete[] _data[i];
    }
    delete[] _data;
}
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./cluster.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <cstdlib>
#include "cluster.h"

#define DEBUG false 
#define SEG_DEBUG true
#define MIN_PROB_VALUE -70000000

Cluster::Cluster(Config* config) {
    _config = config;
    _state_num = _config -> state_num();
    _is_fixed = false;
    _id = -1;
    for (int i = 0; i < _state_num; ++i) {
        GMM emission(_config);
        _emissions.push_back(emission);
        vector<float> trans_prob(_state_num + 1, 0);
        _transition_probs.push_back(trans_prob);
    }
}

Cluster::Cluster(Config* config, int state_num, int mix_num, int id) {
    _config = config;
    _state_num = state_num;
    _id = id;
    _is_fixed = false;
    for (int i = 0; i < _state_num; ++i) {
        GMM emission(_config, mix_num);
        _emissions.push_back(emission);
        vector<float> trans_prob(_state_num + 1, 0);
        _transition_probs.push_back(trans_prob);
    }
}

Cluster::Cluster(Config* config, int id) {
    _config = config;
    _state_num = _config -> state_num();
    _id = id;
    _is_fixed = false;
    for (int i = 0; i < _state_num; ++i) {
        GMM emission(_config);
        _emissions.push_back(emission);
        vector<float> trans_prob(_state_num + 1, 0);
        _transition_probs.push_back(trans_prob);
    }
}

Cluster::Cluster(Cluster& rhs) {
    _id = rhs.id();
    _state_num = rhs.state_num();
    _config = rhs.config();
    _transition_probs = rhs.transition_probs();
    _is_fixed = rhs.is_fixed();
    for (int i = 0; i < _state_num; ++i) {
        _emissions.push_back(rhs.emission(i));
    }
}

void Cluster::set_transition_probs(vector<vector<float> >& trans_prob) {
    _transition_probs = trans_prob;
}

void Cluster::set_emission(GMM& rhs, int index) {
    _emissions[index] = rhs;
}

void Cluster::set_emissions(vector<GMM>& rhs) {
    _emissions = rhs;
}

int Cluster::FindNextBoundary(vector<Bound*>& bounds, int cur_ptr) {
    int k = cur_ptr;
    while (k < (int) bounds.size()) {
        if (bounds[k] -> is_boundary()) {
           return k; 
        }
        ++k;
    }
    return --k;
}

float** Cluster::ConstructSegProbTable(vector<Bound*>& bounds) {
    int b = bounds.size();
    int total_frame_num = 0;
    vector<int> accumulated_frame_nums(b, 0);
    int start_frame = bounds[0] -> start_frame();
    int end_frame = bounds[b - 1] -> end_frame();
    vector<float*> frames(end_frame - start_frame + 1, NULL); 
    for (unsigned int i = 0 ; i < b; ++i) {
        if (!(_config -> precompute())) {
            vector<float*> bound_data = bounds[i] -> data();
            copy(bound_data.begin(), \
                bound_data.end(), frames.begin() + total_frame_num);
        }
        total_frame_num += bounds[i] -> frame_num();
        accumulated_frame_nums[i] = total_frame_num;
    }
    float** frame_prob_for_each_state;
    frame_prob_for_each_state = new float* [_state_num];
    for (int i = 0 ; i < _state_num; ++i) {
        frame_prob_for_each_state[i] = new float[total_frame_num];
        if (!(_config -> precompute())) {
            _emissions[i].ComputeLikehood(frames, frame_prob_for_each_state[i]);
        }
        else {
            _emissions[i].ComputeLikehood(start_frame, end_frame, frame_prob_for_each_state[i]);
        }
    }
    if (DEBUG) {
        cout << "done emission likelihood" << endl;
    }
    float** prob_table;
    prob_table = new float* [b];
    for (unsigned int i = 0 ; i < b; ++i) {
        prob_table[i] = new float [b];
        for (size_t k = 0; k < b; ++k) {
            prob_table[i][k] = MIN_PROB_VALUE;
        }
    }
    ToolKit toolkit;
    int max_duration = _config -> max_duration();
    for (unsigned int i = 0 ; i < b; ++i) {
        if (bounds[i] -> is_labeled()) {
            if (bounds[i] -> label() == _id) {
                // Start computing probs by extending the vad result
                int start_ptr = i;
                while (start_ptr - 1 >= 0 && !(bounds[start_ptr - 1] -> is_labeled())) {
                    start_frame = start_ptr - 1 == 0 ? 0 : accumulated_frame_nums[start_ptr - 1];
                    if (accumulated_frame_nums[i] - start_frame <= max_duration * 2) {
                        --start_ptr;
                    }
                    else {
                        break;
                    }
                }
                int end_ptr = i;
                while (end_ptr + 1 < b && !(bounds[end_ptr + 1] -> is_labeled())) {
                    start_frame = i == 0 ? 0 : accumulated_frame_nums[i - 1];
                    if (accumulated_frame_nums[end_ptr + 1] - start_frame <= max_duration * 2) {
                        ++end_ptr;
                    } 
                    else {
                        break;
                    }
                }
                for (int b1 = start_ptr; b1 <= i; ++b1) {
                    start_frame = b1 == 0 ? 0 : accumulated_frame_nums[b1 - 1];
                    for (int b2 = i; b2 <= end_ptr; ++b2) {
                        end_frame = accumulated_frame_nums[b2] - 1;
                        vector<float> cur_prob(_state_num, MIN_PROB_VALUE);
                        for (int ptr = start_frame; ptr <= end_frame; ++ptr) {
                            if (ptr == start_frame) {
                                cur_prob[0] = frame_prob_for_each_state[0][ptr];
                            }
                            else {
                                vector<float> next_prob(_state_num, 0);
                                for (int k = 0; k < _state_num; ++k) {
                                    vector<float> summands;
                                    for (int l = 0; l <= k; ++l) {
                                        summands.push_back(cur_prob[l] + _transition_probs[l][k]);
                                    }
                                    next_prob[k] = _toolkit.SumLogs(summands) + frame_prob_for_each_state[k][ptr];
                                }
                                cur_prob = next_prob;
                            }
                        }
                        vector<float> next_prob;
                        for (int k = 0; k < _state_num; ++k) {
                            next_prob.push_back(cur_prob[k] + _transition_probs[k][_state_num]);
                        }
                        prob_table[b1][b2] = _toolkit.SumLogs(next_prob);
                    }
                }
            }
        }
        else {
            int j = i;
            start_frame = i == 0 ? 0 : accumulated_frame_nums[i - 1];
            int duration = accumulated_frame_nums[i] - start_frame; 
            int ptr = start_frame;
            vector<float> cur_prob(_state_num, MIN_PROB_VALUE);
            while (ptr < start_frame + duration && j < b \
                    && (duration <= max_duration || (int) i == j) && !(bounds[j] -> is_labeled())) {
                if (ptr == start_frame) {
                    cur_prob[0] = frame_prob_for_each_state[0][ptr];
                }
                else {
                    vector<float> next_prob(_state_num, 0);
                    for (int k = 0; k < _state_num; ++k) {
                        vector<float> summands(k + 1, 0);
                        for (int l = 0; l <= k; ++l) {
                            summands[l] = cur_prob[l] + _transition_probs[l][k]; 
                        }
                        next_prob[k] = toolkit.SumLogs(summands) + frame_prob_for_each_state[k][ptr];
                    }
                    cur_prob = next_prob;
                }
                if (ptr == accumulated_frame_nums[j] - 1) {
                    vector<float> next_prob(_state_num, 0);
                    for (int k = 0; k < _state_num; ++k) {
                        next_prob[k] = cur_prob[k] + _transition_probs[k][_state_num];
                    }
                    prob_table[i][j] = toolkit.SumLogs(next_prob);
                    if (++j < b) {
                        duration = accumulated_frame_nums[j] - start_frame;
                    }
                }
                ++ptr;
            }
        }
    }
    for (int i = 0 ; i < _state_num; ++i) {
        delete[] frame_prob_for_each_state[i];
    }
    delete [] frame_prob_for_each_state;
    return prob_table;
}

ProbList<int>** Cluster::MessageBackwardForASegment(Segment* segment) {
    int frame_num = segment -> frame_num();
    ProbList<int>** B;
    B = new ProbList<int>* [_state_num + 1];
    for (int i = 0 ; i <= _state_num; ++i) {
        B[i] = new ProbList<int> [frame_num + 1];
    }
    // Initialization [need to check what the initial value should be!]
    for (int i = 1; i <= _state_num; ++i) {
        B[i][frame_num].push_back(_transition_probs[i - 1][_state_num], -1);
    }
    // Message Backward
    for (int j = frame_num - 1; j > 0; --j) {
       float* data = segment -> frame(j);
       int data_index = segment -> frame_index(j);
       vector<float> emit_probs(_state_num);
       for (int k = 0; k < _state_num; ++k) {
           float emit_prob = _config -> precompute() ? \
                   _emissions[k].ComputeLikehood(data_index) : \
                   _emissions[k].ComputeLikehood(data);
           emit_probs[k] = emit_prob;
       }
       for (int i = 1; i <= _state_num; ++i) {
           for (int k = i; k <= _state_num; ++k) {
               B[i][j].push_back(_transition_probs[i - 1][k - 1] + \
                   emit_probs[k - 1] + B[k][j + 1].value(), k);
           }
       } 
    }
    float emit_prob = _config -> precompute() ? \
        _emissions[0].ComputeLikehood(segment -> frame_index(0)) : \
        _emissions[0].ComputeLikehood(segment -> frame(0));
    B[0][0].push_back(emit_prob + B[1][1].value(), 1); 
    return B;
}

float Cluster::ComputeSegmentProb(Segment* segment) {
    int frame_num = segment -> frame_num();
    vector<float> prob_s(_state_num, 0);
    for (int i = 0; i < _state_num; ++i) {
        prob_s[i] = _transition_probs[i][_state_num];
    }
    for (int j = frame_num - 2; j >= 0; --j) {
        float* data = segment -> frame(j + 1);
        float data_index = segment -> frame_index(j + 1);
        vector<float> emit_probs(_state_num, 0);
        for (int k = 0; k < _state_num; ++k) {
            float emit_prob = _config -> precompute() ? \
                _emissions[k].ComputeLikehood(data_index) : \
                _emissions[k].ComputeLikehood(data);
            emit_probs[k] = emit_prob;
        }
        vector<float> prob_s_1;
        for (int s_1 = 0; s_1 < _state_num; ++s_1) {
            vector<float> to_sum;
            for (int s = s_1; s < _state_num; ++s) {
                to_sum.push_back(_transition_probs[s_1][s] + \
                        emit_probs[s] + prob_s[s]);
            }
            prob_s_1.push_back(_toolkit.SumLogs(to_sum));
        }
        prob_s = prob_s_1;
    }
    if (_config -> precompute()) {
        return prob_s[0] + _emissions[0].ComputeLikehood(segment -> frame_index(0));
    }
    else {
        return prob_s[0] + _emissions[0].ComputeLikehood(segment -> frame(0));
    }
}

void Cluster::Plus(Segment* segment) {
    if (_is_fixed) {
        return;
    }
    const vector<int> state_seq = segment -> state_seq(); 
    const vector<int> mix_seq = segment -> mix_seq();
    const vector<float*> data = segment -> data();
    if (state_seq.size() != data.size()) {
        cout << "In ClusterCounter::Plus, state_seq and data have different sizes." << endl;
        exit(2);
    }
    else if (mix_seq.size() != data.size()) {
        cout << "In ClusterCounter::Plus, mix_seq and data have different sizes." << endl;
        exit(2);
    }
    else {
        for (int i= 0 ; i < (int) state_seq.size(); ++i) {
            int cur_state = state_seq[i];
            int next_state = i == (int) state_seq.size() - 1 ? \
                             _state_num : state_seq[i + 1];
            ++_transition_probs[cur_state][next_state];
            _emissions[cur_state].Plus(data[i], mix_seq[i]);
        }
    }
}

void Cluster::Minus(Segment* segment) {
    if (_is_fixed) {
        return;
    }
    vector<int> state_seq = segment -> state_seq();
    vector<int> mix_seq = segment -> mix_seq();
    vector<float*> data = segment -> data();
    if (state_seq.size() != data.size()) {
        cout << "In ClusterCounter::Minus, state_seq and data have different sizes." << endl;
        exit(2);
    }
    else if (mix_seq.size() != data.size()) {
        cout << "In ClusterCounter::Minus, mix_seq and data have different sizes." << endl;
        exit(2);
    }
    else {
        for (int i = 0; i < (int) state_seq.size(); ++i) {
            int cur_state = state_seq[i];
            int next_state = i == (int) state_seq.size() - 1 ? \
                             _state_num : state_seq[i + 1];
            --_transition_probs[cur_state][next_state];
            _emissions[cur_state].Minus(data[i], mix_seq[i]);
        }
    }
}


Cluster& Cluster::operator+= (Cluster& rhs) {
    vector<vector<float> > rhs_transition_probs = rhs.transition_probs();
    for (int i = 0 ; i < _state_num; ++i) {
        for (int j = 0; j <= _state_num; ++j) {
            _transition_probs[i][j] += rhs_transition_probs[i][j];
        }
    }
    for (int i = 0; i < _state_num; ++i) {
        _emissions[i] += rhs.emission(i);
    }
    return *this;
}

void Cluster::Save(ofstream& fout) {
    fout.write(reinterpret_cast<char*> (&_id), sizeof(int));
    int fixed = _is_fixed ? 1 : 0;
    fout.write(reinterpret_cast<char*> (&fixed), sizeof(int));
    fout.write(reinterpret_cast<char*> (&_state_num), sizeof(int));
    for (int i = 0; i < _state_num; ++i) {
        fout.write(reinterpret_cast<char*> (&_transition_probs[i][0]), sizeof(float) * (_state_num + 1));
    }
    for (int i = 0 ; i < _state_num; ++i) {
        _emissions[i].Save(fout);
    }
}

void Cluster::Load(ifstream& fin) {
    fin.read(reinterpret_cast<char*> (&_id), sizeof(int));
    int fixed;
    fin.read(reinterpret_cast<char*> (&fixed), sizeof(int));
    _is_fixed = fixed == 1 ? true : false;
    fin.read(reinterpret_cast<char*> (&_state_num), sizeof(int));
    // Initialize space for _transition_probs and _emissions
    for (int i = 0; i < _state_num; ++i) {
        GMM emission(_config);
        _emissions.push_back(emission);
        vector<float> trans_prob(_state_num + 1, 0);
        _transition_probs.push_back(trans_prob);
    }
    for (int i = 0; i < _state_num; ++i) {
        fin.read(reinterpret_cast<char*> (&_transition_probs[i][0]), \
                sizeof(float) * (_state_num + 1));
        for (int j = 0; j <= _state_num; ++j) {
        }
    }
    for (int i = 0; i < _state_num; ++i) {
        _emissions[i].Load(fin);
    }
}

void Cluster::PreCompute(float** data, int frame_num) {
    for (int i = 0; i < _state_num; ++i) {
        _emissions[i].PreCompute(data, frame_num);
    }
}

Cluster::~Cluster() {
}
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
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./counter.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <cstdlib>
#include "counter.h"

Counter::Counter(Config* config) {
    _config = config;
    _weak_limit = _config -> weak_limit();
    for (int i = 0; i < _weak_limit; ++i) {
        Cluster* c = new Cluster(_config, i);
        _cluster_counter.push_back(c);
    }
    _cluster_counter[0] -> set_is_fixed(true);
    _pi = new float [_weak_limit];
    _A = new float* [_weak_limit];
    for (int i = 0; i < _weak_limit; ++i) {
        _pi[i] = 0;
        _A[i] = new float[_weak_limit];
        for (int j = 0; j < _weak_limit; ++j) {
            _A[i][j] = 0;
        }
    }
}

Counter& Counter::operator+= (Counter& rhs) {
    vector<Cluster*> rhs_cluster_counter = rhs.clusters();
    // Sum ClusterCounters
    if (rhs.weak_limit() != _weak_limit) {
        exit(3);
    }
    else {
        // Sum pi stats
        float* rhs_pi = rhs.pi();
        for (int i = 0; i < _weak_limit; ++i) {
            _pi[i] += rhs_pi[i]; 
        }
        float** rhs_A = rhs.A();
        for (int i = 0; i < _weak_limit; ++i) {
            for (int j = 0; j < _weak_limit; ++j) {
                _A[i][j] += rhs_A[i][j];
            }
        }
        for (int i = 0 ; i < _weak_limit; ++i) {
            (*_cluster_counter[i]) += *rhs_cluster_counter[i];
        }
    }
    return *this;
}

Counter::~Counter() {
    if (_A != NULL) {
        for (int i = 0; i < _weak_limit; ++i) {
            delete[] _A[i];
        }
        delete[] _A;
    }
    if (_pi != NULL) {
        delete[] _pi;
    }
    vector<Cluster*>::iterator c_iter = _cluster_counter.begin();
    for (; c_iter != _cluster_counter.end(); ++c_iter) {
        delete *c_iter;
    }
}
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
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./gmm.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include "gmm.h"

GMM::GMM(Config* config) {
    _config = config;
    _mix_num = _config -> mix_num();
    _weight.resize(_mix_num, 0);
    for (int i = 0; i < _mix_num; ++i) {
        Mixture mixture(_config);
        _mixtures.push_back(mixture);
    }
}

GMM::GMM(Config* config, int mix_num) {
    _config = config;
    _mix_num = mix_num;
    _weight.resize(_mix_num, 0);
    for (int i = 0; i < _mix_num; ++i) {
        Mixture mixture(_config);
        _mixtures.push_back(mixture);
    }
}

GMM::GMM(const GMM& rhs) {
    _config = rhs.config();
    _mix_num = rhs.mix_num();
    _mixtures = rhs.mixtures(); 
    _weight = rhs.weight();
}

GMM& GMM::operator= (const GMM& rhs) {
    _config = rhs.config();
    _mix_num = rhs.mix_num();
    _mixtures = rhs.mixtures();
    _weight = rhs.weight();
    return *this;
}

void GMM::set_mixture(Mixture& mixture, int index) {
    _mixtures[index] = mixture;
}

void GMM::set_mixtures(vector<Mixture>& mixtures) {
    _mixtures = mixtures;
}

void GMM::set_weight(vector<float> weight) {
    _weight = weight; 
}

void GMM::Minus(float* data, int index) {
    --_weight[index];
    _mixtures[index].Minus(data);
}

void GMM::Plus(float* data, int index) {
    ++_weight[index];
    _mixtures[index].Plus(data);
}

vector<float> GMM::ComponentLikelihood(float* data) {
    vector<float> likelihood;
    for (int i = 0; i < _mix_num; ++i) {
        likelihood.push_back(_weight[i] + _mixtures[i].likelihood(data));
    } 
    return likelihood;
}

vector<float> GMM::ComponentLikelihood(int index) {
    vector<float> likelihood;
    for (int i = 0; i < _mix_num; ++i) {
        likelihood.push_back(_weight[i] + _mixtures[i].likelihood(index));
    } 
    return likelihood;
}

float GMM::ComputeLikehood(float* data) {
    vector<float> likelihood;
    for (int i = 0; i < _mix_num; ++i) {
        likelihood.push_back(_weight[i] + _mixtures[i].likelihood(data));
    } 
    return _toolkit.SumLogs(likelihood);
}

float GMM::ComputeLikehood(int index) {
    vector<float> likelihood;
    for (int i = 0; i < _mix_num; ++i) {
        likelihood.push_back(_weight[i] + _mixtures[i].likelihood(index));
    }
    return _toolkit.SumLogs(likelihood);
}

void GMM::ComputeLikehood(vector<float*> data, float* likelihood) {
    for (int i = 0; i < (int) data.size(); ++i) {
        likelihood[i] = ComputeLikehood(data[i]);
    }
}

void GMM::ComputeLikehood(int start_frame, int end_frame, float* likelihood) {
    for (int i = start_frame; i <= end_frame; ++i) {
        likelihood[i - start_frame] = ComputeLikehood(i);
    }
}

GMM& GMM::operator+= (GMM& rhs) {
    vector<float> rhs_weight = rhs.weight();
    for (int i = 0; i < _mix_num; ++i) {
        _weight[i] += rhs_weight[i];
        _mixtures[i] += rhs.mixture(i);
    }
    return *this;
}

void GMM::PreCompute(float** data, int frame_num) {
    for (int i = 0; i < _mix_num; ++i) {
        _mixtures[i].PreCompute(data, frame_num);
    }
}

void GMM::Save(ofstream& fout) {
    fout.write(reinterpret_cast<char*> (&_mix_num), sizeof(int));
    fout.write(reinterpret_cast<char*> (&_weight[0]), sizeof(float) * _mix_num);
    for (int m = 0; m < _mix_num; ++m) {
       float det = mixture(m).det();
       vector<float> mean = mixture(m).mean();
       vector<float> pre = mixture(m).pre();
       fout.write(reinterpret_cast<char*> (&det), sizeof(float));
       fout.write(reinterpret_cast<char*> (&mean[0]), sizeof(float) * mean.size());
       fout.write(reinterpret_cast<char*> (&pre[0]), sizeof(float) * pre.size()); 
    }
}

void GMM::Load(ifstream& fin) {
    fin.read(reinterpret_cast<char*> (&_mix_num), sizeof(int));
    fin.read(reinterpret_cast<char*> (&_weight[0]), sizeof(float) * \
            _mix_num);
    for (int m = 0; m < _mix_num; ++m) {
        float det;
        vector<float> mean(_config -> dim(), 0);  
        vector<float> pre(_config -> dim(), 0);
        fin.read(reinterpret_cast<char*> (&det), sizeof(float));
        fin.read(reinterpret_cast<char*> (&mean[0]), sizeof(float) * \
                _config -> dim());
        fin.read(reinterpret_cast<char*> (&pre[0]), sizeof(float) * \
                _config -> dim());
        _mixtures[m].set_det(det);
        _mixtures[m].set_mean(mean);
        _mixtures[m].set_pre(pre);
    }
}

GMM::~GMM() {
}
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./main.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <string>
#include <cstdlib>

#include "config.h"
#include "manager.h"

using namespace std;

void usage();

int main(int argc, char* argv[]) {
    if (argc < 3) {
        usage();
        return 0;
    }
    int mode = atoi(argv[2]);
    if (mode == 0) {
        if (argc != 7) {
            cout << "./gibbs -m 0 -s snapshot -c config" << endl;
            exit(1);
        }
        string fn_snapshot = argv[4];
        string fn_config = argv[6];
        Config config;
        if (!config.Load(fn_config)) {
            cout << "Cannot load configuration file." 
                << " Check " << fn_config << endl;
        }
        else {
            cout << "Configuration file loaded successfully." << endl;
        }
        config.print();
        Manager manager(&config);
        if (fn_snapshot == "") {
            cout << "No file model is specified. Need a previous snapshot file" << endl;
        }
        else {
            manager.InitializeModel(fn_snapshot);
            cout << "Model has shown successfully" << endl;
        }
    }
    else if (mode == 1) {
        if (argc != 19 && argc != 21) {
            usage();
            return -1;
        }
        string fn_list = argv[4];
        string fn_config = argv[6];
        int n_iter = atoi(argv[8]);
        string fn_gaussian = argv[10];
        string basedir = argv[12];
        int batch_size = atoi(argv[14]);
        string fn_gseed = argv[16];
        string fn_sil = ""; 
        string fn_snapshot = "";

        if (argc == 19) {
            fn_sil = argv[18];
        }
        if (argc == 21) {
            fn_sil = argv[18];
            fn_snapshot = argv[20];
        }

        Config config;
        if (!config.Load(fn_config, fn_gaussian)) {
            cout << "Cannot load configuration file." 
                << " Check " << fn_config << endl;
        }
        else {
            cout << "Configuration file loaded successfully." << endl;
        }
        config.print();
        Manager manager(&config);
        if (mode == 0) {
        }
        else if (mode == 1) {
            if (fn_sil != "") {
                if (!manager.LoadSilenceModel(fn_sil)) {
                    cout << "Cannot load silence model " 
                        << "Check " << fn_sil << endl;
                }
                else {
                    config.set_silence(true);
                }
            }
            else {
                cout << "Training model without using silence model" << endl;
            }
            if (!config.LoadSeedingMixtures(fn_gseed)) {
                cout << "Cannot load Gaussian Seeding models" 
                    << " Check " << fn_gseed << endl;
                exit(1);
            }
            if (fn_snapshot == "") {
                manager.InitializeModel();
            }
            else {
                manager.InitializeModel(fn_snapshot);
            }

            if (!manager.LoadData(fn_list)) {
            cout << "Cannot load bounds" 
                << " Check " << fn_list << endl; 
            }
            else {
                cout << "Data loaded successfully." << endl;
            }
            manager.Inference(batch_size, n_iter, basedir);
        }
        else {
            cout << "Undefined mode: [0: read model; 1: training]" << endl;
        }
        cout << "Returning" << endl;
    }
    return 0;
}

void usage() {
    cout << "gibbs -m [0: read model; 1: training] -l data_list -c configuration -n num_iteration " 
        << "-g gaussian_prior -b basedir -z batch_size -sd gaussian_seeds -s silence_model -snapshot snapshot_file" << endl;
}
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./manager.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <set>
#include <omp.h>
#include <ctime>

#include "cluster.h"
#include "gmm.h"
#include "manager.h"
#include "counter.h"
#include "sampler.h"

Manager::Manager(Config* config) {
    _config = config;
    _model = new Model(_config);
    _total_frame_num = 0;
}

bool Manager::LoadSilenceModel(string& fn_sil) {
    ifstream fsil(fn_sil.c_str(), ios::binary);
    if (!fsil.is_open()) {
        return false;
    }
    // load silence model
    int num_state = _config -> num_sil_states();
    int num_mixture = _config -> num_sil_mix();
    Cluster* sil_cluster = new Cluster(_config, num_state, num_mixture, 0); 
    vector<vector<float> > trans_probs;
    for (int i = 0; i < num_state; ++i) {
        vector<float> trans_prob(num_state + 1, 0);
        trans_prob[0]= log(_config -> sil_self_trans_prob());
        trans_prob[1] = log(1 - _config -> sil_self_trans_prob());
        trans_probs.push_back(trans_prob);
    }
    sil_cluster -> set_transition_probs(trans_probs); 
    for (int i = 0 ; i < num_state; ++i) {
        vector<float> weights;
        for (int m = 0; m < num_mixture; ++m) {
            float weight;
            fsil.read(reinterpret_cast<char*> (&weight), sizeof(float));
            weights.push_back(weight);
            vector<float> mean(_config -> dim(), 0);
            vector<float> pre(_config -> dim(), 0);
            fsil.read(reinterpret_cast<char*> (&mean[0]), sizeof(float) * _config -> dim());
            fsil.read(reinterpret_cast<char*> (&pre[0]), sizeof(float) * _config -> dim());
            (sil_cluster -> emission(i)).mixture(m).set_mean(mean);
            (sil_cluster -> emission(i)).mixture(m).set_pre(pre);
            (sil_cluster -> emission(i)).mixture(m).set_det();
        }
        (sil_cluster -> emission(i)).set_weight(weights);
    }
    fsil.close();
    sil_cluster -> set_is_fixed(true);
    _model -> AddSilenceCluster(sil_cluster);
    return true;
}

void Manager::InitializeModel() {
    _model -> Initialize();
    cout << "There are " << (_model -> clusters()).size() << " clusters." << endl;
}

void Manager::InitializeModel(const string& fn_snapshot) {
    _model -> LoadSnapshot(fn_snapshot);
}

void Manager::ParallelInference(int batch_size, int n_iter, const string& basedir) {
    cout << "Doing parallel inference" << endl;
    Counter global_counter(_config);
    Sampler global_sampler(_config);
    // Initialization
    global_sampler.SampleModelParams(_model);

    for (int i = 0; i <= n_iter; ++i) {
        time_t start_time = time(NULL);        
        vector<Datum*> batch = i < 10 ? _datum : randomizer.GetBatch(batch_size);
        LoadDataIntoMatrix(batch);
        if (_config -> precompute()) {
            _model -> PreCompute(&_features[0], _total_frame_num);
        }
        omp_set_num_threads(24);
    #pragma omp parallel 
    { 
        Sampler local_sampler(_config);
        Counter local_counter(_config);
    #pragma omp for schedule(dynamic, 1)
        for (vector<Datum*>::iterator d_iter = batch.begin(); \
                d_iter < batch.end(); ++d_iter) {
            if (!((*d_iter) -> is_corrupted())) {
                local_sampler.SampleSegments((*d_iter), _model, &local_counter); 
            }
            else {
                cout << "!!!!!!!!!! " << (*d_iter) -> tag() << " is corrupted." << endl;  
            }
        }
    #pragma omp critical
        global_counter += local_counter;
    }
        global_sampler.SampleModelParams(_model, &global_counter);
        if (i % 100 == 0) {
            stringstream n;
            n << i;
            string output_dir = basedir + "/" + n.str() + "/";
            SaveData(output_dir);
            SaveModel(output_dir);
        }
        cout << "Done the " << i << "th iteration." << endl;
        time_t end_time = time(NULL);
        cout << "It took " << end_time - start_time << " seconds to finish one iteration" << endl;
    }
}

void Manager::SerielInference(int batch_size, int n_iter, const string& basedir) {
    Counter global_counter(_config);
    Sampler global_sampler(_config);
    // Initialization
    cout << "Sampling global model" << endl;
    global_sampler.SampleModelParams(_model);

    for (int i = 0; i <= n_iter; ++i) {
        time_t start_time = time(NULL); 
        vector<Datum*> batch = i < 10 ? _datum : randomizer.GetBatch(batch_size);
        LoadDataIntoMatrix(batch);
        if (_config -> precompute()) {
            cout << "precomputing" << endl;
            _model -> PreCompute(&_features[0], _total_frame_num);
        }
        for (vector<Datum*>::iterator d_iter = batch.begin(); \
                d_iter != batch.end(); ++d_iter) {
            if (!((*d_iter) -> is_corrupted())) {
                if (i == 0) {
                    cout << "Inferencing " << (*d_iter) -> tag() << endl;
                }
                global_sampler.SampleSegments((*d_iter), _model, &global_counter); 
            }
            else {
                cout << "!!!!!!!!!! " << (*d_iter) -> tag() << " is corrupted." << endl;  
            }
        }
        cout << "Sampling model" << endl;
        global_sampler.SampleModelParams(_model, &global_counter);
        if (i % 1000 == 0) {
            stringstream n;
            n << i;
            string output_dir = basedir + "/" + n.str() + "/";
            SaveData(output_dir);
            SaveModel(output_dir);
        }
        cout << "Done the " << i << "th iteration." << endl;
        time_t end_time = time(NULL); 
        cout << "It took " << end_time - start_time << " seconds to finish one iteration" << endl;
    }
}

void Manager::SaveModel(const string& output_dir) {
    string path = output_dir + "snapshot";
    _model -> Save(path);
}

void Manager::SaveData(const string& output_dir) {
    #pragma omp parallel
    {
        #pragma omp for schedule (dynamic, 1)
        for (int d = 0; d < (int) _datum.size(); ++d) {
            _datum[d] -> Save(output_dir);
        }
    }
}

void Manager::Inference(int batch_size, int n_iter, const string& basedir) {
    if (_config -> parallel()) {
        ParallelInference(batch_size, n_iter, basedir);
    }
    else {
        SerielInference(batch_size, n_iter, basedir);
    }
}

string Manager::GetTag(const string& s) {
   size_t found_last_slash, found_last_period;
   found_last_slash = s.find_last_of("/");
   found_last_period = s.find_last_of(".");
   return s.substr(found_last_slash + 1, \
     found_last_period - 1 - found_last_slash);
}

bool Manager::LoadData(string& fn_data_list) {
    ifstream flist(fn_data_list.c_str());
    while (flist.good()) {
        string fn_index;
        string fn_data;
        flist >> fn_index;
        flist >> fn_data;
        if (fn_index != "" && fn_data != "") {
            string tag = GetTag(fn_data);
            ifstream findex(fn_index.c_str());
            ifstream fdata(fn_data.c_str(), ios::binary);
            if (!findex.is_open()) {
                cout << "Cannot open " << fn_index << endl;
                return false;
            }
            if (!fdata.is_open()) {
                cout << "Cannot open " << fn_data << endl;
                return false;
            }
            cout << "loading " << fn_data << endl;
            Datum* datum = new Datum(_config);
            datum -> set_tag(tag);
            LoadBounds(datum, findex, fdata);
            findex.close();
            fdata.close();
            _datum.push_back(datum);
        }
    }
    flist.close();
    randomizer.LoadData(_datum);
    return true;
}

void Manager::LoadBounds(Datum* datum, ifstream& findex, ifstream& fdata) {
     vector<Bound*> bounds;
     int total_frame;
     findex >> total_frame;
     int start_frame = 0;
     int end_frame = 0;
     int label = -2; // >= 0: is labeled, is boundary. == -1: is boundary. == -2 : normal
     while (end_frame != total_frame - 1) {
         findex >> start_frame;
         findex >> end_frame;
         findex >> label;
         if (start_frame > end_frame) {
            datum -> set_corrupted(true);
            break;
         }
         int frame_num = end_frame - start_frame + 1;
         Bound* bound = new Bound(_config);
         // float** data is deleted in bound.cc
         float** data = new float* [frame_num];
         for (int i = 0; i < frame_num; ++i) {
             data[i] = new float [_config -> dim()];
             fdata.read(reinterpret_cast<char*> (data[i]), \
                     sizeof(float) * _config -> dim());
         }
         bound -> set_data(data, frame_num);
         if (label >= 0) {
             if (bounds.size() > 0) {
                bounds[bounds.size() - 1] -> set_is_boundary(true);
             }
             bound -> set_is_labeled(true);
             bound -> set_is_boundary(true);
             bound -> set_label(label);
         }
         if (label == -1) {
             bound -> set_is_boundary(true);
         }
         bound -> set_start_frame(start_frame + _total_frame_num);
         bound -> set_end_frame(end_frame + _total_frame_num);
         bounds.push_back(bound);
     }
     if (datum -> is_corrupted()) {
         vector<Bound*>::iterator b_iter = bounds.begin();
         for (; b_iter != bounds.end(); ++b_iter) {
            delete *b_iter;
         }
     }
     else {
        _total_frame_num += total_frame;
        bounds[bounds.size() - 1] -> set_is_boundary(true);
        datum -> set_bounds(bounds);
     }
}

void Manager::LoadDataIntoMatrix(vector<Datum*>& batch) {
    _features.clear();
    _total_frame_num = 0;
    vector<Datum*>::iterator d_iter = batch.begin();
    for (; d_iter != batch.end(); ++d_iter) {
        vector<Bound*> bounds = (*d_iter) -> bounds(); 
        vector<Bound*>::iterator b_iter = bounds.begin();
        for (; b_iter != bounds.end(); ++b_iter) {
            vector<float*> f = (*b_iter) -> data(); 
            for (unsigned int i = 0; i < f.size(); ++i) {
                _features.push_back(f[i]);
            }
            (*b_iter) -> set_start_frame(_total_frame_num);
            (*b_iter) -> set_end_frame(_total_frame_num + (*b_iter) -> frame_num() - 1); 
            _total_frame_num += (*b_iter) -> frame_num(); 
        }
    }
}

Manager::~Manager() {
    vector<Datum*>::iterator d_iter = _datum.begin();
    for (; d_iter != _datum.end(); ++d_iter) {
        delete *d_iter;
    }
    _datum.clear();
    delete _model;
}
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./mixture.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <cmath>
#include <cstring>
#include <iostream>
#include <mkl_cblas.h>
#include <mkl_vml.h>

#include "mixture.h"

Mixture::Mixture(Config* config) {
    _config = config;
    _dim = _config -> dim();
    _mean.resize(_dim, 0);
    _pre.resize(_dim, 0);
    _det = 0;
}

Mixture::Mixture(const Mixture& rhs) {
    _mean = rhs.mean();
    _pre = rhs.pre();
    _config = rhs.config();
    _dim = rhs.dim();
    _det = rhs.det();
    _likelihood = rhs.likelihood();
}

Mixture& Mixture::operator = (const Mixture& rhs) {
    _mean = rhs.mean();
    _pre = rhs.pre();
    _config = rhs.config();
    _dim = rhs.dim();
    _det = rhs.det();
    _likelihood = rhs.likelihood();
    return *this;
}

void Mixture::set_mean(vector<float>& mean) {
    _mean = mean;
}

void Mixture::set_pre(vector<float>& pre) {
    _pre = pre;
}

void Mixture::set_det(float det) {
    _det = det;
}

void Mixture::set_det() {
    _det = 0;
    for (int i = 0; i < _dim; ++i) {
        _det += log(_pre[i]);
    }
    _det *= 0.5;
    _det -= 0.5 * _dim * 1.83787622175;
    // log(2*3.1415926) = 1.83787622175
}

float Mixture::likelihood(float* data) {
    float likelihood = 0;
    for (int i = 0; i < _dim; ++i) {
        likelihood += (data[i] - _mean[i]) * (data[i] - _mean[i]) * _pre[i];
    }
    likelihood *= -0.5;
    return _det + likelihood;
}

float Mixture::likelihood(int i) {
    return _likelihood[i]; 
}

void Mixture::Plus(float* data) {
    for(int i = 0 ; i < _dim; ++i) {
        _mean[i] += data[i];
        _pre[i] += data[i] * data[i];
    }
}

void Mixture::Minus(float* data) {
    for (int i = 0; i < _dim; ++i) {
        _mean[i] -= data[i];
        _pre[i] -= data[i] * data[i];
    }
}

void Mixture::PreCompute(float** data, int frame_num) {
    if ((int) _likelihood.size() != frame_num) {
        _likelihood.resize(frame_num);
    }
    float* copy_data = new float [_dim * frame_num];
    float all_ones[frame_num];
    for (int i = 0; i < frame_num; ++i) {
        _likelihood[i] = _det;
        all_ones[i] = 1;
        memcpy(copy_data + i * _dim, data[i], sizeof(float) * _dim);
    }
    MKL_INT m, n, k;
    m = frame_num;
    k = 1;
    n = _dim;
    MKL_INT lda, ldb, ldc;
    float   alpha, beta;
    alpha = -1.0;
    beta = 1.0;
    CBLAS_ORDER order = CblasRowMajor;
    CBLAS_TRANSPOSE transA, transB;
    transA = CblasNoTrans;
    transB = CblasNoTrans;
    lda = 1;
    ldb = _dim;
    ldc = _dim;
    cblas_sgemm(order, transA, transB, m, n, k, alpha, all_ones, lda, \
                                    &_mean[0], ldb, beta, copy_data, ldc);
    vsMul(frame_num * _dim, copy_data, copy_data, copy_data);
    alpha = -0.5;
    MKL_INT icnx = 1, icny = 1;
    cblas_sgemv(order, transA, m, n, alpha, copy_data, n, &_pre[0], icnx, \
                                    beta, &_likelihood[0], icny); 
    delete[] copy_data;
}

Mixture& Mixture::operator+= (Mixture& rhs) {
    vector<float> rhs_mean = rhs.mean();
    vector<float> rhs_pre = rhs.pre();
    for (int i = 0; i < _dim; ++i) {
        _mean[i] += rhs_mean[i];
        _pre[i] += rhs_pre[i];
    }
    return *this;
}

Mixture::~Mixture() {
}
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
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./sampler.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mkl_vml.h>
#include <queue>
#include <cstring>

#include "sampler.h"
#include "bound.h"

#define BRNG VSL_BRNG_MT19937 
#define GAMMA_METHOD VSL_RNG_METHOD_GAMMA_GNORM
#define BETA_METHOD VSL_RNG_METHOD_BETA_CJA
#define UNIFORM_METHOD VSL_RNG_METHOD_UNIFORM_STD
#define GAUSSIAN_METHOD VSL_RNG_METHOD_GAUSSIAN_ICDF 

#define DEBUG false 
#define SEG_DEBUG false 

#define MIN_PROB_VALUE -70000000

using namespace std;

Sampler::Sampler(Config* config) {
    _config = config;
    _uniform = new Uniformer(50000);
    unsigned int SEED = time(0);
    vslNewStream(&stream, BRNG,  SEED);
}

void Sampler::SampleModelParams(Model* model, Counter* counter) {
    // Sample Cluster parameters
    // TODO: Add sample for _A and _pi
    vector<Cluster*> clusters = model -> clusters();
    if (counter != NULL) {
        SampleInitialProbs(model -> pi(), counter -> pi());
        vector<float> m(_config -> weak_limit(), 0);
        SampleAuxiliaryM(model -> beta(), counter -> A(), &m[0]);
        SampleAlphaForA(m, counter -> A());
        SampleTransitionBeta(model -> beta(), &m[0]);
        SampleTransitionA(model -> beta(), model -> A(), counter -> A());
        vector<Cluster*> cluster_counter = counter -> clusters();
        if (clusters.size() != cluster_counter.size()) {
            cout << "Clusters don't have the right number of ClusterCounters" << endl;
            exit(4);
        }
        else {
            for (int  i = 0 ; i < (int) clusters.size(); ++i) {
                SampleClusterParams(clusters[i], cluster_counter[i]);
            }
        }
    }
    else {
        SampleInitialProbs(model -> pi());
        SampleTransitionBeta(model -> beta());
        SampleTransitionA(model -> beta(), model -> A());
        for (int i = 0; i < (int) clusters.size(); ++i) {
            SampleClusterParams(clusters[i]);
        }
    }
}

void Sampler::SampleInitialProbs(float* pi, float* counter_pi) {
    int n = _config -> weak_limit();
    vector<float> alpha_with_data(n, _config -> alpha_pi());
    if (counter_pi != NULL) {
        for (size_t i = 0; i < n; ++i) {
            alpha_with_data[i] += counter_pi[i];
        }    
    }
    alpha_with_data[0] += _config -> prior_for_pi0();
    vector<float> new_pi = SampleDirFromGamma(n, &alpha_with_data[0]); 
    memcpy(pi, &new_pi[0], sizeof(float) * n);
}

void Sampler::SampleAlphaForA(vector<float> m, float** counter_A) {
    int num_clusters = _config -> weak_limit();
    vector<float> n(num_clusters, 0);
    for (size_t i = 0; i < num_clusters; ++i) {
        for (size_t j = 0; j < num_clusters; ++j) {
            n[i] += counter_A[i][j];
        }
    }
    vector<float> w(num_clusters, 0);
    float alpha0 = _config -> cluster_transition_gamma(); 
    for (size_t i = 0; i < num_clusters; ++i) {
        if (n[i]) {
            while (!w[i]) {
                if(vsRngBeta(BETA_METHOD, stream, 1, &w[i], 1 + alpha0, n[i], 0, 1) != VSL_STATUS_OK) {
                    cerr << "Error in vsRngBeta" << endl;
                    cerr<< "parameters: a = 0, p = " << 1 + alpha0 << ", beta = 1, q = " << n[i] << endl;
                }
            }
        }
    }
    vector<float> s(num_clusters, 0);
    for (size_t i = 0; i < num_clusters; ++i) {
        if (n[i]) {
            s[i] = (_uniform -> GetSample() < (n[i] / (n[i] + alpha0)));
        }
    }
    float total_m = 0;
    float total_s = 0;
    float total_logw = 0;
    for (size_t i = 0; i < num_clusters; ++i) {
        total_m += m[i];
        if (n[i]) {
            total_s += s[i];
            if (!w[i]) {
                total_logw += -100;
            }
            else {
                total_logw += log(w[i]);
            }
        }
    }
    float new_alpha0;
    if (vsRngGamma(GAMMA_METHOD, stream, 1, \
                &new_alpha0, 1 + total_m - total_s, 0, 1 - total_logw) != VSL_STATUS_OK) {
        cerr << "Error happed in vsRngGamma when sampling new alpha0" << endl;
        cerr << "Parameters: an = " << 1 + total_m - total_s << " and bn = " << 1 - total_logw << endl;
    }
    _config -> set_cluster_transition_alpha(new_alpha0); 
}

void Sampler::SampleAuxiliaryM(float* beta, float** counter_A, float* m) {
        // Sample auxiliary variables
        int num_clusters = _config -> weak_limit(); 
        float** M = new float* [num_clusters];
        for (int i = 0; i < num_clusters; ++i) {
            M[i] = new float [num_clusters];
            for (int j = 0; j < num_clusters; ++j) {
                M[i][j] = 0;
            }
        }
        float alpha = _config -> cluster_transition_alpha();
        for (int i = 0; i < num_clusters; ++i) {
            for (int j = 0; j < num_clusters; ++j) {
                for (int n = 0; n < counter_A[i][j]; ++n) {
                    M[i][j] += (_uniform -> GetSample() <= (alpha * beta[j] / (n + alpha * beta[j]))); 
                }
            }
        }
        for (int j = 0; j < num_clusters; ++j) {
            m[j] = 0;
            for (int i = 0; i < num_clusters; ++i) {
                m[j] += M[i][j];
            }
        }
        for (int i = 0; i < num_clusters; ++i) {
            delete[] M[i];
        }
        delete[] M;
}

void Sampler::SampleTransitionBeta(float* beta, float* m) {
    int num_clusters = _config -> weak_limit();
    if (m != NULL) {
        // Sample beta
        vector<float> gamma(num_clusters, _config -> cluster_transition_alpha());
        for (int i = 0; i < num_clusters; ++i) {
            gamma[i] += m[i];
        }
        SampleDirFromGamma_nonLog(num_clusters, &gamma[0], beta);
    }
    else {
        // Sample from prior
        vector<float> gamma(num_clusters, _config -> cluster_transition_gamma());
        SampleDirFromGamma_nonLog(num_clusters, &gamma[0], beta);
    }
}

void Sampler::SampleTransitionA(float* beta, float** A, float** counter_A) {
    int num_clusters = _config -> weak_limit();
    vector<float> alpha(beta, beta + num_clusters);
    for (int i = 0; i < num_clusters; ++i) {
        alpha[i] *= _config -> transition_alpha();
        if (alpha[i] == 0) {
            alpha[i] = pow(0.1, 20);
        }
    }
    if (counter_A != NULL) {
        // Sample A
        for (int i = 0; i < num_clusters; ++i) {
            vector<float> alpha_with_data = alpha;
            for (int j = 0; j < num_clusters; ++j) {
                alpha_with_data[j] += counter_A[i][j];
            }
            if (!CheckGammaParams(alpha_with_data)) {
                cerr << "Note: gamma parameters are invalid when sampling for A." << endl;
                for (size_t z = 0; z < alpha_with_data.size(); ++z) {
                    cerr << "alpha_data_count[" << z << "]: " << alpha_with_data[z] << " ";
                }
                cerr << endl;
            }
            vector<float> a = SampleDirFromGamma(num_clusters, &alpha_with_data[0], -10); 
            memcpy(A[i], &a[0], sizeof(float) * num_clusters);
        }
    }
    else {
        // sample A (make it uniform at the beginning)
        for (int i = 0; i < num_clusters; ++i) {
            vector<float> a(num_clusters, log(1.0 / num_clusters));
            memcpy(A[i], &a[0], sizeof(float) * num_clusters);
        }
    }
}

bool Sampler::CheckGammaParams(vector<float>& a) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] <= 0) {
            return false;
        }
    }
    return true;
}

/*
vector<float> Sampler::SampleGamma(int n, float alpha) {
   vector<float> samples(n, 0);
   vsRngGamma(GAMMA_METHOD, stream, n, &samples[0], alpha, 0, 1);
   return samples;
}
*/

void Sampler::SampleDirFromGamma_nonLog(int n, float* in, float* out) {
    float total = 0;
    for (int i = 0; i < n; ++i) {
        if (vsRngGamma(GAMMA_METHOD, stream, 1, &out[i], in[i], 0, 1) != VSL_STATUS_OK) {
           cout << "Error when calling SampleDirFromGamma_nonLog" << endl;
           cout << "The parameters are:" << endl;
           cout << "an: " << in[i] << " bn: 1" << endl; 
           exit(1);
        }
        else {
            total += out[i];
        }
    }
    if (!total) {
        out[0] = 1;
    }
    else {
        for (int i = 0; i < n; ++i) {
            out[i] /= total;
        }
    }
}

vector<float> Sampler::SampleDirFromGamma(int n, float* alpha, float min) {
    vector<float> samples(n, 0);
    bool need_to_set_sparse = true;
    for (int i = 0; i < n; ++i) {
        if (vsRngGamma(GAMMA_METHOD, stream, 1, &samples[i], alpha[i], 0, 1) != VSL_STATUS_OK) {
           cout << "Error when calling SampleDirFromGamma" << endl;
           cout << "The parameters are:" << endl;
           cout << "an: " << alpha[i] << " bn: 1" << endl; 
           exit(1);
        }
        samples[i] = samples[i] == 0 ? min : log(samples[i]);
        if (samples[i] < min) {
            samples[i] = min;
        }
        if (samples[i] > min) {
            need_to_set_sparse = false;
        }
    }
    if (need_to_set_sparse) {
        samples[0] = 0;
    }
    float sum = _toolkit.SumLogs(samples); 
    for (int i = 0; i < n; ++i) {
        samples[i] -= sum; 
    }
    return samples;
}

void Sampler::SampleClusterParams(Cluster* cluster, Cluster* counter) {
    if (cluster -> is_fixed()) {
        return;
    }
    int state_num_ = _config -> state_num();
    int mix_num_ = _config -> mix_num();
    vector<vector<float> > trans_probs;
    if (counter == NULL) {
        for (int i = 0; i < state_num_; ++i) {
            vector<float> alpha(state_num_ + 1 - i, _config -> transition_alpha());
            if (!CheckGammaParams(alpha)) {
                cerr << "Note: gamma parameters are invalid when sampling for trans_probs from prior." << endl;
                for (size_t z = 0; z < alpha.size(); ++z) {
                    cerr << "trans alpha[" << z << "]: " << alpha[z] << " ";
                }
                cerr << endl;
            }
            vector<float> trans_prob = SampleDirFromGamma(state_num_ + 1 - i, &alpha[0]);
            for (int j = 0; j < i; ++j) {
                trans_prob.insert(trans_prob.begin(), MIN_PROB_VALUE);
            }
            trans_probs.push_back(trans_prob);
        }  
        cluster -> set_transition_probs(trans_probs);
        // Sample each GMM from prior (mixture weight and Gaussian)

        for (int i = 0; i < state_num_; ++i) {
            // cout << "For cluster " << cluster -> id() << " state " << i << endl;
            vector<float> alpha(mix_num_, _config -> mix_alpha());
            if (!CheckGammaParams(alpha)) {
                cerr << "Note: gamma parameters are invalid when sampling for weight from prior." << endl;
                for (size_t z = 0; z < alpha.size(); ++z) {
                    cerr << "weight alpha[" << z << "]: " << alpha[z] << " ";
                }
                cerr << endl;
            }
            // vector<float> mix_weight = SampleDirFromGamma(mix_num_, &alpha[0], -5); 
            vector<float> mix_weight(mix_num_, -log(mix_num_)); 
            for (int j = 0; j < mix_num_; ++j) {
                // cout << "mix_weight[" << j << "]: " << mix_weight[j] << " ";
                vector<float> mean_count(_config -> dim(), 0);
                vector<float> pre_count(_config -> dim(), 0);
                vector<float> pre = SampleGaussianPre(mean_count, pre_count, 0, cluster -> id(), false);
                vector<float> mean = SampleGaussianMean(pre, mean_count, 0, cluster -> id(), false);
                (cluster -> emission(i)).mixture(j).set_mean(mean);
                (cluster -> emission(i)).mixture(j).set_pre(pre);
                (cluster -> emission(i)).mixture(j).set_det();
            }
            // cout << endl;
            (cluster -> emission(i)).set_weight(mix_weight);
        }
    }
    else {
        vector<vector<float> > trans_probs;
        vector<vector<float> > trans_counts = counter -> transition_probs();
        for (int i = 0; i < state_num_; ++i) {
            for (int j = 0; j < state_num_ + 1; ++j) {
                trans_counts[i][j] += _config -> transition_alpha();
            }
            if (!CheckGammaParams(trans_counts[i])) {
                cerr << "Note: gamma parameters are invalid when sampling for trans_probs." << endl;
                for (size_t j = 0; j < trans_counts[i].size(); ++j) {
                    cerr << "trans_count[" << i << "][" << j << "]: " << trans_counts[i][j] << " ";
                }
                cerr << endl;
            }
            vector<float> trans_prob = SampleDirFromGamma(state_num_ + 1 - i, &trans_counts[i][i]);
            for (int j = 0; j < i; ++j) {
                trans_prob.insert(trans_prob.begin(), MIN_PROB_VALUE);
            }
            trans_probs.push_back(trans_prob);
        }
        cluster -> set_transition_probs(trans_probs);
        for (int i = 0; i < state_num_; ++i) {
            vector<float> weight_count = (counter -> emission(i)).weight();
            for (int j = 0; j < mix_num_; ++j) {
                // Sample Gaussian
                vector<float> mean_count = (counter -> emission(i)).mixture(j).mean();
                vector<float> pre_count = (counter -> emission(i)).mixture(j).pre();
                vector<float> pre = SampleGaussianPre(mean_count, pre_count, weight_count[j], cluster -> id(), false);
                vector<float> mean = SampleGaussianMean(pre, mean_count, weight_count[j], cluster -> id(), false);
                (cluster -> emission(i)).mixture(j).set_mean(mean);
                (cluster -> emission(i)).mixture(j).set_pre(pre);
                (cluster -> emission(i)).mixture(j).set_det();
                weight_count[j] += _config -> mix_alpha();
            }
            if (!CheckGammaParams(weight_count)) {
                cerr << "Note: gamma parameters are invalid when sampling for weight." << endl;
                for (size_t z = 0; z < weight_count.size(); ++z) {
                    cerr << "weight_count[" << z << "]: " << weight_count[z] << " ";
                }
                cerr << endl;
            }
            vector<float> weight = SampleDirFromGamma(mix_num_, &weight_count[0], -5);
            (cluster -> emission(i)).set_weight(weight);
        }
    }
}

float Sampler::UpdateGammaRate(float b0, float x2, float x, float n, float u0) {
    float k0 = _config -> gaussian_k0();
    if (n == 0) {
        return b0;
    }
    else {
        float value = b0 + 0.5 * (x2 - x * x / n) + (k0 * n * (x / n - u0) * (x / n - u0))/(2 * (k0 + n));
        if (value > 0) {
            return value;
        } 
        else {
            return b0 + (k0 * n * (x / n - u0) * (x / n - u0))/(2 * (k0 + n));
        }
    }
}

vector<float> Sampler::SampleGaussianPre(vector<float> mean_count, \
        vector<float> pre_count, float n, int id, bool strong_seed) {
    vector<float> gaussian_b0;
    vector<float> gaussian_u0;
    vector<float> new_pre(_config -> dim(), 1);
    int true_id = _config -> UseSilence() ? id - 1 : id;
    if (true_id < _config -> num_gaussian_seed()) {
        gaussian_b0 = (_config -> mixture(true_id)).pre();
        gaussian_u0 = (_config -> mixture(true_id)).mean();
        if (strong_seed) {
            for (int i = 0; i < _config -> dim(); ++i) {
                new_pre[i] = (_config -> gaussian_a0()) / gaussian_b0[i]; 
            }
            return new_pre;
        }
    }
    else {
        gaussian_b0 = _config -> gaussian_b0();
        gaussian_u0 = _config -> gaussian_u0();
    }
    for (int i = 0; i < _config -> dim(); ++i) {
        float bn = UpdateGammaRate(\
                gaussian_b0[i], pre_count[i], mean_count[i], \
                n, gaussian_u0[i]);
        float an = _config -> gaussian_a0() + n / 2;
        if (vsRngGamma(GAMMA_METHOD, stream, 1, &new_pre[i], an, 0, 1 / bn) != VSL_STATUS_OK) {
            cout << "Error when calling SampleGaussianPre" << endl;
            cout << "The parameters are: " << endl;
            cout << "an: " << an << " bn: " << bn << endl;
            exit(1);
        } 
    }
    return new_pre;
}

vector<float> Sampler::SampleGaussianMean(vector<float> pre, \
        vector<float> count, float n, int id, bool strong_seed) {
    int true_id = _config -> UseSilence() ? id - 1 : id;
    vector<float> new_mean(_config -> dim(), 0);
    float k0 = _config -> gaussian_k0();
    vector<float> gaussian_u0; 
    if (true_id < _config -> num_gaussian_seed()) {
        gaussian_u0 = (_config -> mixture(true_id)).mean();
        if (strong_seed) {
            new_mean = gaussian_u0;
            return new_mean;
        }
        k0 = 500;
    }
    else {
        gaussian_u0 = _config -> gaussian_u0();
    }
    for (int i = 0; i < _config -> dim(); ++i) {
        float un = (k0 * gaussian_u0[i] + count[i]) / (k0 + n);
        float kn = k0 + n;
        float std = sqrt(1 / (kn * pre[i]));
        vsRngGaussian(GAUSSIAN_METHOD, stream, 1, &new_mean[i], un, std); 
    }
    return new_mean;
}


void Sampler::SampleSegments(Datum* datum, \
                     Model* model, Counter* counter) {
    vector<Cluster*> cluster = model -> clusters();
    int b = (datum -> bounds()).size();
    if ((datum -> segments()).size()) {
        RemoveClusterAssignment(datum, counter);
    }
    // Compute P(b|c) 
    if (SEG_DEBUG) {
        cout << "Computing P(b|c)" << endl;
    }
    float*** segment_prob_given_cluster = new float** [_config -> weak_limit()];
    ComputeSegProbGivenCluster(datum, cluster, segment_prob_given_cluster);
    // Compute B and Bstar
    if (SEG_DEBUG) {
        cout << "Allocating space for B and Bstar" << endl;
    }
    int c = _config -> weak_limit();
    ProbList<int> **Bstar, **B; 
    B = new ProbList<int>* [c + 1];
    Bstar = new ProbList<int>* [c + 1];
    for (int i = 0 ; i <= c; ++i) {
        B[i] = new ProbList<int> [b + 1];
        Bstar[i] = new ProbList<int> [b + 1];
    }
    if (SEG_DEBUG) {
        cout << "Message Backward" << endl;
    }
    MessageBackward(datum, model -> pi(), model -> A(), \
            segment_prob_given_cluster, B, Bstar);
    // Sample forward
    if (SEG_DEBUG) {
        cout << "Sample Forward" << endl;
    }
    SampleForward(datum, B, Bstar);
    // Sample State sequence and Mixture ID for segments
    if (SEG_DEBUG) {
        cout << "Sampling state and mixture seq" << endl;
    }
    vector<Segment*> segments = datum -> segments();
    for (int i = 0; i < (int) segments.size(); ++i) {
        if (SEG_DEBUG) {
            cout << "Sampling: " << i << " out of " << segments.size() << endl;
            cout << "id: " << segments[i] -> id() << endl;
        }
        SampleStateMixtureSeq(segments[i], cluster[segments[i] -> id()]);
    }
    // Add newly sampled alignment results
    if (SEG_DEBUG) {
        cout << "Adding cluster assignement" << endl;
    }
    AddClusterAssignment(datum, counter);
    // Delete allocated memory: B, Bstar, ProbList, segment_prob_given_cluster
    if (DEBUG) {
        cout << "deleting B and Bstar" << endl;
    }
    for (int i = 0; i <= c; ++i) {
        delete[] B[i];
        delete[] Bstar[i];
    }
    delete[] B;
    delete[] Bstar;
    if (DEBUG) {
        cout << "deleting segment_prob_given_cluster" << endl;
    }
    for (int i = 0; i < _config -> weak_limit(); ++i) {
        for (int j = 0; j < b; ++j) {
            delete[] segment_prob_given_cluster[i][j];
        }
        delete[] segment_prob_given_cluster[i]; 
    }
    delete[] segment_prob_given_cluster;
}

void Sampler::SampleStateMixtureSeq(Segment* segment, Cluster* cluster) {
    if (cluster -> is_fixed()) {
        return;
    }
    // Message Backward
    ProbList<int> **B = cluster -> MessageBackwardForASegment(segment);
    // Sample Forward
    vector<int> state_seq;
    vector<int> mix_seq;
    int s;
    int m;
    for (int i = 0, j = 0; j < segment -> frame_num(); ++j) {
        s = B[i][j].index(SampleIndexFromLogDistribution(B[i][j].probs()));
        state_seq.push_back(s - 1);
        if (!(_config -> precompute())) {
            m = SampleIndexFromLogDistribution((cluster -> emission(s - 1)).\
                ComponentLikelihood(segment -> frame(j)));
        }
        else {
            m = SampleIndexFromLogDistribution((cluster -> emission(s - 1)).\
                ComponentLikelihood(segment -> frame_index(j)));
        }
        mix_seq.push_back(m);
        i = s;
    }
    segment -> set_state_seq(state_seq);
    segment -> set_mix_seq(mix_seq);
    // Delete B
    for (int i = 0; i <= cluster -> state_num(); ++i) {
        delete[] B[i];
    }
    delete[] B;
}

int Sampler::SampleIndexFromLogDistribution(vector<float> log_probs) {
    _toolkit.MaxRemovedLogDist(log_probs);
    return SampleIndexFromDistribution(log_probs);
}

int Sampler::SampleIndexFromDistribution(vector<float> probs) {
   float sum = _toolkit.NormalizeDist(probs);
   // sample a random number from a uniform dist [0,1]
   float random_unit_sample = _uniform -> GetSample(); 
   while (random_unit_sample == 1 || random_unit_sample == 0) {
       random_unit_sample = _uniform -> GetSample(); 
   }
   // figure out the index 
   int index = 0; 
   sum = probs[index];
   while (random_unit_sample > sum) {
       if (++index < (int) probs.size()) {
           sum += probs[index];
       }
       else {
           break;
       }
   }
   if (index >= probs.size()) {
       index = probs.size() - 1;
   }
   return index;
}

void Sampler::ComputeSegProbGivenCluster(Datum* datum, \
                            vector<Cluster*>& clusters, \
                            float*** segment_prob_given_cluster) {
    for (int i = 0; i < (int) clusters.size(); ++i) {
        if (DEBUG) {
            cout << "cluster " << i << endl;
        }
        segment_prob_given_cluster[i] = \
            clusters[i] -> ConstructSegProbTable(datum -> bounds());
        if (DEBUG) {
            cout << "For Cluster: " << i << endl;
            int b = (datum -> bounds()).size();
            for (int j = 0; j < b; ++j) {
                for (int k = 0; k < b; ++k) {
                   cout << "b[" << j << "][" << k << "]: " << segment_prob_given_cluster[i][j][k] << " "; 
                }
                cout << endl;
            }
            cout << endl;
        }
    }
}

void Sampler::SampleForward(Datum* datum, \
            ProbList<int>** B, ProbList<int>** Bstar) {
    vector<Bound*> bounds = datum -> bounds();
    int b = bounds.size();
    int i = 0, j = 0;
    while (j < b) {
        // Sample from B to decide which letter to go to
        int next_i = B[i][j].index(\
                SampleIndexFromLogDistribution(B[i][j].probs()));
        if (SEG_DEBUG) {
            cout << "Getting next_i: " << next_i - 1 << endl;
            vector<float> next_i_probs = B[i][j].probs();
            for (size_t k = 0; k < next_i_probs.size(); ++k) {
                if (next_i_probs[k] > MIN_PROB_VALUE) {
                    cout << "next_i_probs[" << k << "]: " << next_i_probs[k] << " ";
                }
            }
            cout << endl;
        }
        if (DEBUG) {
            cout << "Next_i: " << next_i - 1 << endl;
        }
        // Sample from Bstar to decide which bound to go to
        if (DEBUG) {
            cout << "Getting next_j" << endl;
        }
        int next_j = Bstar[next_i][j].index(\
                SampleIndexFromLogDistribution(Bstar[next_i][j].probs())); 
        if (DEBUG) {
            cout << "Next_j: " << next_j << endl;
        }
        if (DEBUG) {
            cout << "Creating segments" << endl;
        }
        // Create proper segments
        Segment* segment = new Segment(_config, next_i - 1);
        (datum -> segments()).push_back(segment);
        for (int  p = j; p < next_j; ++p) {
            segment -> push_back(bounds[p]);
        }
        i = next_i;
        j = next_j;
    }
}

void Sampler::MessageBackward(Datum* datum, \
        float* pi, float** A, \
        float*** segment_prob_given_cluster, \
            ProbList<int>** B, ProbList<int>** Bstar) {
    int b = (datum -> bounds()).size();
    int c = _config -> weak_limit();
    vector<Bound*> bounds = datum -> bounds();
    // Initialization
    for (int i = 0; i <= c; ++i) {
        B[i][b].push_back(0, -1);
    }
    int max_duration = _config -> max_duration();
    int total_frame_num = 0;
    vector<int> accumulated_frame_num(b, 0);
    for (int i = 0; i < b; ++i) {
        total_frame_num += bounds[i] -> frame_num();
        accumulated_frame_num[i] = total_frame_num;
    }
    // Compute B and Bstar
    for (int i = b - 1; i >= 0; --i) {
        for (int j = c; j > 0; --j) {
            // Compute Bstar
            int start_frame = i == 0 ? 0 : accumulated_frame_num[i - 1];
            for (int k = i + 1; k <= b && ((accumulated_frame_num[k - 1] - start_frame) <= max_duration || k == i + 1) ; ++k) { 
                Bstar[j][i].push_back(B[j][k].value() + segment_prob_given_cluster[j - 1][i][k - 1], k);
            }
        }
        for (int j = c; j >= 0; --j) {
            // Compute B
            if ((j > 0 && i > 0) || (j == 0 && i == 0)){
                if (j == 0 && i == 0) {
                    for (int k = 1; k <= c; ++k) {
                        B[j][i].push_back(pi[k - 1] + Bstar[k][i].value(), k);
                    }
                }
                else {
                    for (int k = 1; k <= c; ++k) {
                        B[j][i].push_back(A[j - 1][k - 1] + Bstar[k][i].value(), k);
                    }
                }
            }
        }
    }
}

void Sampler::RemoveClusterAssignment(Datum* datum, Counter* counter) {
    // Need to remove counts for A and for pi
    vector<Cluster*> cluster_counter = counter -> clusters();
    vector<Segment*> segments = datum -> segments();
    // Update pi
    counter -> MinusPi(segments[0] -> id());
    // Update A
    for (size_t i = 0; i < segments.size() - 1; ++i) {
        counter -> MinusA(segments[i] -> id(), segments[i + 1] -> id());
    }
    vector<Segment*>::iterator s_iter = segments.begin();
    for (; s_iter != segments.end(); ++s_iter) {
        if (!(cluster_counter[(*s_iter) -> id()] -> is_fixed())) {
            cluster_counter[(*s_iter) -> id()] -> Minus(*s_iter); 
        }
    }
    datum -> ClearSegs();
}

void Sampler::AddClusterAssignment(Datum* datum, Counter* counter) {
    // Need to add counts for A and for pi
    vector<Cluster*> cluster_counter = counter -> clusters();
    vector<Segment*> segments = datum -> segments();
    // update pi
    counter -> PlusPi(segments[0] -> id());
    // update A
    for (size_t i = 0; i < segments.size() - 1; ++i) {
        counter -> PlusA(segments[i] -> id(), segments[i + 1] -> id());
    } 
    vector<Segment*>::iterator s_iter = segments.begin();
    for (; s_iter != segments.end(); ++s_iter) {
        if (!(cluster_counter[(*s_iter) -> id()] -> is_fixed())) {
            cluster_counter[(*s_iter) -> id()] -> Plus(*s_iter);
        }
    }
}

Sampler::~Sampler() {
    delete _uniform;
    vslDeleteStream(&stream);
}
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
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./test.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include "uniformer.h"

using namespace std;

int main() {
    Uniformer *_uniform = new Uniformer(500000);
    cout << _uniform -> GetSample() << endl;
    cout << _uniform -> GetSample() << endl;
    delete _uniform;
    return 0;
}
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./toolkit.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <mkl_vml.h>
#include <mkl_cblas.h>
#include <cmath>
#include "toolkit.h"

int ToolKit::NormalizeDist(vector<float>& dis) {
    float total = 0;
    for (int i = 0; i < (int) dis.size(); ++i) {
        total += dis[i];
    }
    for (int i = 0 ; i < (int) dis.size(); ++i) {
        dis[i] /= total;
    }
    return total;
}

void ToolKit::MaxRemovedLogDist(vector<float>& log_probs) {
    float max_num = FindLogMax(log_probs);
    for (int i = 0 ; i < (int) log_probs.size(); ++i) {
        log_probs[i] -= max_num;
//        log_probs[i] = exp(log_probs[i]);
    }
    vsExp(log_probs.size(), &log_probs[0], &log_probs[0]);
}

float ToolKit::FindLogMax(vector<float>& log_probs) {
    MKL_INT incx = 1;
    int i = cblas_isamin(log_probs.size(), &log_probs[0], incx); 
    return log_probs[i];
    /*
    vector<float>::iterator iter = log_probs.begin();
    double max = *(iter++); 
    for (; iter != log_probs.end(); ++iter) { 
       if (*iter > max) {
          max = *iter;
       }
    }
    return max;
    */
}

float ToolKit::SumLogs(vector<float>& sub_probs) {
    float max = FindLogMax(sub_probs);
    float sum = 0;
    for (int i = 0; i < (int) sub_probs.size(); ++i) {
        if (sub_probs[i] - max > -99) {
            sum += exp(sub_probs[i] - max); 
        }
    }
    return max + log(sum);
}

float ToolKit::SumLogs(float a, float b) {
    // make a always the bigger number
    if (a < b) {
        float t = a;
        a = b;
        b = t;
    }
    return a + log(1 + exp(b - a));
}
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./uniformer.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <mkl_vsl.h>
#include <ctime>
#include <climits>
#include <iostream>
#include "uniformer.h"

using namespace std;

#define BRNG VSL_BRNG_MT19937
#define UNIFORM_METHOD VSL_RNG_METHOD_UNIFORM_STD

Uniformer::Uniformer() {
    _samples = NULL;
    _int_samples = NULL;
    unsigned int SEED = time(0);
    vslNewStream(&_stream, BRNG, SEED);
}

Uniformer::Uniformer(int batch) {
    _batch_size = batch;
    _index = batch;
    _int_index = batch;
    _samples = NULL;
    _int_samples = NULL;
    unsigned int SEED = time(0);
    vslNewStream(&_stream, BRNG, SEED);
}

int Uniformer::GetDiscreteSample() {
    if (_index >= _batch_size) {
        if (_int_samples == NULL) {
            _int_samples = new int [_batch_size];
        }
        viRngUniform(UNIFORM_METHOD, _stream, _batch_size, _int_samples, 0, INT_MAX); 
        _int_index = 0;
    }
    return _int_samples[_int_index++];
}

float Uniformer::GetSample() {
    if (_index >= _batch_size) {
        if (_samples == NULL) {
            _samples = new float [_batch_size];
        }
        vsRngUniform(UNIFORM_METHOD, _stream, _batch_size, _samples, 0, 1);
        _index = 0;
    }
    return _samples[_index++]; 
}

Uniformer::~Uniformer() {
    if (_samples != NULL) {
        delete[] _samples;
    }
    if (_int_samples != NULL) {
        delete[] _int_samples;
    }
    vslDeleteStream(&_stream);
}
/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./bound.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include "bound.h"

Bound::Bound(Config* config) {
    _config = config;
    _dim = _config -> dim();
    _is_labeled = false;
    _is_boundary = false;
    _label = -2;
}

void Bound::set_data(float** data, int frame_num) {
    _data = data;
    _frame_num = frame_num;
    for( int i = 0; i < _frame_num; ++i) {
        _data_array.push_back(_data[i]);
    }
}

void Bound::print_data() {
    for (int i = 0; i < (int) _data_array.size(); ++i) {
        for (int j = 0; j < _dim; ++j) {
            cout << _data_array[i][j] << " ";
        }
        cout << endl;
    }
}

Bound::~Bound() {
    for (int i = 0; i < _frame_num; ++i) {
        delete[] _data[i];
    }
    delete[] _data;
}
