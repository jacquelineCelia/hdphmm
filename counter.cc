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
