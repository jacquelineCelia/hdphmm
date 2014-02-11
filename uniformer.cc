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
