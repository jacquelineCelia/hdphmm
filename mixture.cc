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
