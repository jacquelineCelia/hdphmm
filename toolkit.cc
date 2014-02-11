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
