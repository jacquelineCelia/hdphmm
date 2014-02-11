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
