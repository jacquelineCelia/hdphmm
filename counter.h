#ifndef COUNTER_H
#define COUNTER_H

#include "config.h"
#include "cluster.h"
#include <vector>
#include <map>

using namespace std;

class Counter {
 public:
  Counter(Config*);
  vector<Cluster*>& clusters() {return _cluster_counter;}
  Counter& operator+= (Counter&);
  float** A() {return _A;}
  float* pi() {return _pi;}
  int weak_limit() {return _weak_limit;}
  void MinusPi(int i, int n = 1) {_pi[i] -= n;}
  void MinusA(int i, int j, int n = 1) {_A[i][j] -= n;}
  void PlusPi(int i, int n = 1) {_pi[i] += n;}
  void PlusA(int i, int j, int n = 1) {_A[i][j] += n;}
  ~Counter();
 private:
  vector<Cluster*> _cluster_counter;
  Config* _config; 
  int _weak_limit;
  float** _A;
  float* _pi;
};

#endif
