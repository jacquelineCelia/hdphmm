#ifndef UNIFORMER_H
#define UNIFORMER_H

#include <mkl.h>

class Uniformer {
 public:
  Uniformer();
  Uniformer(int batch);
  float GetSample();
  int GetDiscreteSample();
  ~Uniformer();
 private:
  float* _samples;
  int* _int_samples;
  int _batch_size;
  int _index;
  int _int_index;
  VSLStreamStatePtr _stream;
};

#endif
