#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <cstddef>

using namespace std;

class Tensor {
private:
  vector<size_t> shape;
  size_t total_size;
  double* data;

public:
  Tensor(const vector<size_t>& shape_ , const vector<double>& values_);
  ~Tensor();

  static Tensor zeros(const vector<size_t>& shape);
  static Tensor ones(const vector<size_t>& shape);
  static Tensor random(const vector<size_t>& shape, double min, double max);
  static Tensor arange(double start, double end);
};

#endif // TENSOR_H
