#include "Tensor.h"
#include <cstdlib>

using namespace std;

Tensor::Tensor(const vector<size_t>& shape_ , const vector<double>& values_){
  shape = shape_;
  
  total_size = 1;
  for (size_t i = 0; i < shape.size(); i++){
    total_size *= shape[i];
  }
  
  data = new double[total_size];
  for (size_t i = 0; i < total_size; i++){
    data[i] = values_[i];
  }
}

Tensor::~Tensor(){
  delete[] data;
}

Tensor Tensor::zeros(const vector<size_t>& shape){
  size_t total = 1;
  for (size_t i=0; i<shape.size(); i++){
    total *= shape[i];
  }

  vector<double> ceros;
  for(size_t i = 0; i<total; i++){
    ceros.push_back(0.0);
  }
  return Tensor(shape,ceros);
}

Tensor Tensor::ones(const vector<size_t>& shape){
  size_t total = 1;
  for (size_t i=0; i<shape.size(); i++){
    total *= shape[i];
  }

  vector<double> unos;
  for(size_t i = 0; i<total; i++){
    unos.push_back(1.0);
  }
  return Tensor(shape,unos);
}

Tensor Tensor::random(const vector<size_t>& shape, double min, double max){
  size_t total = 1;
  for (size_t i=0; i<shape.size(); i++) {
    total *= shape[i];
  }

  vector <double> randoms;
  for(size_t i = 0; i<total; i++){
    double num_random = min + ((double)rand() / RAND_MAX) * (max-min);
    randoms.push_back(num_random);
  }
  return Tensor(shape, randoms);
}

Tensor Tensor::arange(double start, double end) {
  vector <double> valores;
  for(double i = start; i < end; i++){
    valores.push_back(i);
  }

  vector<size_t> shape_1d = {valores.size()};
  return Tensor(shape_1d, valores);
}






