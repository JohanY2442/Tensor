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


