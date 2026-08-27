#include "Tensor.h"
#include <cstdlib>

using namespace std;

// SECCION 3: Constructor y creacion de tensores pre-definidos

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


// SECCION 4: Gestion de memoria y ciclo de vida

Tensor::Tensor(const Tensor& other){
  shape = other.shape;
  total_size = other.total_size;
  data = new double[total_size];

  for(size_t i = 0; i<total_size; i++){
    data[i] = other.data[i];
  }
}

Tensor::Tensor(Tensor&& other) noexcept {
  shape = other.shape;
  total_size = other.total_size;
  data = other.data;
  other.data = nullptr;
  other.total_size = 0;
  other.shape.clear();
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    delete[] data;
    shape = other.shape;
    total_size = other.total_size;
    data = new double[total_size];

    for(size_t i = 0; i<total_size; i++){
      data[i] = other.data[i];
    }
  }
  return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other){
    delete[] data;
    shape = other.shape;
    total_size = other.total_size;
    data = other.data;
    other.data = nullptr;
    other.total_size = 0;
    other.shape.clear();
  }

  return *this;
}


// SECCION 5: Sobrecarga de operadores

Tensor Tensor::operator+(const Tensor& other) const { 
  if (this->shape != other.shape) { 
    throw invalid_argument("Dimensiones incompatibles para la suma."); 
  } 
  
  vector<double> nuevos_valores(total_size); 
  
  for (size_t i = 0; i < total_size; i++) { 
    nuevos_valores[i] = this->data[i] + other.data[i]; 
  } 
  return Tensor(this->shape, nuevos_valores); 
}

Tensor Tensor::operator-(const Tensor& other) const { 
  if (this->shape != other.shape) { 
    throw invalid_argument("Dimensiones incompatibles para la resta."); 
  } 
  
  vector<double> nuevos_valores(total_size); 
  
  for (size_t i = 0; i < total_size; i++) { 
    nuevos_valores[i] = this->data[i] - other.data[i]; 
  } 
  
  return Tensor(this->shape, nuevos_valores); 
} 

Tensor Tensor::operator*(const Tensor& other) const { 
  if (this->shape != other.shape) { 
    throw invalid_argument("Dimensiones incompatibles para la multiplicacion."); 
  } 
  
  vector<double> nuevos_valores(total_size); 
  
  for (size_t i = 0; i < total_size; i++) { 
    nuevos_valores[i] = this->data[i] * other.data[i]; 
  } 
  return Tensor(this->shape, nuevos_valores); 
} 

Tensor Tensor::operator*(double scalar) const { 
  vector<double> nuevos_valores(total_size); 
  
  for (size_t i = 0; i < total_size; i++) { 
    nuevos_valores[i] = this->data[i] * scalar; 
  } 
  return Tensor(this->shape, nuevos_valores); 
} 

// SECCIÓN 6: MODIFICACION DE DIMENSIONES

Tensor Tensor::view(const vector<size_t>& new_shape) const { 
  if (new_shape.size() > 3) { 
    throw invalid_argument("El numero de dimensiones no puede exceder tres."); 
  } 
  
  size_t new_size = 1; 
  
  for (size_t i = 0; i < new_shape.size(); i++) { 
    new_size *= new_shape[i]; 
  } 
  
  if (new_size != this->total_size) { 
    throw invalid_argument("El numero total de elementos debe mantenerse constante."); 
  } 
  
  vector<double> valores; 
  
  for (size_t i = 0; i < total_size; i++) { 
    valores.push_back(data[i]); 
  } 
  return Tensor(new_shape, valores); 
} 

Tensor Tensor::unsqueeze(size_t dim) const { 
  if (this->shape.size() >= 3) { 
    throw invalid_argument("El tensor ya tiene 3 dimensiones."); 
  } 
  
  if (dim > this->shape.size()) { 
    throw invalid_argument("Posicion de dimension invalida para unsqueeze."); 
  } 
  
  vector<size_t> new_shape = this->shape; 
  new_shape.insert(new_shape.begin() + dim, 1); 
  vector<double> valores; 
  
  for (size_t i = 0; i < total_size; i++) { 
    valores.push_back(data[i]); 
  } 
  return Tensor(new_shape, valores); 
}








