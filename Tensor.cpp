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
  if (this->shape == other.shape) {
    vector<double> nuevos_valores(total_size);
    
    for (size_t i = 0; i < total_size; i++) {
      nuevos_valores[i] = this->data[i] + other.data[i];
    }
    
    return Tensor(this->shape, nuevos_valores);
  }

  // Caso de (m x n) + (1 x n)
  if (this->shape.size() == 2 && other.shape.size() == 2 && other.shape[0] == 1 && other.shape[1] == this->shape[1]) {
    size_t m = this->shape[0];
    size_t n = this->shape[1];
    vector<double> nuevos_valores(total_size);

    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        nuevos_valores[i * n + j] = this->data[i * n + j] + other.data[j];
      }
    }
    
    return Tensor(this->shape, nuevos_valores);
  }
  throw invalid_argument("Dimensiones incompatibles para la suma.");
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


// SECCION 7: Concatenacion

Tensor Tensor::concat(const vector<Tensor>& tensores, size_t dim) {
  if (tensores.empty()) {
    throw invalid_argument("Debe proporcionar al menos un tensor para concatenar.");
  }

  size_t ndims = tensores[0].shape.size();
  if (dim >= ndims) {
    throw invalid_argument("Dimension de concatenacion invalida.");
  }

  for (const auto& t : tensores) {
    if (t.shape.size() != ndims) {
      throw invalid_argument("Todos los tensores deben tener el mismo numero de dimensiones.");
    }
    
    for (size_t i = 0; i < ndims; i++) {
      if (i != dim && t.shape[i] != tensores[0].shape[i]) {
        throw invalid_argument("Las dimensiones deben coincidir excepto en el eje de concatenacion.");
      }
    }
  }

  vector<size_t> new_shape = tensores[0].shape;
  size_t suma_dim = 0;
  
  for (const auto& t : tensores) {
    suma_dim += t.shape[dim];
  }
  
  new_shape[dim] = suma_dim;
  size_t new_total = 1;
  
  for (size_t s : new_shape) new_total *= s;

  vector<double> nuevos_datos(new_total);

  vector<size_t> out_strides(ndims);
  out_strides[ndims - 1] = 1;
  
  for (int i = (int)ndims - 2; i >= 0; i--) {
    out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
  }

  size_t offset_dim = 0;

  for (const auto& t : tensores) {
    vector<size_t> in_strides(ndims);
    in_strides[ndims - 1] = 1;
    
    for (int i = (int)ndims - 2; i >= 0; i--) {
      in_strides[i] = in_strides[i + 1] * t.shape[i + 1];
    }

    for (size_t flat = 0; flat < t.total_size; flat++) {
      size_t resto = flat;
      vector<size_t> idx(ndims);
      
      for (size_t i = 0; i < ndims; i++) {
        idx[i] = resto / in_strides[i];
        resto = resto % in_strides[i];
      }

      idx[dim] += offset_dim;

      size_t out_flat = 0;
      for (size_t i = 0; i < ndims; i++) {
        out_flat += idx[i] * out_strides[i];
      }

      nuevos_datos[out_flat] = t.data[flat];
    }

    offset_dim += t.shape[dim];
  }

  return Tensor(new_shape, nuevos_datos);
}


// SECCION 8: Funciones amiga permitidas

Tensor dot(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) {
    throw invalid_argument("Los tensores deben tener la misma forma para el producto punto.");
  }

  double suma = 0.0;
  for (size_t i = 0; i < a.total_size; i++) {
    suma += a.data[i] * b.data[i];
  }

  return Tensor({1}, {suma});
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw invalid_argument("matmul requiere tensores bidimensionales.");
  }
  if (a.shape[1] != b.shape[0]) {
    throw invalid_argument("Dimensiones incompatibles para matmul.");
  }

  size_t m = a.shape[0];
  size_t n = a.shape[1];
  size_t p = b.shape[1];

  vector<double> resultado(m * p, 0.0);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double suma = 0.0;
      for (size_t k = 0; k < n; k++) {
        suma += a.data[i * n + k] * b.data[k * p + j];
      }
      resultado[i * p + j] = suma;
    }
  }

  return Tensor({m, p}, resultado);
}


// SECCION 9: Red neuronal

Tensor Tensor::relu() const {
  vector<double> resultado(total_size);
  for (size_t i = 0; i < total_size; i++) {
    resultado[i] = (data[i] > 0.0) ? data[i] : 0.0;
  }
  return Tensor(shape, resultado);
}

Tensor Tensor::sigmoid() const {
  vector<double> resultado(total_size);
  for (size_t i = 0; i < total_size; i++) {
    resultado[i] = 1.0 / (1.0 + exp(-data[i]));
  }
  return Tensor(shape, resultado);
}

void Tensor::print_shape() const {
  cout << "(";
  for (size_t i = 0; i < shape.size(); i++) {
    cout << shape[i];
    if (i != shape.size() - 1) cout << ", ";
  }
  cout << ")";
}
