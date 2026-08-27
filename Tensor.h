#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <cmath>

using namespace std;

class Tensor {
private:
  vector<size_t> shape;
  size_t total_size;
  double* data;

public:
  // SECCION 3: Constructor principal y creacion de tensores predefinidos
  Tensor(const vector<size_t>& shape_ , const vector<double>& values_);
  ~Tensor();

  static Tensor zeros(const vector<size_t>& shape);
  static Tensor ones(const vector<size_t>& shape);
  static Tensor random(const vector<size_t>& shape, double min, double max);
  static Tensor arange(double start, double end);


  // SECCION 4: Gestion de memoria y ciclo de vida
  Tensor(const Tensor& other);
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(const Tensor& other);
  Tensor& operator=(Tensor&& other) noexcept;


  // SECCION 5 : Sobrecarga de operadores
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator*(double scalar) const;


  // SECCION 6: Modificacion de dimensiones
  Tensor view(const vector<size_t>& new_shape) const;
  Tensor unsqueeze(size_t dim) const;

  // SECCION 7: Concatenacion
  static Tensor concat(const vector<Tensor>& tensores, size_t dim);

  // SECCION 8: Funciones amigas permitidas
  friend Tensor dot(const Tensor& a, const Tensor& b);
  friend Tensor matmul(const Tensor& a, const Tensor& b);

  // SECCION 9: Red Neuronal
  Tensor relu() const;
  Tensor sigmoid() const;
  size_t size() const { return total_size; }
  void print_shape() const;
};

#endif // TENSOR_H
