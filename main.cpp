#include <iostream>
#include "Tensor.h"

using namespace std;

int main() {
    // 1. Tensor de entrada de 1000x20x20
    Tensor entrada = Tensor::random({1000, 20, 20}, 0.0, 1.0);
    cout << "Paso 1 Size entrada: " << entrada.size() << endl;

    // 2. Redimensionar a 1000x400
    Tensor x = entrada.view({1000, 400});
  
    cout << "Paso 2 Size view: " << x.size() << endl;

    // 3. Multiplicar por pesos W1 (400x100)
    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor z1 = matmul(x, W1);
    cout << "Paso 3 Size matmul W1: " << z1.size() << endl;

    // 4. Sumar bias b1 (1x100)
    Tensor b1 = Tensor::zeros({1, 100});
    Tensor z1_bias = z1 + b1;
  
    cout << "Paso 4 Size suma bias b1: " << z1_bias.size() << endl;
  

    // 5. Funcion ReLU
    Tensor a1 = z1_bias.relu();
  
    cout << "Paso 5 Size ReLU: " << a1.size() << endl;

    // 6. Multiplicar por pesos W2 (100x10)
    Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor z2 = matmul(a1, W2);

    cout << "Paso 6 - Size matmul W2: " << z2.size() << endl;

  
    // 7. Sumar bias b2 (1x10)
  
    Tensor b2 = Tensor::zeros({1, 10});
    Tensor z2_bias = z2 + b2;
    cout << "Paso 7 - Size suma bias b2: " << z2_bias.size() << endl;

  
    // 8. Funcion Sigmoid
  
    Tensor res = z2_bias.sigmoid();
    cout << "Paso 8 - Size salida Sigmoid: " << res.size() << endl;

    return 0;
}
