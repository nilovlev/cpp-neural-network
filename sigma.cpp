#include "sigma.h"

namespace neural_network {

Vector Sigma::evaluate(const Vector& x) {
  return x.unaryExpr(&sigmoid);
}

Vector Sigma::evaluateDerivative(const Vector& x) {
  return x.unaryExpr(&sigmoidDerivative);
}

double Sigma::sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

double Sigma::sigmoidDerivative(double x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

}  // namespace neural_network
