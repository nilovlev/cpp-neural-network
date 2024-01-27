#include "sigma.h"

namespace neural_network {

VectorXd Sigma::evaluate(VectorXd x) {
  VectorXd res = VectorXd(x.rows());
  for (int i = 0; i < x.rows(); ++i) {
    res(i) = sigmoid(x(i));
  }
  return res;
}

VectorXd Sigma::evaluateDerivative(VectorXd x) {
  VectorXd res = VectorXd(x.rows());
  for (int i = 0; i < x.rows(); ++i) {
    res(i) = sigmoidDerivative(x(i));
  }
  return res;
}

double Sigma::sigmoid(double x) { return 1 / (1 + exp(-x)); }

double Sigma::sigmoidDerivative(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

}  // namespace neural_network
