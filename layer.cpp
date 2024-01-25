#include "layer.h"

namespace neural_network {

Layer::Layer(int n, int m) {
  a = MatrixXd(m, n);
  b = VectorXd(m);

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.cols(); ++j) {
      a(i, j) = distribution(generator);
    }
    b(i) = distribution(generator);
  }
}

VectorXd Layer::evaluate(VectorXd x) {
  return Sigma::evaluate(a * x + b);
}

MatrixXd Layer::gradA(VectorXd x, VectorXd u) {
  return Sigma::evaluateDerivative(a * x + b).asDiagonal() * u * x.transpose();
}

VectorXd Layer::gradB(VectorXd x, VectorXd u) {
  return Sigma::evaluateDerivative(a * x + b).asDiagonal() * u;
}

VectorXd Layer::evaluateU(VectorXd x, VectorXd u) {
  return u.transpose() * Sigma::evaluateDerivative(a * x + b).asDiagonal() * a;
}

void Layer::shift(VectorXd x, VectorXd u) {
  a -= learningRate * gradA(x, u);
  b -= learningRate * gradB(x, u);
}

}
