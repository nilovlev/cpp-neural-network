#include "layer.h"

namespace neural_network {

Layer::Layer(Index in, Index out, ActivationFunction activationFunction)
    : a_(getDefaultA(out, in)),
      b_(getDefaultB(out)),
      activationFunction_(std::move(activationFunction)) {
}

Matrix Layer::getDefaultA(Index rows, Index cols) {
  return rand.getNormalMatrix(rows, cols);
}

Vector Layer::getDefaultB(Index cols) {
  return rand.getNormalMatrix(cols, 1);
}

Matrix Layer::gradA(const Vector& x, const Vector& u) const {
  return activationFunction_.evaluateDerivative(a_ * x + b_).asDiagonal() * u * x.transpose();
}

Vector Layer::gradB(const Vector& x, const Vector& u) const {
  return activationFunction_.evaluateDerivative(a_ * x + b_).asDiagonal() * u;
}

Vector Layer::evaluate(const Vector& x) const {
  return activationFunction_.evaluate(a_ * x + b_);
}

Vector Layer::evaluateU(const Vector& x, const Vector& u) const {
  return u.transpose() * activationFunction_.evaluateDerivative(a_ * x + b_).asDiagonal() * a_;
}

void Layer::update(const Vector& x, const Vector& u, double learningRate) {
  a_ -= learningRate * gradA(x, u);
  b_ -= learningRate * gradB(x, u);
}

}  // namespace neural_network
