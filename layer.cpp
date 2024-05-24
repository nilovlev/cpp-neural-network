#include "layer.h"

namespace neural_network {

Layer::Layer(int in, int out) : a_(getDefaultA(out, in)), b_(getDefaultB(out)) {
}

Eigen::Rand::P8_mt19937_64& Layer::randGen() {
  static Eigen::Rand::P8_mt19937_64 r;
  return r;
}

Matrix Layer::getNormalRandMatrix(Index rows, Index cols) {
  Eigen::Rand::NormalGen<double> norm_gen{normalDistributionParam1, normalDistributiomParam2};
  return norm_gen.generate<Matrix>(rows, cols, Layer::randGen());
}

Matrix Layer::getDefaultA(Index rows, Index cols) {
  return Layer::getNormalRandMatrix(rows, cols);
}

Vector Layer::getDefaultB(Index cols) {
  return Layer::getNormalRandMatrix(cols, 1);
}

Matrix Layer::gradA(const Vector& x, const Vector& u) const {
  return Sigma::evaluateDerivative(a_ * x + b_).asDiagonal() * u * x.transpose();
}

Vector Layer::gradB(const Vector& x, const Vector& u) const {
  return Sigma::evaluateDerivative(a_ * x + b_).asDiagonal() * u;
}

Vector Layer::evaluate(const Vector& x) const {
  return Sigma::evaluate(a_ * x + b_);
}

Vector Layer::evaluateU(const Vector& x, const Vector& u) const {
  return u.transpose() * Sigma::evaluateDerivative(a_ * x + b_).asDiagonal() * a_;
}

void Layer::shift(const Vector& x, const Vector& u) {
  a_ -= learningRate * gradA(x, u);
  b_ -= learningRate * gradB(x, u);
}

}  // namespace neural_network
