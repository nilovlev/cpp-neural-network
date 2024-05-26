#pragma once

#include <EigenRand/EigenRand>

#include "common.h"
#include "sigma.h"

namespace neural_network {

constexpr double learningRate = 0.2;

class Layer {
 private:
  Matrix a_;
  Vector b_;
  static constexpr double normalDistributionParam1 = 0.0;
  static constexpr double normalDistributiomParam2 = 1.0;
  static Eigen::Rand::P8_mt19937_64& randGen();
  static Matrix getNormalRandMatrix(Index rows, Index cols);
  static Matrix getDefaultA(Index rows, Index cols);
  static Vector getDefaultB(Index cols);
  Matrix gradA(const Vector& x, const Vector& u) const;
  Vector gradB(const Vector& x, const Vector& u) const;

 public:
  Layer(int n, int m);
  Vector evaluate(const Vector& x) const;
  Vector evaluateU(const Vector& x, const Vector& u) const;
  void shift(const Vector& x, const Vector& u);
};

}  // namespace neural_network
