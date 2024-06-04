#pragma once

#include <EigenRand/EigenRand>

#include "activation_function.h"
#include "common.h"
#include "rand.h"

namespace neural_network {

class Layer {
 private:
  Matrix a_;
  Vector b_;
  ActivationFunction activationFunction_;
  static Rand rand;
  static Matrix getDefaultA(Index rows, Index cols);
  static Vector getDefaultB(Index cols);
  Matrix gradA(const Vector& x, const Vector& u) const;
  Vector gradB(const Vector& x, const Vector& u) const;

 public:
  Layer(Index in, Index out, ActivationFunction activationFunction);
  Vector evaluate(const Vector& x) const;
  Vector evaluateU(const Vector& x, const Vector& u) const;
  void update(const Vector& x, const Vector& u, double learningRate);
};

}  // namespace neural_network
