#pragma once

#include "common.h"

namespace neural_network {

class Sigma {
 private:
  static double sigmoid(double x);
  static double sigmoidDerivative(double x);

 public:
  static Vector evaluate(const Vector& x);
  static Vector evaluateDerivative(const Vector& x);
};

}  // namespace neural_network
