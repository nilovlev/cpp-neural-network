#pragma once

#include "common.h"

namespace neural_network {

class LossFunction {
 private:
  static double MSE(const Vector& yPred, const Vector& yTrue);
  static Vector MSEGrad(const Vector& yPred, const Vector& yTrue);

 public:
  static double evaluate(const Vector& yPred, const Vector& yTrue);
  static Vector evaluateGrad(const Vector& yPred, const Vector& yTrue);
};

}  // namespace neural_network
