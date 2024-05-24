#pragma once

#include "common.h"

namespace neural_network {

class LossFunction {
 public:
  static double evaluate(const Vector& yPred, const Vector& yTrue);
  static Vector evaluateGrad(const Vector& yPred, const Vector& yTrue);

 private:
  static double func(const Vector& yPred, const Vector& yTrue);
  static Vector funcGrad(const Vector& yPred, const Vector& yTrue);
};

}  // namespace neural_network
