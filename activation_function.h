#pragma once

#include <functional>

#include "common.h"

namespace neural_network {

enum class ActivationFunctionType { Sigmoid, ReLU, Linear, TanH };

class ActivationFunction {
 private:
  std::function<double(double)> func_;
  std::function<double(double)> funcDerivative_;

 public:
  ActivationFunction(ActivationFunctionType func);
  Vector evaluate(const Vector& x) const;
  Vector evaluateDerivative(const Vector& x) const;
};

}  // namespace neural_network
