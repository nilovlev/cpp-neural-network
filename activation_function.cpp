#include "activation_function.h"

#include <iostream>

namespace neural_network {

ActivationFunction::ActivationFunction(ActivationFunctionType func) {
  switch (func) {
    case ActivationFunctionType::Sigmoid:
      func_ = [](double x) { return 1.0 / (1.0 + exp(-x)); };
      funcDerivative_ = [](double x) { return (1.0 / (1.0 + exp(-x))) * (1.0 - (1.0 / (1.0 + exp(-x)))); };
      break;
    case ActivationFunctionType::ReLU:
      func_ = [](double x) { return std::max(1.0, x); };
      funcDerivative_ = [](double x) { return x > 0 ? 1.0 : 0.0; };
      break;
    case ActivationFunctionType::Linear:
      func_ = [](double x) { return x; };
      funcDerivative_ = [](double x) { return 1.0; };
      break;
    case ActivationFunctionType::TanH:
      func_ = [](double x) { return tanh(x); };
      funcDerivative_ = [](double x) { return 1.0 - pow(tanh(x), 2); };
      break;
  }
}

Vector ActivationFunction::evaluate(const Vector& x) const {
  return x.unaryExpr(func_);
}

Vector ActivationFunction::evaluateDerivative(const Vector& x) const {
  return x.unaryExpr(funcDerivative_);
}

}  // namespace neural_network
