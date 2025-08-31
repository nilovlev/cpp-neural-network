#include "activation_function.h"

namespace neural_network {

std::function<double(double)> ActivationFunctionLib::getFunc(ActivationFunctionType funcType) {
  switch (funcType) {
    case ActivationFunctionType::Sigmoid:
      return [](double x) { return 1.0 / (1.0 + exp(-x)); };
    case ActivationFunctionType::ReLU:
      return [](double x) { return std::max(1.0, x); };
    case ActivationFunctionType::Linear:
      return [](double x) { return x; };
    case ActivationFunctionType::TanH:
      return [](double x) { return tanh(x); };
  }
  return [](double x) { return 1.0 / (1.0 + exp(-x)); };
}

std::function<double(double)> ActivationFunctionLib::getFuncDerivative(
    ActivationFunctionType funcType) {
  switch (funcType) {
    case ActivationFunctionType::Sigmoid:
      return [](double x) { return (1.0 / (1.0 + exp(-x))) * (1.0 - (1.0 / (1.0 + exp(-x)))); };
    case ActivationFunctionType::ReLU:
      return [](double x) { return x > 0 ? 1.0 : 0.0; };
    case ActivationFunctionType::Linear:
      return [](double x) { return 1.0; };
    case ActivationFunctionType::TanH:
      return [](double x) { return 1.0 - pow(tanh(x), 2); };
  }
  return [](double x) { return (1.0 / (1.0 + exp(-x))) * (1.0 - (1.0 / (1.0 + exp(-x)))); };
}

ActivationFunction::ActivationFunction(ActivationFunctionType funcType)
    : func_(ActivationFunctionLib::getFunc(funcType)),
      funcDerivative_(ActivationFunctionLib::getFuncDerivative(funcType)) {
}

Vector ActivationFunction::evaluate(const Vector& x) const {
  return x.unaryExpr(func_);
}

Vector ActivationFunction::evaluateDerivative(const Vector& x) const {
  return x.unaryExpr(funcDerivative_);
}

}  // namespace neural_network
