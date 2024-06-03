#include "loss.h"

namespace neural_network {

double LossFunction::MSE(const Vector& yPred, const Vector& yTrue) {
  return (yPred - yTrue).dot(yPred - yTrue) / yTrue.rows();
}

Vector LossFunction::MSEGrad(const Vector& yPred, const Vector& yTrue) {
  return (yPred - yTrue) * 2;
}

double LossFunction::evaluate(const Vector& yPred, const Vector& yTrue) {
  return MSE(yPred, yTrue);
}

Vector LossFunction::evaluateGrad(const Vector& yPred, const Vector& yTrue) {
  return MSEGrad(yPred, yTrue);
}

}  // namespace neural_network 
