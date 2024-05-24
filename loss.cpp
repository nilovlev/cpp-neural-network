#include "loss.h"

namespace neural_network {

double LossFunction::evaluate(const Vector& yPred, const Vector& yTrue) {
  return func(yPred, yTrue);
}

Vector LossFunction::evaluateGrad(const Vector& yPred, const Vector& yTrue) {
  return funcGrad(yPred, yTrue);
}

double LossFunction::func(const Vector& yPred, const Vector& yTrue) {
  return (yPred - yTrue).dot(yPred - yTrue) / yTrue.rows();
}

Vector LossFunction::funcGrad(const Vector& yPred, const Vector& yTrue) {
  return (yPred - yTrue) * 2;
}

}  // namespace neural_network 
