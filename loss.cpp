#include "loss.h"

namespace neural_network {

double LossFunction::evaluate(VectorXd yPred, VectorXd yTrue) { return MSE(yPred, yTrue); }

VectorXd LossFunction::evaluateGrad(VectorXd yPred, VectorXd yTrue) {
  return MSEGrad(yPred, yTrue);
}

double LossFunction::MSE(VectorXd yPred, VectorXd yTrue) {
  VectorXd diff = (yPred - yTrue).unaryExpr([](double x) { return x * x; });
  return diff.mean();
}

VectorXd LossFunction::MSEGrad(VectorXd yPred, VectorXd yTrue) {
  VectorXd grad = (yPred - yTrue) * 2 / yPred.rows();
  return grad;
}

}  // namespace neural_network
