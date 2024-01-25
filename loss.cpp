#include "loss.h"

namespace neural_network {

double LossFunction::evaluate(VectorXd y_pred, VectorXd y_true) {
  return MSE(y_pred, y_true);
}

VectorXd LossFunction::evaluateGrad(VectorXd y_pred, VectorXd y_true) {
  return MSEGrad(y_pred, y_true);
}

double LossFunction::MSE(VectorXd y_pred, VectorXd y_true) {
  VectorXd diff = (y_pred - y_true).unaryExpr([](double x) { return x * x; });
  return diff.mean();
}

VectorXd LossFunction::MSEGrad(VectorXd y_pred, VectorXd y_true) {
  VectorXd grad = (y_pred - y_true) * 2 / y_pred.rows();
  return grad;
}

}
