#include <Eigen/Dense>
#include <iostream>

using Eigen::VectorXd;

class LossFunction {
public:
  static double evaluate(VectorXd y_pred, VectorXd y_true) {
    return MSE(y_pred, y_true);
  }

  static VectorXd evaluateGrad(VectorXd y_pred, VectorXd y_true) {
    return MSEGrad(y_pred, y_true);
  }

  static double MSE(VectorXd y_pred, VectorXd y_true) {
    VectorXd diff = (y_pred - y_true).unaryExpr([](double x) { return x * x; });
    return diff.mean();
  }

  static VectorXd MSEGrad(VectorXd y_pred, VectorXd y_true) {
    VectorXd grad = (y_pred - y_true) * 2 / y_pred.rows();
    return grad;
  }
};
