#pragma once

#include <Eigen/Dense>

namespace neural_network {

using Eigen::VectorXd;

class LossFunction {
 public:
  static double evaluate(VectorXd y_pred, VectorXd y_true);
  static VectorXd evaluateGrad(VectorXd y_pred, VectorXd y_true);

 private:
  static double MSE(VectorXd y_pred, VectorXd y_true);
  static VectorXd MSEGrad(VectorXd y_pred, VectorXd y_true);
};

}
