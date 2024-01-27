#pragma once

#include <Eigen/Dense>

namespace neural_network {

using Eigen::VectorXd;

class LossFunction {
 public:
  static double evaluate(VectorXd yPred, VectorXd yTrue);
  static VectorXd evaluateGrad(VectorXd yPred, VectorXd yTrue);

 private:
  static double MSE(VectorXd yPred, VectorXd yTrue);
  static VectorXd MSEGrad(VectorXd yPred, VectorXd yTrue);
};

}  // namespace neural_network
