#pragma once

#include <Eigen/Dense>

namespace neural_network {

using Eigen::VectorXd;

class Sigma {
 private:
  static double sigmoid(double x);
  static double sigmoidDerivative(double x);

 public:
  static VectorXd evaluate(VectorXd x);
  static VectorXd evaluateDerivative(VectorXd x);
};

}  // namespace neural_network
