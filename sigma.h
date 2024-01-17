#include <Eigen/Dense>

using Eigen::VectorXd;

class Sigma {
 public:
  
  static VectorXd evaluate(VectorXd x) {
    VectorXd res = VectorXd(x.rows());
    for (int i = 0; i < x.rows(); ++i) {
      res(i) = sigmoid(x(i));
    }
    return res;
  }

  static VectorXd evaluateDerivative(VectorXd x) {
    VectorXd res = VectorXd(x.rows());
    for (int i = 0; i < x.rows(); ++i) {
      res(i) = sigmoidDerivative(x(i));
    }
    return res;
  }

  static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }

  static double sigmoidDerivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
  }
};
