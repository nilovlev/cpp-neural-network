#include <cmath>
#include <random>
#include <Eigen/Dense>
#include "sigma.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

const double h = 2;

class Layer {
 public:
  MatrixXd a;
  VectorXd b;

  Layer(int n, int m) {
    a = MatrixXd(m, n);
    b = VectorXd(m);
    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < a.rows(); ++i) {
      for (int j = 0; j < a.cols(); ++j) {
        a(i, j) = distribution(generator);
      }
      b(i) = distribution(generator);
    }
  }
  
  VectorXd evaluate(VectorXd x) {
    return Sigma::evaluate(a * x + b);
  }

  MatrixXd gradA(VectorXd x, VectorXd u) {
    return Sigma::evaluateDerivative(a * x + b).asDiagonal() * u * x.transpose();
  }

  VectorXd gradB(VectorXd x, VectorXd u) {
    return Sigma::evaluateDerivative(a * x + b).asDiagonal() * u;
  }

  VectorXd evaluateU(VectorXd x, VectorXd u) {
    return u.transpose() * Sigma::evaluateDerivative(a * x + b).asDiagonal() * a;
  }

  void shift(VectorXd x, VectorXd u) {
    a -= h * gradA(x, u);
    b -= h * gradB(x, u);
  }

};
