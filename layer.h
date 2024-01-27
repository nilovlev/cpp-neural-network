#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <random>

#include "sigma.h"

namespace neural_network {

using Eigen::MatrixXd;
using Eigen::VectorXd;

const double learningRate = 2;

class Layer {
 private:
  MatrixXd a;
  MatrixXd b;
  MatrixXd gradA(VectorXd x, VectorXd u);
  VectorXd gradB(VectorXd x, VectorXd u);

 public:
  Layer() = default;
  Layer(int n, int m);
  VectorXd evaluate(VectorXd x);
  VectorXd evaluateU(VectorXd x, VectorXd u);
  void shift(VectorXd x, VectorXd u);
};

}  // namespace neural_network
