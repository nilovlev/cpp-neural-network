#pragma once

#include "sigma.h"

#include <cmath>
#include <random>
#include <Eigen/Dense>

namespace neural_network {

using Eigen::VectorXd;
using Eigen::MatrixXd;

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

}
