#include "rand.h"

namespace neural_network {

Eigen::Rand::P8_mt19937_64& Rand::generator() {
  static Eigen::Rand::P8_mt19937_64 r;
  return r;
}

Matrix Rand::getNormalMatrix(Index rows, Index cols) {
  Eigen::Rand::NormalGen<double> norm_gen{normalDistributionParam1, normalDistributiomParam2};
  return norm_gen.generate<Matrix>(rows, cols, generator());
}

}  // namespace neural_network
