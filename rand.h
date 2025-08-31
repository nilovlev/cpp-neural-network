#pragma once

#include <EigenRand/EigenRand>

#include "common.h"

namespace neural_network {

class Rand {
 private:
  static constexpr double normalDistributionParam1 = 0.0;
  static constexpr double normalDistributiomParam2 = 1.0;
  static Eigen::Rand::P8_mt19937_64& generator();

 public:
  static Matrix getNormalMatrix(Index rows, Index cols);
};

}  // namespace neural_network
