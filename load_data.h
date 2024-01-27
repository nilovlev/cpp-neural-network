#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace neural_network {

using Eigen::VectorXd;

struct Data {
  VectorXd pixels;
  int answer;
};

class LoadData {
 public:
  static std::vector<Data> getStartLayerValues(std::string fileName, int startLayerLength);
};

}  // namespace neural_network
