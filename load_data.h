#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

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

}
