#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"

namespace neural_network {

struct Data {
  Vector pixels;
  int answer;
};

class LoadData {
 public:
  static std::vector<Data> getStartLayerValues(const std::string& fileName, int startLayerLength);
};

}  // namespace neural_network
