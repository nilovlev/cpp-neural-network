#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "common.h"

namespace neural_network {

struct Data {
  Matrix input;
  Vector answer;
};

class LoadData {
 public:
  static Data read(const std::string& filePath);
};

}  // namespace neural_network
