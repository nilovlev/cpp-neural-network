#pragma once

#include "layer.h"
#include "loss.h"

#include <iostream>
#include <vector>

namespace neural_network {

class NeuralNetwork {
 private:
  int epochs_ = 60000;
  Layer layer1_;
  Layer layer2_;
  std::vector<std::pair<VectorXd, int>> trainData_;
  std::vector<std::pair<VectorXd, int>> testData_;

 public:
  NeuralNetwork(std::vector<std::pair<VectorXd, int>> trainData, std::vector<std::pair<VectorXd, int>> testData);

  void train();
  void test();
};

}
