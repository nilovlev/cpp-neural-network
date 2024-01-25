#pragma once

#include "layer.h"
#include "loss.h"
#include "load_data.h"

#include <iostream>
#include <vector>

namespace neural_network {

class NeuralNetwork {
 private:
  int epochs_;
  std::vector<Layer> layers_;
  std::vector<Data> trainData_;
  std::vector<Data> testData_;

 public:
  NeuralNetwork(std::vector<int> layerLenghts, int epochs, std::vector<Data> trainData, std::vector<Data> testData);

  void train();
  void test();
};

}
