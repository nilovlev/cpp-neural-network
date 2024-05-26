#pragma once

#include <vector>

#include "layer.h"
#include "load_data.h"
#include "loss.h"

namespace neural_network {

class NeuralNetwork {
 private:
  std::vector<Layer> layers_;
  void backPropagation(std::vector<Vector>& layerValues, Index answer);

 public:
  NeuralNetwork(const std::vector<Index>& layerLenghts);
  void train(const Data& trainData, Index epochs);
  std::vector<Vector> getLayerValues(const Matrix::ConstRowXpr& firstLayerValues);
  void test(const Data& testData);
};

}  // namespace neural_network
