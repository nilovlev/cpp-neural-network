#pragma once

#include <vector>

#include "layer.h"
#include "load_data.h"
#include "loss.h"

namespace neural_network {

class NeuralNetwork {
 private:
  std::vector<Layer> layers_;
  void backPropagation(const std::vector<Vector>& layerValues, Index answer, double learningRate);
  Vector convertToVector(Index answer);

 public:
  NeuralNetwork(std::initializer_list<Index> layerLenghts,
                std::initializer_list<ActivationFunctionType> layerFuncTypes);
  void train(const Data& trainData, Index epochs, double learningRate);
  std::vector<Vector> getLayerValues(const Matrix::ConstRowXpr& firstLayerValues) const;
};

}  // namespace neural_network
