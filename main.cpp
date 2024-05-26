#include <vector>

#include "load_data.h"
#include "network.h"

using neural_network::Index;
using neural_network::LoadData;
using neural_network::NeuralNetwork;

int main() {
  constexpr Index pixelsCount = 28;
  constexpr Index startLayerLength = pixelsCount * pixelsCount;
  std::vector<Index> layerLengths = {startLayerLength, 16, 10};
  NeuralNetwork network = NeuralNetwork(layerLengths);
  network.train(LoadData::read("../mnist/mnist_train.csv"), 1);
  network.test(LoadData::read("../mnist/mnist_test.csv"));

  // const neural_network::Data testData = LoadData::read("../mnist/mnist_test.csv");
  // std::vector<neural_network::Vector> layerValues = network.getLayerValues(testData.input.row(0));
  // std::cout << layerValues[2] << std::endl;
}
