#include "network.h"
#include "load_data.h"

#include <vector>

int main() {
  const int pixelsCount = 28;
  const int startLayerLength = pixelsCount * pixelsCount;
  std::vector<int> layers = {startLayerLength, 16, 10};
  neural_network::NeuralNetwork network = neural_network::NeuralNetwork(layers, 60000,
    neural_network::LoadData::getStartLayerValues("mnist/mnist_train.csv", startLayerLength),
    neural_network::LoadData::getStartLayerValues("mnist/mnist_test.csv", startLayerLength));

  network.train();
  network.test();
}
