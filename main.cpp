#include <iostream>
#include <vector>

#include "load_data.h"
#include "network.h"

namespace {

using neural_network::ActivationFunctionType;
using neural_network::Data;
using neural_network::Index;
using neural_network::LoadData;
using neural_network::NeuralNetwork;
using neural_network::Vector;

void test(const Data& testData, const NeuralNetwork& network) {
  int rightAnswersCount = 0;
  for (int i = 0; i < testData.answer.size(); ++i) {
    std::vector<Vector> layersValues = network.getLayerValues(testData.input.row(i));

    Index maxIndex;
    layersValues[layersValues.size() - 1].maxCoeff(&maxIndex);

    if (maxIndex == testData.answer(i)) {
      ++rightAnswersCount;
    }
  }

  std::cout << "correct: " << rightAnswersCount << std::endl;
  std::cout << "all: " << testData.answer.size() << std::endl;
  std::cout << "accuracy: " << rightAnswersCount * 1.0 / testData.answer.size() * 100 << "%"
            << std::endl;
}

}

int main() {
  constexpr Index pixelsCount = 28;
  constexpr Index startLayerLength = pixelsCount * pixelsCount;
  NeuralNetwork network = NeuralNetwork(
      {startLayerLength, 16, 10}, {ActivationFunctionType::Sigmoid, ActivationFunctionType::Sigmoid,
                                   ActivationFunctionType::Sigmoid});
  constexpr double defaultLearningRate = 0.2;
  constexpr int epochs = 1;
  Data trainData = LoadData::read("../mnist/mnist_train.csv");
  Data testData = LoadData::read("../mnist/mnist_test.csv");
  network.train(trainData, epochs, defaultLearningRate);
  test(testData, network);
}
