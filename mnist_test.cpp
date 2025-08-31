#include "mnist_test.h"

namespace neural_network {

void MnistTest::runAllTests() {
  accuracyTest();
}

void MnistTest::accuracyTest() {
  constexpr Index pixelsCount = 28;
  constexpr Index startLayerLength = pixelsCount * pixelsCount;
  NeuralNetwork network = NeuralNetwork(
      {startLayerLength, 64, 16, 10}, {ActivationFunctionType::Sigmoid, ActivationFunctionType::Sigmoid,
                                   ActivationFunctionType::Sigmoid});
  constexpr double defaultLearningRate = 0.2;
  constexpr int epochs = 1;
  const Data trainData = LoadData::read("../mnist/mnist_train.csv");
  const Data testData = LoadData::read("../mnist/mnist_test.csv");
  network.train(trainData, epochs, defaultLearningRate);

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

}  // namespace neural_network
