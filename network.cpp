#include "network.h"

namespace neural_network {

NeuralNetwork::NeuralNetwork(std::vector<int> layerLengths, int epochs, std::vector<Data> trainData,
                             std::vector<Data> testData) {
  layers_.resize(layerLengths.size() - 1);
  for (int i = 0; i < layerLengths.size() - 1; ++i) {
    layers_[i] = Layer(layerLengths[i], layerLengths[i + 1]);
  }

  epochs_ = epochs;
  trainData_ = trainData;
  testData_ = testData;
}

void NeuralNetwork::train() {
  for (int i = 0; i < trainData_.size(); ++i) {
    if (i >= epochs_) {
      break;
    }

    VectorXd startLayerValues = trainData_[i].pixels;
    int ans = trainData_[i].answer;
    VectorXd ans_vector = VectorXd::Zero(10);
    ans_vector[ans] = 1;

    std::vector<VectorXd> layerValues = std::vector<VectorXd>(layers_.size() + 1);
    layerValues[0] = startLayerValues;

    for (int i = 0; i < layers_.size(); ++i) {
      layerValues[i + 1] = layers_[i].evaluate(layerValues[i]);
    }

    VectorXd lastLayerU = LossFunction::evaluateGrad(layerValues[layers_.size()], ans_vector);

    MatrixXd currentU = lastLayerU;
    for (int i = layers_.size() - 1; i >= 0; --i) {
      layers_[i].shift(layerValues[i], currentU);
      currentU = layers_[i].evaluateU(layerValues[i], currentU);
    }
  }
}

void NeuralNetwork::test() {
  int rightAnswersCount = 0;
  for (int i = 0; i < testData_.size(); ++i) {
    VectorXd startLayerValues = testData_[i].pixels;
    int ans = testData_[i].answer;
    VectorXd ans_vector = VectorXd::Zero(10);
    ans_vector[ans] = 1;

    VectorXd currentLayerValues = startLayerValues;
    for (int i = 0; i < layers_.size(); ++i) {
      currentLayerValues = layers_[i].evaluate(currentLayerValues);
    }

    double maxVal = 0;
    int maxIndex = -1;
    for (int i = 0; i < currentLayerValues.rows(); ++i) {
      if (currentLayerValues(i) > maxVal) {
        maxVal = currentLayerValues(i);
        maxIndex = i;
      }
    }

    if (maxIndex == ans) {
      ++rightAnswersCount;
    }
  }

  std::cout << "correct: " << rightAnswersCount << std::endl;
  std::cout << "all: " << testData_.size() << std::endl;

  std::cout << "percent: " << rightAnswersCount * 1.0 / testData_.size() * 100 << std::endl;
}

}  // namespace neural_network
