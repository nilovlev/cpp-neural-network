#include "network.h"

namespace neural_network {

NeuralNetwork::NeuralNetwork(const std::vector<Index>& layerLengths, ActivationFunctionType funcType) {
  for (int i = 0; i < layerLengths.size() - 1; ++i) {
    layers_.emplace_back(Layer(layerLengths[i], layerLengths[i + 1], ActivationFunction(funcType)));
  }
}

void NeuralNetwork::train(const Data& trainData, Index epochs, double learningRate) {
  for (int epoch = 0; epoch < epochs; ++epoch) {
    for (int i = 0; i < trainData.answer.size(); ++i) {
      std::vector<Vector> layerValues = getLayerValues(trainData.input.row(i));
      backPropagation(layerValues, trainData.answer(i), learningRate);
    }
  }
}

std::vector<Vector> NeuralNetwork::getLayerValues(const Matrix::ConstRowXpr& firstLayerValues) {
  std::vector<Vector> layerValues(layers_.size() + 1);
  layerValues[0] = firstLayerValues;
  for (int i = 0; i < layers_.size(); ++i) {
    layerValues[i + 1] = layers_[i].evaluate(layerValues[i]);
  }
  return layerValues;
}

void NeuralNetwork::backPropagation(std::vector<Vector>& layerValues, Index answer, double learningRate) {
  Vector ansVector = Vector::Zero(10);
  ansVector[answer] = 1;
  Vector lastLayerU = LossFunction::evaluateGrad(layerValues[layers_.size()], ansVector);

  Matrix currentU = lastLayerU;
  for (int i = layers_.size() - 1; i >= 0; --i) {
    layers_[i].shift(layerValues[i], currentU, learningRate);
    currentU = layers_[i].evaluateU(layerValues[i], currentU);
  }
}

void NeuralNetwork::test(const Data& testData) {
  int rightAnswersCount = 0;
  for (int i = 0; i < testData.answer.size(); ++i) {
    int ans = testData.answer(i);
    Vector ansVector = Vector::Zero(10);
    ansVector[ans] = 1;

    Vector currentLayerValues = testData.input.row(i);
    for (int i = 0; i < layers_.size(); ++i) {
      currentLayerValues = layers_[i].evaluate(currentLayerValues);
    }

    Index maxIndex;
    currentLayerValues.maxCoeff(&maxIndex);

    if (maxIndex == ans) {
      ++rightAnswersCount;
    }
  }

  std::cout << "correct: " << rightAnswersCount << std::endl;
  std::cout << "all: " << testData.answer.size() << std::endl;

  std::cout << "percent: " << rightAnswersCount * 1.0 / testData.answer.size() * 100 << std::endl;
}

}  // namespace neural_network
