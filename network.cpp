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

std::vector<Vector> NeuralNetwork::getLayerValues(const Matrix::ConstRowXpr& firstLayerValues) const {
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

}  // namespace neural_network
