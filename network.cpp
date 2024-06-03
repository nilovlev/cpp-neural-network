#include "network.h"

namespace neural_network {

void NeuralNetwork::backPropagation(const std::vector<Vector>& layerValues, Index answer,
                                    double learningRate) {
  Vector ansVector = convertToVector(answer);
  Vector lastLayerU = LossFunction::evaluateGrad(layerValues[layers_.size()], ansVector);

  Matrix currentU = lastLayerU;
  for (int i = layers_.size() - 1; i >= 0; --i) {
    layers_[i].update(layerValues[i], currentU, learningRate);
    currentU = layers_[i].evaluateU(layerValues[i], currentU);
  }
}

Vector NeuralNetwork::convertToVector(Index answer) {
  Vector ansVector = Vector::Zero(10);
  ansVector[answer] = 1;
  return ansVector;
}

NeuralNetwork::NeuralNetwork(std::initializer_list<Index> layerLengths,
                             std::initializer_list<ActivationFunctionType> layerFuncTypes) {
  std::initializer_list<Index>::iterator lengthIterator = layerLengths.begin();
  std::initializer_list<ActivationFunctionType>::iterator funcTypeIterator = layerFuncTypes.begin();
  Index prevLayerLength = *layerLengths.begin();
  ++lengthIterator;
  for (; lengthIterator != layerLengths.end() && funcTypeIterator != layerFuncTypes.end();
       ++lengthIterator, ++funcTypeIterator) {
    layers_.emplace_back(Layer(prevLayerLength, *lengthIterator, ActivationFunction(*funcTypeIterator)));
    prevLayerLength = *lengthIterator;
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

std::vector<Vector> NeuralNetwork::getLayerValues(
    const Matrix::ConstRowXpr& firstLayerValues) const {
  std::vector<Vector> layerValues;
  layerValues.reserve((layers_.size() + 1));
  layerValues.emplace_back(firstLayerValues);
  for (auto& layer : layers_) {
    layerValues.emplace_back(layer.evaluate(layerValues.back()));
  }
  return layerValues;
}

}  // namespace neural_network
