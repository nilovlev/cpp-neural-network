#include "network.h"

namespace neural_network {


  NeuralNetwork::NeuralNetwork(std::vector<std::pair<VectorXd, int>> trainData, std::vector<std::pair<VectorXd, int>> testData) {
    int pixels_count = 28;
    int input_values_length = pixels_count * pixels_count;
    int layer1_values_length = 16;
    int final_values_length = 10;

    layer1_ = Layer(input_values_length, layer1_values_length);
    layer2_ = Layer(layer1_values_length, final_values_length);

    trainData_ = trainData;
    testData_ = testData;
  }

  void NeuralNetwork::train() {
    for (int i = 0; i < trainData_.size(); ++i) {
      if (i >= epochs_) {
        break;
      }

      VectorXd layer0_values = trainData_[i].first;
      int ans = trainData_[i].second;
      VectorXd ans_vector = VectorXd::Zero(10);
      ans_vector[ans] = 1;

      VectorXd layer1_values = layer1_.evaluate(layer0_values);

      VectorXd layer2_values = layer2_.evaluate(layer1_values);

      double loss_res = LossFunction::evaluate(layer2_values, ans_vector);

      VectorXd u = LossFunction::evaluateGrad(layer2_values, ans_vector);

      layer2_.shift(layer1_values, u);
      MatrixXd u_2 = layer2_.evaluateU(layer1_values, u);

      layer1_.shift(layer0_values, u_2);
    }
  }

  void NeuralNetwork::test() {
    int right_answers_count = 0;
    for (int i = 0; i < testData_.size(); ++i) {

      VectorXd layer0_values = testData_[i].first;
      int ans = testData_[i].second;
      VectorXd ans_vector = VectorXd::Zero(10);
      ans_vector[ans] = 1;

      VectorXd layer1_values = layer1_.evaluate(layer0_values);

      VectorXd layer2_values = layer2_.evaluate(layer1_values);

      double max_val = 0;
      int max_index = -1;
      for (int i = 0; i < layer2_values.rows(); ++i) {
        if (layer2_values(i) > max_val) {
          max_val = layer2_values(i);
          max_index = i;
        }
      }

      if (max_index == ans) {
        ++right_answers_count;
      }
    }

    std::cout << "correct: " << right_answers_count << std::endl;
    std::cout << "all: " << testData_.size() << std::endl;

    std::cout << "percent: " << right_answers_count * 1.0 / testData_.size() * 100 << std::endl;
  }

}
