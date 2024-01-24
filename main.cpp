#include "layer.h"
#include "loss.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

const int pixels_count = 28;
const int input_values_length = pixels_count * pixels_count;

std::vector<std::pair<VectorXd, int>> get_layer0_values(std::string filename) {
  std::vector<std::pair<VectorXd, int>> res = std::vector<std::pair<VectorXd, int>>();
  std::ifstream file(filename);
  std::string line, val;

  std::getline(file, line);

  while (std::getline(file, line)) {
    VectorXd vector = VectorXd(input_values_length);
    std::stringstream ss(line);

    int index = -1;
    int ans;
    while (std::getline(ss, val, ',')) {
      if (index == -1) {
        ans = std::stoi(val);
        ++index;
        continue;
      }

      vector(index) = std::stod(val) / 255;
      ++index;
    }

    res.push_back({vector, ans});
  }

  file.close();
  return res;
}

int main() {
  const int epochs = 60000;
  const int layer1_values_length = 16;
  const int final_values_length = 10;

  Layer layer1 = Layer(input_values_length, layer1_values_length);
  Layer layer2 = Layer(layer1_values_length, final_values_length);

  auto data = get_layer0_values("mnist/mnist_train.csv");
  for (int i = 0; i < data.size(); ++i) {
    if (i >= epochs) {
      break;
    }

    VectorXd layer0_values = data[i].first;
    int ans = data[i].second;
    VectorXd ans_vector = VectorXd::Zero(10);
    ans_vector[ans] = 1;

    VectorXd layer1_values = layer1.evaluate(layer0_values);

    VectorXd layer2_values = layer2.evaluate(layer1_values);

    double loss_res = LossFunction::evaluate(layer2_values, ans_vector);

    VectorXd u = LossFunction::evaluateGrad(layer2_values, ans_vector);

    layer2.shift(layer1_values, u);
    MatrixXd u_2 = layer2.evaluateU(layer1_values, u);

    layer1.shift(layer0_values, u_2);
  }

  int right_answers_count = 0;
  data = get_layer0_values("mnist/mnist_test.csv");
  for (int i = 0; i < data.size(); ++i) {

    VectorXd layer0_values = data[i].first;
    int ans = data[i].second;
    VectorXd ans_vector = VectorXd::Zero(10);
    ans_vector[ans] = 1;

    VectorXd layer1_values = layer1.evaluate(layer0_values);

    VectorXd layer2_values = layer2.evaluate(layer1_values);

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
  std::cout << "all: " << data.size() << std::endl;

  std::cout << "percent: " << right_answers_count * 1.0 / data.size() * 100 << std::endl;
}
