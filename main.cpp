#include "network.h"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace neural_network {

using Eigen::MatrixXd;
using Eigen::VectorXd;

const int pixels_count = 28;
const int input_values_length = pixels_count * pixels_count;

std::vector<std::pair<VectorXd, int>> getStartLayerValues(std::string filename) {
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

}

int main() {
  neural_network::NeuralNetwork network = neural_network::NeuralNetwork(
    neural_network::getStartLayerValues("mnist/mnist_train.csv"), neural_network::getStartLayerValues("mnist/mnist_test.csv"));

  network.train();
  network.test();
}
