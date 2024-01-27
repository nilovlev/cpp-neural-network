#include "load_data.h"

namespace neural_network {

std::vector<Data> LoadData::getStartLayerValues(std::string fileName, int startLayerLength) {
  std::vector<Data> res = std::vector<Data>();
  std::ifstream file(fileName);
  std::string line, val;

  std::getline(file, line);

  while (std::getline(file, line)) {
    VectorXd vector = VectorXd(startLayerLength);
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

}  // namespace neural_network
