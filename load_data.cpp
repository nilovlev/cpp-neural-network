#include "load_data.h"

namespace neural_network {

Data LoadData::read(const std::string& filePath) {
  std::ifstream file(filePath);
  std::string line;
  std::getline(file, line);
  Index pixelCount = std::count(line.begin(), line.end(), ',');
  Index pictureCount = 0;
  while (std::getline(file, line)) {
    ++pictureCount;
  }
  Data res = {Matrix(pictureCount, pixelCount), Vector(pictureCount)};
  file.clear();
  file.seekg(0);
  std::getline(file, line);
  
  Index pictureIndex = 0;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    Index pixelIndex = 0;
    std::string val;
    std::getline(ss, val, ',');
    res.answer(pictureIndex) = std::stoi(val);
    while (std::getline(ss, val, ',')) {
      res.input(pictureIndex, pixelIndex) = std::stod(val) / 255;
      ++pixelIndex;
    }
    ++pictureIndex;
  }

  return res;
}

}  // namespace neural_network
