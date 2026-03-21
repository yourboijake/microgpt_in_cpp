#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>

std::vector<std::string> read_dataset(std::string fpath) {
  std::vector<std::string> data;
  std::ifstream file(fpath);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file");
  }

  std::string line;
  while(std::getline(file, line)) {
    data.push_back(line);
  }

  return data;
}



int main() {
  auto data = read_dataset("input.txt");
  std::cout << data.size() << " " << data[0] << std::endl;

  return 0;
}
