#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <numeric>
#include <algorithm>

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

std::unordered_map<char, int> tokenize(std::vector<std::string> v) {
  auto concat = std::accumulate(v.begin(), v.end(), std::string(""));
  std::sort(concat.begin(), concat.end());

  std::unordered_map<char, int> um;
  int count = 0;
  for (const char c : concat) {
    if (um.find(c) == um.end()) {
      um[c] = count;
      count++;
    }
  }

  return um;
} 

int main() {
  auto data = read_dataset("input.txt");
  std::cout << data.size() << " " << data[0] << std::endl;

  auto um = tokenize(data);
  std::cout << um['a'] << std::endl;

  return 0;
}
