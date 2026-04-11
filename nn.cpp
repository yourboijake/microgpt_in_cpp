#include "value.h"
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

class Matrix {
public:
  std::vector<std::vector<Value>> values;

  Matrix(int nout, int nin, float stdev = 0.08) {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dist{0.0f, stdev};

    for (int i = 0; i < nout; i++) {
      std::vector<Value> row;
      for (int j = 0; j < nin; j++) {
        auto val = Value{dist(gen)};
        row.push_back(val);
      }
      values.push_back(row);
    }
  }

  int num_elements() {
    if (this->values.size() == 0)
      return 0;
    return this->values.size() * this->values[0].size();
  }
};

class Model {
public:
  int n_embd;
  int n_head;
  int n_layer;
  int block_size;
  int head_dim;
  int vocab_size;
  std::unordered_map<std::string, Matrix> state_dict;

  Model(int ne, int nh, int nl, int bs, int hd, int vs) {
    n_embd = ne;
    n_head = nh;
    n_layer = nl;
    block_size = bs;
    vocab_size = vs;
    head_dim = ne / nh;

    state_dict = {
        {"wte", Matrix{vocab_size, n_embd}},
        {"wpe", Matrix{block_size, n_embd}},
        {"lm_head", Matrix{vocab_size, n_embd}},
    };
  }

  int param_count() { return 0; }
};

int main() {
  Matrix m = Matrix{10, 10};

  return 0;
}
