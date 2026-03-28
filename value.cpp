#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <algorithm>

class Value {
  private:
    std::vector<Value*> children;
    std::vector<float> local_grads;

  public:
    float data;
    float grad;
    Value(float dt, std::vector<Value*> chn = {}, std::vector<float> lgs = {}) {
      data = dt;
      grad = 0;
      children = chn;
      local_grads = lgs;
    }

    Value operator+ (Value other) {
      float sum = this->data + other.data;
      return Value(sum, {this, &other}, {1.0, 1.0});
    }

    Value operator- (Value other) {
      return *this + other.neg();
    }

    Value operator* (Value other) {
      float prod = this-> data * other.data;
      return Value(prod, {this, &other}, {other.data, this->data});
    }

    Value operator/ (Value other) {
      return *this * other.pow(-1.0f);
    }

    Value pow(float exp) {
      return Value(std::pow(this->data, exp), {this}, {exp * std::pow(this->data, exp - 1.0f)});
    }

    Value log() {
      return Value(std::log(this->data), {this}, {1/this->data});
    }

    Value exp() {
      float e = std::exp(this->data);
      return Value(e, {this}, {e});
    }

    Value relu() {
      float d = this->data > 0.0f ? this->data : 0.0f;
      float deriv = this->data > 0.0f ? 1.0f : 0.0f;
      return Value(d, {this}, {deriv});
    }

    Value neg() {
      return *this * Value(-1.0f);
    }

    std::vector<Value*> get_children() const {
      return this->children;
    }

    std::vector<float> get_local_grads() const {
      return this->local_grads;
    }

    void backward() {
      std::vector<Value> topo = {};
      std::unordered_set<Value*> visited = {};
      void build_topo(Value v, std::vector<Value> t, std::unordered_set<Value*> vis);
      build_topo(*this, topo, visited);
      this->grad = 1.0f;
      std::reverse(topo.begin(), topo.end());
      for (const Value v : topo) {
        auto children = v.get_children();
        auto lgs = v.get_local_grads();
        for (int i = 0; i < children.size(); i++) {
          children[i]->grad += lgs[i] * v.grad;
        }
      }
    }
};

void build_topo(Value v, std::vector<Value> topo, std::unordered_set<Value*> visited) {
  if (visited.find(&v) == visited.end()) {
    visited.insert(&v);
    for (const Value* child : v.get_children()) {
      return build_topo(*child, topo, visited);
    }
    topo.push_back(v);
  }
}

int main() {
  Value v1 = {10};
  Value v2 = {20};
  auto v3 = v1 + v2;
  std::cout << v1.data << std::endl;
  std::cout << &v1 << std::endl;
  std::cout << v2.data << std::endl;
  std::cout << v3.data << std::endl;

  auto ch = v3.get_children();
  std::cout << ch[0] << " " << ch[0]->data << std::endl;

  auto v4 = v1 * v2;
  std::cout << v4.data << std::endl;

  auto v5 = v1.pow(2);
  std::cout << v5.data << std::endl;

  auto v6 = v1.log();
  std::cout << v6.data << std::endl;

  auto v7 = v1.neg();
  std::cout << v7.data << std::endl;

  auto v8 = v2 / v1;
  std::cout << v8.data << std::endl;

  auto v9 = v2 - v1;
  std::cout << v9.data << std::endl;

  return 0;
}
