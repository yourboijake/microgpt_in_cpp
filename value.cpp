#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include "value.h"

Value::Value(float dt, std::vector<std::shared_ptr<Value>> chn, std::vector<float> lgs) {
    data = dt;
    grad = 0;
    children = chn;
    local_grads = lgs;
  }

std::shared_ptr<Value> Value::add (std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
  float sum = a->data + b->data;
  auto out = std::make_shared<Value>(sum, {a, b}, {1.0, 1.0});
  return out;
}

std::shared_ptr<Value> Value::sub(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
  auto n = Value::neg(b);
  return Value::add(a, n);
}

std::shared_ptr<Value> Value::mult(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
  float prod = a->data * b->data;
  auto out = std::make_shared<Value>(prod, {a, b}, {b->data, a->data});
  return out;
}

std::shared_ptr<Value> Value::div(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
  auto inv = b.pow(-1.0f);
  return Value::mult(a, inv);
}

std::shared_ptr<Value> Value::pow(std::shared_ptr<Value> v, float exp) {
  auto out = Value(std::pow(v->data, exp), {v}, {exp * std::pow(v->data, exp - 1.0f)});
  return out;
}

std::shared_ptr<Value> Value::log(std::shared_ptr<Value> v) {
  auto out = std::make_shared<Value>(std::log(v->data), {v}, {1/v->data});
  return out;
}

std::shared_ptr<Value> Value::exp(std::shared_ptr<Value> v) {
  float e = std::exp(v->data);
  auto out = std::make_shared<Value>(e, {v}, {e});
  return out;
}

std::shared_ptr<Value> Value::relu(std::shared_ptr<Value> v) {
  float d = v->data > 0.0f ? v->data : 0.0f;
  float deriv = v->data > 0.0f ? 1.0f : 0.0f;
  auto out = make_shared<Value>(d, {v}, {deriv});
  return out;
}

std::shared_ptr<Value> Value::neg(std::shared_ptr<Value> v) {
  auto v = Value(-1.0f);
  auto out = std::make_shared<Value>(v->data * -1.0f, {v}, {-1.0f});
  return out;
}

std::vector<std::shared_ptr<Value>> Value::get_children() const {
  return this->children;
}

std::vector<float> Value::get_local_grads() const {
  return this->local_grads;
}

void Value::backward(std::shared_ptr<Value> root) {
  std::vector<Value> topo = {};
  std::unordered_set<std::shared_ptr<Value>> visited = {};
  void build_topo(std::shared_ptr<Value> v, std::vector<Value> t, std::unordered_set<std::shared_ptr<Value>> vis);
  build_topo(root, topo, visited);
  root->grad = 1.0f;
  std::reverse(topo.begin(), topo.end());
  for (const Value& v : topo) {
    auto children = v.get_children();
    auto lgs = v.get_local_grads();
    for (int i = 0; i < children.size(); i++) {
      children[i]->grad += lgs[i] * v.grad;
    }
  }
}

void build_topo(std::shared_ptr<Value> v, std::vector<Value> topo, std::unordered_set<std::shared_ptr<Value>> visited) {
  if (visited.find(v) == visited.end()) {
    visited.insert(v);
    for (const std::shared_ptr<Value> child : v.get_children()) {
      return build_topo(child, topo, visited);
    }
    topo.push_back(v);
  }
}

/*
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
*/
