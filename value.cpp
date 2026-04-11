#include "value.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <vector>

Value::Value(float dt, std::vector<std::shared_ptr<Value>> chn,
             std::vector<float> lgs) {
  data = dt;
  grad = 0;
  children = chn;
  local_grads = lgs;
}

std::shared_ptr<Value> Value::add(std::shared_ptr<Value> a,
                                  std::shared_ptr<Value> b) {
  float sum = a->data + b->data;
  std::vector<std::shared_ptr<Value>> chn = {a, b};
  std::vector<float> lgs = {1.0, 1.0};
  auto out = std::make_shared<Value>(sum, chn, lgs);
  return out;
}

std::shared_ptr<Value> Value::sub(std::shared_ptr<Value> a,
                                  std::shared_ptr<Value> b) {
  auto n = Value::neg(b);
  return Value::add(a, n);
}

std::shared_ptr<Value> Value::mult(std::shared_ptr<Value> a,
                                   std::shared_ptr<Value> b) {
  float prod = a->data * b->data;
  auto chn = {a, b};
  auto lgs = {b->data, a->data};
  auto out = std::make_shared<Value>(prod, chn, lgs);
  return out;
}

std::shared_ptr<Value> Value::div(std::shared_ptr<Value> a,
                                  std::shared_ptr<Value> b) {
  auto inv = Value::pow(b, -1.0f);
  return Value::mult(a, inv);
}

std::shared_ptr<Value> Value::pow(std::shared_ptr<Value> v, float exp) {
  float data = std::pow(v->data, exp);
  float deriv = exp * std::pow(v->data, exp - 1.0f);
  auto chn = {v};
  auto lgs = {deriv};
  auto out = std::make_shared<Value>(data, chn, lgs);
  return out;
}

std::shared_ptr<Value> Value::log(std::shared_ptr<Value> v) {
  float data = std::log(v->data);
  auto chn = {v};
  auto lgs = {1 / v->data};
  auto out = std::make_shared<Value>(data, chn, lgs);
  return out;
}

std::shared_ptr<Value> Value::exp(std::shared_ptr<Value> v) {
  float e = std::exp(v->data);
  auto chn = {v};
  auto lgs = {e};
  auto out = std::make_shared<Value>(e, chn, lgs);
  return out;
}

std::shared_ptr<Value> Value::relu(std::shared_ptr<Value> v) {
  float d = v->data > 0.0f ? v->data : 0.0f;
  float deriv = v->data > 0.0f ? 1.0f : 0.0f;
  auto chn = {v};
  auto lgs = {deriv};
  auto out = std::make_shared<Value>(d, chn, lgs);
  return out;
}

std::shared_ptr<Value> Value::neg(std::shared_ptr<Value> v) {
  float data = v->data * -1.0f;
  auto chn = {v};
  auto lgs = {-1.0f};
  auto out = std::make_shared<Value>(data, chn, lgs);
  return out;
}

std::vector<std::shared_ptr<Value>> Value::get_children() const {
  return this->children;
}

std::vector<float> Value::get_local_grads() const { return this->local_grads; }

void Value::backward(std::shared_ptr<Value> root) {
  std::vector<std::shared_ptr<Value>> topo = {};
  std::unordered_set<std::shared_ptr<Value>> visited = {};
  void build_topo(std::shared_ptr<Value> v,
                  std::vector<std::shared_ptr<Value>> t,
                  std::unordered_set<std::shared_ptr<Value>> vis);
  build_topo(root, topo, visited);
  root->grad = 1.0f;
  std::reverse(topo.begin(), topo.end());
  for (std::shared_ptr<Value> v : topo) {
    auto children = v->get_children();
    auto lgs = v->get_local_grads();
    for (size_t i = 0; i < children.size(); i++) {
      children[i]->grad += lgs[i] * v->grad;
    }
  }
}

void build_topo(std::shared_ptr<Value> v,
                std::vector<std::shared_ptr<Value>> topo,
                std::unordered_set<std::shared_ptr<Value>> visited) {
  if (visited.find(v) == visited.end()) {
    visited.insert(v);
    for (const std::shared_ptr<Value> &child : v->get_children()) {
      return build_topo(child, topo, visited);
    }
    topo.push_back(v);
  }
}
