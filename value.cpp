#include <iostream>
#include <vector>
#include <cmath>

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

    std::vector<Value*> get_children() {
      return this->children;
    }
};

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
