#pragma once

#include <unordered_set>
#include <memory>

class Value {

  private:
    std::vector<std::shared_ptr<Value>> children;
    std::vector<float> local_grads;

  public:
    float data;
    float grad;
    Value(float dt, std::vector<std::shared_ptr<Value>> chn = {}, std::vector<float> lgs = {});
    
    static std::shared_ptr<Value> add(std::shared_ptr<Value> a, std::shared_ptr<Value> b);
    
    static std::shared_ptr<Value> sub(std::shared_ptr<Value> a, std::shared_ptr<Value> b);
    
    static std::shared_ptr<Value> mult(std::shared_ptr<Value> a, std::shared_ptr<Value> b);
    
    static std::shared_ptr<Value> div(std::shared_ptr<Value> a, std::shared_ptr<Value> b);
    
    static std::shared_ptr<Value> pow(std::shared_ptr<Value> v, float exp);
    
    static std::shared_ptr<Value> log(std::shared_ptr<Value> v);
    
    static std::shared_ptr<Value> exp(std::shared_ptr<Value> v);
    
    static std::shared_ptr<Value> relu(std::shared_ptr<Value> v);
    
    static std::shared_ptr<Value> neg(std::shared_ptr<Value> v);
    
    static void backward(std::shared_ptr<Value>);
    
    std::vector<std::shared_ptr<Value>> get_children() const;
    
    std::vector<float> get_local_grads() const;
};

void build_topo(Value v, std::vector<Value> topo, std::unordered_set<Value*> visited);
