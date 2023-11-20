#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
#include "Layer.h"

// initialization 
Layer::Layer(){
}


Layer::Layer(int num_neurons) {
    num_neurons_ = num_neurons;
    depth_ = 0;
    Activation act;
    activation_ = act;
    dropout_ = 0;
}

Layer::Layer(Activation activation){
    activation_ = activation;
    num_neurons_ = 0;
    dropout_ = 0;
    depth_ = 0;
}
    
Layer::Layer(int num_neurons, Activation activation){
    activation_ = activation;
    num_neurons_ = num_neurons;
    dropout_ = 0;
    depth_ = 0;    
}

Layer::Layer(int num_neurons, std::string activation){
    Activation act(activation);
    activation_ = act;
    num_neurons_ = num_neurons;
    dropout_ = 0;
    depth_ = 0;    
}

Layer::Layer(int num_neurons, Activation activation, float dropout){
    activation_ = activation;
    num_neurons_ = num_neurons;
    dropout_ = dropout;
    depth_ = 0;    
}

Layer::Layer(int num_neurons, std::string activation, float dropout){
    Activation act(activation);
    activation_ = act;
    num_neurons_ = num_neurons;
    dropout_ = dropout;
    depth_ = 0;    
}


// public functions

    // informational function
int Layer::get_num_neurons(){
    return num_neurons_;
}

int Layer::get_depth(){
    return depth_;
}

int Layer::rows(){
    return num_neurons_;
}

int Layer::cols(){
    return num_example_;    
}

Eigen::MatrixXd Layer::get_Z(){
    Eigen::MatrixXd z(1,1);
    return z;
}

Eigen::MatrixXd Layer::get_A(){
    Eigen::MatrixXd a(1,1);
    return a;
}


    // usage function
void Layer::set_num_neurons(int num){
    num_neurons_ = num;
}

void Layer::set_activation(Activation activation){
    activation_ = activation;
}

Eigen::MatrixXd Layer::activate(Eigen::MatrixXd Z){
    Eigen::MatrixXd A;
    A = activation_.activate(Z);
    return A;
}

Eigen::MatrixXd Layer::act_deriv(Eigen::MatrixXd Z){
    Eigen::MatrixXd A_prime;
    A_prime = activation_.derivative(Z);
    return A_prime;
}


// private functions
void Layer::set_depth(int num){
    depth_ = num;
}
