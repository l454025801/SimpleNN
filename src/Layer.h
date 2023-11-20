#pragma once

#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
#include <Eigen/Dense>
#include "Activation.h"

class Layer {
    public:
        // initialization    
        Layer();
        Layer(int num_neurons);
        Layer(Activation activation);
        Layer(int num_neurons, Activation activation);
        Layer(int num_neurons, std::string activation);
        Layer(int num_neurons, Activation activation, float dropout);
        Layer(int num_neurons, std::string activation, float dropout);
        // functions
        int get_num_neurons();
        void set_num_neurons(int num);
        int get_depth();
        void set_activation(Activation activation);
        Eigen::MatrixXd activate(Eigen::MatrixXd Z);
        Eigen::MatrixXd act_deriv(Eigen::MatrixXd Z);
        
        int rows();
        int cols();
        // return the matrix of Z / A
        Eigen::MatrixXd get_Z();
        Eigen::MatrixXd get_A();

    private:
        int depth_;
        int num_neurons_;
        int num_example_;
        Activation activation_;
        
        // dropout rate
        float dropout_;
        
        // The depth is determined when a layer is created in a neural network 
        void set_depth(int num);
        
        // Pointers to The Z and A value
        Eigen::MatrixXd* Z_;
        Eigen::MatrixXd* A_;
        
        // Z and A are determined by input, weight, bias and activation function
        
        Layer *next_layer;
        Layer *previous_layer;
};