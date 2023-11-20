#pragma once

#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
#include <Eigen/Dense>
#include <cmath>
#include "utility.h"

class Activation {
    public:
        Activation();
        Activation(std::string type_fcx);
        void get_activation();
        Eigen::MatrixXd activate(Eigen::MatrixXd Z);
        Eigen::MatrixXd derivative(Eigen::MatrixXd Z);
    private:
        std::string type_fcx_;    
};



/* 
sigmoid             sigma = 1 / (1+e^-z)

tanh                f = tanh(z)

ReLU                R = max(0,z)

leaky ReLU          f = az  z < 0   or f = max(az,z) [these are the same]
                      = z   z > 0
               
exponential linear unit  f = alpha(e^z - 1) z < 0
                           = z              z >= 0

SoftPlus            f = ln(1+e^z)
*/