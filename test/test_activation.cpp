#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
#include <Eigen/Dense>
#include <string>

#include "../src/CostFuc.h"
#include "../src/Activation.h"

int main() {
// Test activation function class
    std::cout << "---------- Activation -----------" << std::endl;
    
    Eigen::MatrixXd Z1(3,2);
    Z1(0,0) = 1;
    Z1(1,0) = 3;
    Z1(2,0) = 5;
    Z1(0,1) = 2;
    Z1(1,1) = 4;
    Z1(2,1) = 6;
    Activation activation("ReLU");
    Eigen::MatrixXd A1;
    A1 = activation.activate(Z1);
    std::cout << A1 << std::endl;
    Activation act2("ReLU");
    std::cout << "---------- Activation OK -----------" << std::endl;    
    return 0;
}