#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
#include <Eigen/Dense>
#include <string>

#include "../src/CostFuc.h"


int main(){
// Test cost function class
    std::cout << "----------- Cost ----------" << std::endl;
    Eigen::MatrixXd AL(3,1);
    AL(0,0) = 0.9;
    AL(1,0) = 0.3;
    AL(2,0) = 0.5;


    Eigen::MatrixXd Y(3,1);
    Y(0,0) = 1;
    Y(1,0) = 1;
    Y(2,0) = 0;
    
    CostFuc cost("Binary Cross-entropy");
    double cost_res;
    cost_res = cost.calculate_cost(AL,Y);
    std::cout << "cost = " << cost_res << std::endl;    
    std::cout << "---------- Cost OK -----------" << std::endl;

    std::cout << "Z"+std::to_string(1) << std::endl;
    
    Eigen::MatrixXd cost_deriv;
    cost_deriv = cost.derivative(AL,Y);
    std::cout << "dAL = " << cost_deriv << std::endl;   
    return 0;
}



