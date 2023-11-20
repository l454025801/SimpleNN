#pragma once

#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
#include <string>
#include <Eigen/Dense>

class CostFuc {
    public:
        CostFuc();
        CostFuc(std::string type_fcx);
        void get_LossFuc();
        double calculate_cost(Eigen::MatrixXd AL, Eigen::MatrixXd Y);
        Eigen::MatrixXd derivative(Eigen::MatrixXd AL, Eigen::MatrixXd Y);
    private:
        std::string type_fcx_;    
};