#pragma once

#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
#include <unordered_map>
#include <Eigen/Dense>

// Define a syntax for unordered map containing string names and a matrix
// Function overloads for min;

using Matrix_Dictionary = std::unordered_map<std::string, Eigen::MatrixXd>;


// Function overload of min with Eigen::Matrix
Eigen::MatrixXd max_myver(int num, Eigen::MatrixXd matrix);
Eigen::MatrixXd max_myver(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2);

Eigen::MatrixXd min_myver(int num, Eigen::MatrixXd matrix);
Eigen::MatrixXd min_myver(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2);

