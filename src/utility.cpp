#include "utility.h"

Eigen::MatrixXd min_myver(int num, Eigen::MatrixXd matrix){
    Eigen::MatrixXd new_matrix(matrix.rows(), matrix.cols());
    for (int i=0; i<matrix.rows(); i++){
        for (int j=0; j<matrix.cols(); j++){
            if (num > matrix(i,j)){new_matrix(i,j) = matrix(i,j);}
            else{new_matrix(i,j) = num;}
        }
    }
    return new_matrix;
}

Eigen::MatrixXd min_myver(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2){
    if (matrix1.rows() != matrix2.rows() || matrix1.cols() != matrix2.cols()){
        throw std::invalid_argument("Input Matrices have difference dimensions");
    }
    Eigen::MatrixXd new_matrix(matrix1.rows(), matrix2.cols());
    for (int i=0; i<matrix1.rows(); i++){
        for (int j=0; j<matrix1.cols(); j++){
            if (matrix1(i,j) > matrix2(i,j)){new_matrix(i,j) = matrix2(i,j);}
            else{new_matrix(i,j) = matrix1(i,j);}
        }
    }
    return new_matrix;
}

Eigen::MatrixXd max_myver(int num, Eigen::MatrixXd matrix){
    Eigen::MatrixXd new_matrix(matrix.rows(), matrix.cols());
    for (int i=0; i<matrix.rows(); i++){
        for (int j=0; j<matrix.cols(); j++){
            if (num > matrix(i,j)){new_matrix(i,j) = num;}
            else{new_matrix(i,j) = matrix(i,j);}
        }
    }
    return new_matrix;
}

Eigen::MatrixXd max_myver(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2){
    if (matrix1.rows() != matrix2.rows() || matrix1.cols() != matrix2.cols()){
        throw std::invalid_argument("Input Matrices have difference dimensions");
    }
    Eigen::MatrixXd new_matrix(matrix1.rows(), matrix2.cols());
    for (int i=0; i<matrix1.rows(); i++){
        for (int j=0; j<matrix1.cols(); j++){
            if (matrix1(i,j) > matrix2(i,j)){new_matrix(i,j) = matrix1(i,j);}
            else{new_matrix(i,j) = matrix2(i,j);}
        }
    }
    return new_matrix;
}