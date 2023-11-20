#include <Eigen/Dense>
#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
#include <cmath>
#include <vector>
#include <unordered_map>

int main(){
    Eigen::MatrixXd A3(5,2);
    A3(0,0) = 0.726914;
    A3(0,1) = 0.342796;
    A3(1,0) = 0;
    A3(1,1) = 0;
    A3(2,0) = 0;
    A3(2,1) = 0;
    A3(3,0) = 0.0703502;
    A3(3,1) = 0;
    A3(4,0) = 0;
    A3(4,1) = 0.0309917;
    
    Eigen::MatrixXd dZ4(1,2);
    dZ4(0,0) = -0.337299;
    dZ4(0,1) = 0.58833;
    std::cout << A3.transpose() << std::endl;
    std::cout << (dZ4 * A3.transpose())/2 << std::endl;
    std::cout << 0.726914 * -0.337299 + 0.342796 * 0.58833 << std::endl;
    return 0;
}