#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
//#include "layer.h"
#include <utility>
#include <unordered_map>
#include "../src/Activation.h"
#include "../src/Layer.h"

int main(){
    Layer L1;
    std::cout << L1.get_num_neurons() << std::endl;
    Eigen::MatrixXd a(2,1);
    a(0,0) = 1;
    a(1,0) = 2;
    std::cout << L1.activate(a) << std::endl;
    std::cout << L1.act_deriv(a) << std::endl;
    std::pair<std::unordered_map<int,int>, std::unordered_map<int,int>> b;
    std::unordered_map<std::string, int> s;
    s["1"] = 1;
    
    int i = 1;
    int c = i;
    int &d = i;
    i = 2;
    std::cout << c << std::endl;
    std::cout << d << std::endl;
    return 0;
    
    
}