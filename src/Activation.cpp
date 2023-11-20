#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
#include <list>
#include <algorithm>
#include "Activation.h"


Activation::Activation() {
    type_fcx_ = "sigmoid";
}

Activation::Activation(std::string type_fcx) {
    std::list<std::string> supported_act{"sigmoid","tanh","ReLU","leaky ReLU","elu","softplus"} ;
    if (std::find(supported_act.begin(),supported_act.end(),type_fcx) != supported_act.end()) {
        type_fcx_ = type_fcx;
    }
    else {throw std::runtime_error("type of activation function not supported");}
}

void Activation::get_activation(){
    std::cout << "Activation function: " << type_fcx_ << std::endl;
}

Eigen::MatrixXd Activation::activate(Eigen::MatrixXd Z){
    if (type_fcx_ == "sigmoid"){return 1/(1+exp(-Z.array()));}
    else if (type_fcx_ == "tanh"){return tanh(Z.array());}
    else if (type_fcx_ == "ReLU"){return max_myver(0, Z);}
    //else if (type_fcx_ == "leaky ReLU"){return min(a*Z.array(), Z.array());}
    //else if (type_fcx_ == "elu"){return min(a*((exp(Z.array()-1))), Z.array());}
    else if (type_fcx_ == "softplus"){return log(1+exp(Z.array()));}
    else {std::cout << "Unknown activation function" << std::endl;}
}

Eigen::MatrixXd Activation::derivative(Eigen::MatrixXd Z){
    if (type_fcx_ == "sigmoid"){return exp(-Z.array())/pow(exp(-Z.array())+1,2);}
    // if (type_fcx_ == "tanh"){return pow(sech(Z.array()),2);}
    else if (type_fcx_ == "ReLU"){return max_myver(Eigen::MatrixXd::Zero(Z.rows(), Z.cols()), Z);}
    //else if (type_fcx_ == "leaky ReLU"){return min(a*Z.array(), Z.array());}
    //else if (type_fcx_ == "elu"){return min(a*((exp(Z.array()-1))), Z.array());}
    else if (type_fcx_ == "softplus"){return log(1+exp(Z.array()));}
    else {std::cout << "Unknown activation function" << std::endl;}
}