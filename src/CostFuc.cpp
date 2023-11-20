#include <string>
#include <Eigen/Dense>
#include <list>
#include <algorithm>
#include "CostFuc.h"

CostFuc::CostFuc(){
    type_fcx_ = "Distance";
}

CostFuc::CostFuc(std::string type_fcx){
    std::list<std::string> supported_cost{"MSE","Distance","RMSE","Binary Cross-entropy","Exponential Cost","Hellinger distance","KL divergence"};
    if (std::find(supported_cost.begin(),supported_cost.end(),type_fcx) != supported_cost.end()) {
        type_fcx_ = type_fcx;
    }
    else {throw std::runtime_error("type of cost function not supported");}
}

void CostFuc::get_LossFuc(){
    std::cout << "Loss function: " << type_fcx_ << std::endl;
}

double CostFuc::calculate_cost(Eigen::MatrixXd AL, Eigen::MatrixXd Y){
    int sample_number;
    sample_number = AL.cols();
    if (type_fcx_ == "Distance"){return (AL-Y).sum()/sample_number;}
    else if (type_fcx_ == "MSE"){return (AL-Y).sum()/sample_number;}
    else if (type_fcx_ == "Binary Cross-entropy"){return (Y.array()*log(AL.array())+(1-Y.array())*log(1-AL.array())).sum()/(-sample_number);}
    //else if ()
    else {std::cout << "Unknown Loss function" << std::endl;}
}

Eigen::MatrixXd CostFuc::derivative(Eigen::MatrixXd AL, Eigen::MatrixXd Y){
    if (type_fcx_ == "Binary Cross-entropy"){return Y.array()*(1/AL.array())-(1-Y.array())/(1-AL.array());}
    //else if(){}
    else {std::cout << "Unknown Loss function" << std::endl;}
}
