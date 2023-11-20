#include "../src/SimpleNN.h"

//#include "../src/Activation.h"
//#include "../src/Layer.h"
//#include "../src/CostFuc.h"
//#include "../src/utility.h"

int main(){
    /*
    SimpleNN nn1;
    nn1.info();
    std::cout << "-----------------------------------------------------------------------" << std::endl;
    Eigen::MatrixXd X(3,2);
    Eigen::MatrixXd Y(1,2);
    
    X(0,0) = 0.123;
    X(1,0) = 0.325;
    X(2,0) = 0.436;
    X(0,1) = 0.754;
    X(1,1) = 0.463;
    X(2,1) = 0.253;
    
    Y(0,0) = 0;
    Y(0,1) = 1;
    
    nn1.X_train = X;
    nn1.Y_train = Y;
    nn1.alpha = 0.02;
    nn1.iter_times = 10;
    Activation act("sigmoid");
    nn1.add_layer(5, act);
    nn1.info();
    std::cout << "-----------------------------------------------------------------------" << std::endl;
    
    nn1.initialize_parameters();
    std::cout << nn1.get_parameters()["W1"].rows() << std::endl;
    std::cout << nn1.get_parameters()["W1"].cols() << std::endl;
    std::cout << nn1.get_parameters()["W1"] << std::endl;
    std::cout << nn1.get_parameters()["b1"] << std::endl;
    nn1.for_prop(nn1.X_train);
    */
    
    SimpleNN coursera_repro;
    coursera_repro.add_layer(20,"ReLU");
    coursera_repro.add_layer(7,"ReLU");
    coursera_repro.add_layer(5,"ReLU");
    coursera_repro.add_layer(1,"sigmoid");
    coursera_repro.set_cost("Binary Cross-entropy");
    coursera_repro.iter_times = 2500;
    coursera_repro.alpha = 0.0075;
    coursera_repro.train_output_frequency = 250;
    
    Eigen::MatrixXd X(3,2);
    Eigen::MatrixXd Y(1,2);
    
    X(0,0) = 0.123;
    X(1,0) = 0.325;
    X(2,0) = 0.436;
    X(0,1) = 0.754;
    X(1,1) = 0.463;
    X(2,1) = 0.253;
    
    Y(0,0) = 0;
    Y(0,1) = 1;
    
    coursera_repro.X_train = X;
    coursera_repro.Y_train = Y;
    
    coursera_repro.train();
    
    return 0;
}