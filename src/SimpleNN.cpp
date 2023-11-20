#include "SimpleNN.h"


// Initialization
SimpleNN::SimpleNN(){

    std::unordered_map<std::string, Layer> layers;
    layers_ = layers;
    depth_ = 0;
    
    std::pair<Matrix_Dictionary, Matrix_Dictionary> caches;
    caches_ = caches;

    Matrix_Dictionary parameters_;
    
    Matrix_Dictionary gradients_;
}

// manipulate NN structure
void SimpleNN::add_layer(int num_neurons, std::string activation){

    Layer new_layer(num_neurons, activation);
    //new_layer.set_depth(depth)
    depth_++;
    layers_["L"+std::to_string(depth_)] = new_layer;
}

void SimpleNN::set_cost(std::string costfc){
    CostFuc cost(costfc);
    cost_fuc = cost;
}
// initialize parameters after fed input
void SimpleNN::initialize_parameters(){
    
    // number of samples
    int m = X_train.cols();
    
    parameters_["W1"] = Eigen::MatrixXd::Random(layers_["L1"].rows(), X_train.rows());
    parameters_["b1"] = Eigen::MatrixXd::Zero(layers_["L1"].rows(), m);
    
    if (depth_ == 1){
        return;
    }
    
    for (int i=2; i<depth_+1; i++) {
        std::string dep = std::to_string(i);
        parameters_["W"+dep] = Eigen::MatrixXd::Random(layers_["L"+dep].rows(), layers_["L"+std::to_string(i-1)].rows());
        parameters_["b"+dep] = Eigen::MatrixXd::Zero(layers_["L"+dep].rows(), m);
    }
}

// cout information about the NN
void SimpleNN::info(){
    std::cout << "Simple Neural Network" << std::endl;
    std::cout << "------- Input info ---------" << std::endl;
    std::cout << "  Sample size: " << X_train.cols() << std::endl;
    std::cout << "  Number of features: " << X_train.rows() << std::endl;
    std::cout << "------- Output info --------" << std::endl;
    std::cout << "  Number of lables: " << Y_train.rows() << std::endl;
    std::cout << "------- Layer info ---------" << std::endl;
    std::cout << "  Number of layers: " << layers_.size() << std::endl;
    std::cout << "------- Training info ------" << std::endl;
    std::cout << "  Number of iteration: " << iter_times << std::endl;
    std::cout << "  Learning rate: " << alpha << std::endl;
}

void SimpleNN::get_layers_info(){
    
}

Matrix_Dictionary SimpleNN::get_parameters(){
    return parameters_;
}

/* -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Major Training Fucntions
SimpleNN.train() calls the following functions to train

input -> forward prop -> compute cost -> backward prop -> update parameters
  ^                                                               |
  |                                                               |
  -----------------------------------------------------------------
   
*/

// forward prop
// return Zs and As as a pair
void SimpleNN::for_prop(Eigen::MatrixXd X_input){
    // first declare Z, A, caches
    Matrix_Dictionary Z;
    Matrix_Dictionary A;
    
    // first layer
    Z["Z1"] = parameters_["W1"] * X_input + parameters_["b1"];
    A["A1"] = layers_["L1"].activate(Z["Z1"]);
    
    // other hidden layers
    for (int i=2; i<depth_+1; i++) {
        std::string dep = std::to_string(i);
        Z["Z"+dep] = parameters_["W"+dep] * A["A"+std::to_string(i-1)] + parameters_["b"+dep];
        A["A"+dep] = layers_["L"+dep].activate(Z["Z"+dep]);
    }    
    
    caches_.first = Z;
    caches_.second = A;
    
    //std::cout << A["A1"] << std::endl;
    //std::cout << "------------------" << std::endl;
    //std::cout << A["A2"] << std::endl;
    //std::cout << "------------------" << std::endl;
    //std::cout << A["A3"] << std::endl;
    //std::cout << "------------------" << std::endl;
    //std::cout << A["A4"] << std::endl;
    //std::cout << "------------------" << std::endl;

    //std::cout << "---- forward prop done -----" << std::endl;
}

// compute the cost for the iteration
void SimpleNN::compute_cost(Eigen::MatrixXd AL, Eigen::MatrixXd Y_real){
    double cost_value = cost_fuc.calculate_cost(AL, Y_real);
    //std::cout << "The cost is " << cost_value << std::endl;
    costs_.push_back(cost_value);
}

// backward prop
void SimpleNN::back_prop(){
    /*
    remeber to implement the second last layer computation
    */
    
    // if the layer is the last layer, take the derivative of the cost function first
    Matrix_Dictionary Z = caches_.first;
    Matrix_Dictionary A = caches_.second;
    
    // number of samples
    int m = X_train.cols();
    
    // last layer
    std::string max_dep = std::to_string(depth_);
    
    gradients_["dA"+max_dep] = cost_fuc.derivative(A["A"+max_dep], Y_train);
    gradients_["dZ"+max_dep] = gradients_["dA"+max_dep].array() * layers_["L"+max_dep].act_deriv(Z["Z"+max_dep]).array();
    gradients_["dW"+max_dep] = (gradients_["dZ"+max_dep] * A["A"+std::to_string(depth_-1)].transpose())/m;
    gradients_["db"+max_dep] = gradients_["dZ"+max_dep].rowwise().sum()/m;
    
    
    // rest of the layers
    for (int i=depth_-1; i>1; i--) {
        std::string dep = std::to_string(i);
        // save one step
        //gradients_["dA"+std::to_string(i)] = parameters_["W"+std::to_string(i+1)].transpose()*gradients_["dZ"+std::to_string(i+1)];
        
        gradients_["dZ"+dep] = (parameters_["W"+std::to_string(i+1)].transpose()*gradients_["dZ"+std::to_string(i+1)]).array()*layers_["L"+dep].act_deriv(Z["Z"+dep]).array();
        gradients_["dW"+dep] = (gradients_["dZ"+dep] * A["A"+std::to_string(i-1)].transpose())/m;
        gradients_["db"+dep] = gradients_["dZ"+dep].rowwise().sum()/m;
    }
    
    // first layer
    gradients_["dZ1"] = (parameters_["W2"].transpose()*gradients_["dZ2"]).array()*layers_["L2"].act_deriv(Z["Z1"]).array();
    gradients_["dW1"] = (gradients_["dZ1"] * X_train.transpose())/m;
    gradients_["db1"] = gradients_["dZ1"].rowwise().sum()/m;    
    
    //std::cout << "---- backward prop done -----" << std::endl;
}

// update parameters
void SimpleNN::update_parameters(){
    int m = X_train.cols();
    
    for (int i=1; i<depth_+1; i++) {
        std::string dep = std::to_string(i);
        
        parameters_["W"+dep] = parameters_["W"+dep] - alpha*gradients_["dW"+dep];
        
        Eigen::MatrixXd broadcast_db(gradients_["db"+dep].rows(), gradients_["db"+dep].cols()*m);
        parameters_["b"+dep] = parameters_["b"+dep] - alpha*broadcast_db;
    }
}

void SimpleNN::train(){
    info();
    
    // initialization
    initialize_parameters();
    
    for (int i=0; i<iter_times; i++){
        for_prop(X_train);
        //std::cout << caches_.second["A"+std::to_string(depth_)] << std::endl;
        
        compute_cost(caches_.second["A"+std::to_string(depth_)], Y_train);
        if (i%train_output_frequency == 0){
            std::cout << "Iteration " << i << std::endl;
            std::cout << "Cost = " << costs_[i] << std::endl;
        }
        
        back_prop();
        update_parameters();
        
    }
}