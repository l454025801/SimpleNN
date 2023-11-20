#pragma once

#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream
#include <cmath>
#include <vector>
#include <unordered_map>

#include "utility.h"
#include "Layer.h"
#include "Activation.h"
#include "CostFuc.h"


class SimpleNN {
    public:
        SimpleNN();
        void get_layers_info();
        void add_layer(int num_neurons, std::string activation);
        void set_cost(std::string cost);
        void train();
        void info();
        Matrix_Dictionary get_parameters();
        
        // input
        Eigen::MatrixXd X_train;
        Eigen::MatrixXd Y_train;
        
        Eigen::MatrixXd X_test;
        Eigen::MatrixXd Y_test;
        
        // hyperparameters
        int iter_times;
        int train_output_frequency;
        CostFuc cost_fuc;
        float alpha;
        
        
        // --------------------move to private after all test----------------------------
        
        // parameter initialization
        void initialize_parameters();
        
        //forward propagation
        void for_prop(Eigen::MatrixXd X_input);
        
        //Cost calculation
        void compute_cost(Eigen::MatrixXd AL, Eigen::MatrixXd Y_real);
        
        //backward propagation
        void back_prop();
        
        //parameter update
        void update_parameters();
        
    private:
        std::unordered_map<std::string, Layer> layers_;
        int depth_;
        Matrix_Dictionary parameters_;
        Matrix_Dictionary gradients_;
        std::pair<Matrix_Dictionary, Matrix_Dictionary> caches_;
        std::vector<float> costs_;
};