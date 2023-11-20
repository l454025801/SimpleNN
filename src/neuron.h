#pragma once

#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream

class Neuron {
    public:
        double get_z();
        double get_a();
        int get_layer();
        
    private:
        double z;
        double a;
        int layer;
}