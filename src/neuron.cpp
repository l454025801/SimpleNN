#pragma once

#include <stdexcept> // for std::runtime_error
#include <iostream> // for std::cerr, std::cout
#include <ostream> // for std::ostream

Neuron::get_z(){
    std::cout << z << std::endl;
    return z;
}

Neuron::get_a(){
    std::cout << a << std::endl;
    return a;
}

Neuron::get_layer(){
    std::cout << layer << std::endl;
    return layer;
}