#ifndef MAIN_HPP
#define MAIN_HPP

#include <iostream>
#include <cmath>
#include <fstream>

class GradientDescent
{
    std::fstream file;
    double learning_rate;
    double theta;

    double function(double x);
    double first_derivative(double x);

    public:
    GradientDescent();
    ~GradientDescent();
    void gradient_descent();
};

#endif