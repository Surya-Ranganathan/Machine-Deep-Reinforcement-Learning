#ifndef MAIN_HPP
#define MAIN_HPP

#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

typedef std::vector<double> vector;

#define ALPHA   0.01

class LinearRegression
{
    double m;       //slope
    double c;       //intercept
    int size;       //data size
    vector x;       //input data x
    vector y;       //input data y
    double error;
    vector ypred;

    void MeanSquareError();
    void GradientDescent();
    void R2Score();

    public:
    LinearRegression();
    void linear_regression();
};

#endif