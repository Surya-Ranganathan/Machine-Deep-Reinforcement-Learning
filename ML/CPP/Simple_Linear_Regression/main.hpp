#ifndef MAIN_HPP
#define MAIN_HPP

#include <iostream>
#include <cmath>
#include <vector>

typedef std::vector<double> vector;

class LinearRegression
{
    double m;                   // Slope
    double c;                   // Intercept

    int size;                   //input size

    double Meanx;
    double Meany;

    vector x;
    vector y;

    void Linear_Regression();
    void Mean();
    
    public:
    LinearRegression(const vector &, const vector &);
};

#endif