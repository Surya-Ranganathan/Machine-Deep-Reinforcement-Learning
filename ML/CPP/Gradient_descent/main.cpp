#include "main.hpp"

int main(void)
{
    GradientDescent GraDes;

    GraDes.gradient_descent();
    return 0;
}

GradientDescent::GradientDescent()
{
    file.open("output.csv", std::ios::out);
    learning_rate = .2;
    theta = -5.0;

    file << "theta,f(theta)\n";
}

GradientDescent::~GradientDescent()
{
    file.close();
}

double GradientDescent::function(double x)
{
    return x * x;
}

double GradientDescent::first_derivative(double x)
{
    return 2 * x;
}

void GradientDescent::gradient_descent()
{
    for (int i = 0; i < 20; i++)
    {
        file << theta << "," << function(theta) << std::endl;
        
        theta = theta - learning_rate * first_derivative(theta); 
    }
}
