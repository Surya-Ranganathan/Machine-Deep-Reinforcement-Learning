#include "main.hpp"

int main()
{
    vector x = {5,7,8,7,2,17,2,9,4,11,12,9,6};
    vector y = {99,86,87,88,111,86,103,87,94,78,77,85,86};

    LinearRegression lr(x, y);

    return 0;
}

LinearRegression::LinearRegression(const vector &x_, const vector &y_)
                            :   x(x_), y(y_), m(0.0), c(0.0), Meanx(0.0), Meany(0.0)
{
    size = x.size();
    Mean();

    Linear_Regression();
    std::cout << "m = " << m <<"\nc = " << c << std::endl;
}

void LinearRegression::Linear_Regression()
{
    double numerator = 0.0;
    double denominator =  0.0;

    for(int i = 0; i < size; i++)
    {
        numerator += (x[i] - Meanx) * (y[i] - Meany);
        denominator += std::pow((x[i] - Meanx), 2);
    }

    m = numerator / denominator;
    c = Meany - (m * Meanx);

}

void LinearRegression::Mean()
{
    for(int i = 0; i < size; i++)
    {
        Meanx += x[i];
        Meany += y[i];
    }
    Meanx /= size;
    Meany /= size;

}