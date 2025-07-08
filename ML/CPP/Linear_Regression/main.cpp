#include "main.hpp"

int main()
{
    LinearRegression lr;

    lr.linear_regression();
    
    return 0;
}

LinearRegression::LinearRegression()
                                :   x{5,7,8,7,2,17,2,9,4,11,12,9,6},
                                    y{99,86,87,88,111,86,103,87,94,78,77,85,86},
                                    m(0.0),
                                    c(0.0)
{
    size = x.size();
}

void LinearRegression::linear_regression()
{
    int EpochTime = 1000;

    while(EpochTime--)
        GradientDescent();

    R2Score();

    std::cout << "m = " << m << "\nc = " << c << std::endl;
}

// R² = 1 - SS.tot / SS.res
void LinearRegression::R2Score()
{
    double ss_tot = 1;
    double ss_res = 1;
    double mean = 0.0;

    for(int i = 0; i < size; i++)
        mean += y[i];
    mean /= size;

    for(int i = 0; i < size; i++)
    {
        ss_tot += ((y[i] - mean) * (y[i] - mean));
        ss_res += ((y[i] - ypred[i]) * (y[i] - ypred[i]));
    }

    double r2 = 1 - (ss_res / ss_tot);

    std::cout << "Traning R² Score : " << r2 * 100 << std::endl;
}

void LinearRegression::GradientDescent()
{
    double grad_m = 0.0;
    double grad_c = 0.0;
    ypred.clear();

    for(int i = 0; i < size; i++)
    {
        double prediction = m * x[i] + c;

        double error = prediction - y[i];

        grad_m += (error) * x[i];
        grad_c += (error);

        ypred.push_back(prediction);
    }
    
    grad_m *= (2.0 / size);
    grad_c *= (2.0 / size);

    m -= ALPHA * grad_m;
    c -= ALPHA * grad_c;
}