#include <iostream>
#include <vector>
#include <algorithm> // for std::shuffle
#include <random>    // for std::default_random_engine
#include <ctime>     // for std::time

int main() 
{
    std::vector<int> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};

    // Seed random number generator for reproducibility
    std::default_random_engine rng(42); // fixed seed like random_state=42
    std::shuffle(x.begin(), x.end(), rng);

    // Split ratio
    double test_ratio = 0.2;
    size_t test_size = static_cast<size_t>(x.size() * test_ratio);
    size_t train_size = x.size() - test_size;

    // Split into train and test
    std::vector<int> train(x.begin(), x.begin() + train_size);
    std::vector<int> test(x.begin() + train_size, x.end());

    // Print train data
    std::cout << "Train: ";
    for (int val : train) {
        std::cout << val << " ";
    }

    std::cout << "\nTest: ";
    for (int val : test) {
        std::cout << val << " ";
    }

    std::cout << std::endl;

    return 0;
}
