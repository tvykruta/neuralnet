// Helper functions for computing neural net. Put all math equations here.

#include <vector>
#include <numeric>
#include <math.h>

using namespace std;

float ComputeNode(const vector<double> &weights, const vector<double> &thetas);
float sigmoid(float x);
inline float rough_sigmoid(float value);


// COmputes new value of node by applying activation function to previous layer
// node values and weights.
//
// Activation func: SUM(weight.0 * theta.0 ... weight.n, theta.n )
float ComputeNode(const vector<double> &weights, const vector<double> &thetas) {
  float value = std::inner_product(begin(weights), end(weights), begin(thetas), 0.0);
  return value;
}

// Sigmoid function. Fits value where:
// 0.0 > result < 1.0 and for +val, result > 0.5 and -val, result < 0.5.l
float sigmoid(float x) {
     float exp_value;
     float return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}

// Approximation of sigmoid.
inline float rough_sigmoid(float value) {
    float x = fabs(value);
    float x2 = x*x;
    float e = 1.0f + x + x2*0.555f + x2*x2*0.143f;
    return 1.0f / (1.0f + (value > 0 ? 1.0f / e : e));
}