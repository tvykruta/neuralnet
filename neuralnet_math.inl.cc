// Helper functions for computing neural net. Put all math equations here.

#include <vector>
#include <numeric>
#include <math.h>

using namespace std;

double ComputeNode(const vector<double> &weights, const vector<double> &thetas);
double sigmoid(const double x, double scale);
double dSigmoid(const double val);
inline float rough_sigmoid(float value);


// COmputes new value of node by applying activation function to previous layer
// node values and weights.
//
// Activation func: SUM(weight.0 * theta.0 ... weight.n, theta.n )
double ComputeNode(const vector<double> &weights, const vector<double> &thetas) {
  double value = std::inner_product(begin(weights), end(weights), begin(thetas), 0.0);
  return value;
}

// Sigmoid and derivative of sigmoid
// f = 1/(1+exp(-x))
// df = f * (1 - f)
double dSigmoid(const double val) {
  return val * (1.0 - val);
}


// Sigmoid function. Fits value where:
// f = 1/(1+exp(-x))
// 0.0 > result < 1.0 and for +val, result > 0.5 and -val, result < 0.5.l
double sigmoid(const double x, double scale = 1.0) {
     double exp_value;
     double return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x*scale);

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