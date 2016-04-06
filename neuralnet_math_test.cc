// Simple unit testing for NeuralNet class.
// We use the "CuTest" unit test library.
#include "includes/cutest.h"
#include "neuralnet_math.inl.cc"

using namespace std;

void test_ComputeNode() {
  vector<double> weights = { 1.0, 2.0};
  vector<double> thetas = { 4.0, 0.5 };

  float output = ComputeNode(weights, thetas);
  TEST_CHECK_(output == 5.0, "Got %f", output);
}

double round_1000th(double val) {
  return val;
  return floorf(val * 1000) / 1000;
}

void test_sigmoid() {
  double test_value = 5.0;
  double result = sigmoid(test_value);
  double expected = 0.9933071732521057129;
  TEST_CHECK_(result == expected, "Got %19.19f expected %f9.9", result, expected);

  test_value = -5.0;
  result = sigmoid(test_value);
  expected = 0.0066928509622812271;
  TEST_CHECK_(result == expected, "Got %19.19f expected %f9.9", result, expected);
}

void test_rough_sigmoid() {
  float diff = fabs(rough_sigmoid(5.0f) - sigmoid(5.0f));
  TEST_CHECK(diff < 0.01);
}

TEST_LIST = {
    { "ComputeNode",  test_ComputeNode },
    { "sigmoid",  test_sigmoid },
    { "rough_sigmoid", test_rough_sigmoid },
    { 0 }
};