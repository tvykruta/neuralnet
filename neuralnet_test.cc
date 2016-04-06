// Simple unit testing for NeuralNet class.
// We use the "CuTest" unit test library.
#include <iostream>
#include <memory>
#include <vector>

#include "includes/cutest.h"
#include "neuralnet.inl.cc"
#include "test_utils.inl.cc"

using namespace std;

#define FLT_SMALL 0.0001f
#define FLT_MIN -10000000000f

void test_create()
{
  int layers = 3;
  std::unique_ptr<NeuralNetwork> n(new NeuralNetwork);
  TEST_CHECK(n != NULL);
  TEST_CHECK(n->Create(1,2,1));

  // Neural net with 2 input nodes, 3 hidden layer nodes, 1 output node.
  vector<int> node_counts = {2, 3, 3};
  std::unique_ptr<NeuralNetwork> n2(new NeuralNetwork);
  TEST_CHECK(n2->Create(node_counts));
}

void test_UpdateNode() {
  
  vector<double> thetas = {0.2f, 0.4f, 0.6f};
  vector<double> weights = {2.0f, 1.5f, 0.5f};
  float output_value = -9999.0f;
  
  TEST_CHECK(UpdateNode(thetas, weights, &output_value));
  TEST_CHECK_(output_value - 1.3 < FLT_SMALL,
              "Expected %f got %f",
              output_value,
              1.3f);
}

void test_NeuralNet_2x1() {
  // Initialize a neural 2 input, 1 output, no hidden layer.
  vector<double> output_values;
  NeuralNetwork nn;
  const vector<int> nodes_per_layer = {3, 1};
  TEST_CHECK(nn.Create(nodes_per_layer));

  // logical OR.
  {
    const vector<vector<double>> weights_init = { { 0.0f, 1.0f, 1.0f } };
    TEST_CHECK(nn.LoadWeights(weights_init));
    
    // First value is always 1, bias value.
    // 1 | 0 == 1
    const vector<double> input_values = { 1, 1, 0 };
    TEST_CHECK(nn.ForwardPropagate(input_values, &output_values));
    TEST_CHECK_(1 == VectorToBinaryClass(output_values),
                "Got %s", PrintVector(output_values).c_str());
  }
  
  // logical AND.
  {
    const vector<vector<double>> weights_init = { { -2.0f, 1.0f, 1.0f } };
    TEST_CHECK(nn.LoadWeights(weights_init));
    
    // First value is always 1, bias value.
    // 1 & 0 == 0
    vector<double> input_values = { 1, 1, 0 };
    TEST_CHECK(nn.ForwardPropagate(input_values, &output_values));
    TEST_CHECK_(0== VectorToBinaryClass(output_values),
                "Got %s", PrintVector(output_values).c_str());

    // 1 & 1 == 1
    input_values = { 1, 1, 1 };
    TEST_CHECK(nn.ForwardPropagate(input_values, &output_values));
    TEST_CHECK((1 & 1) == VectorToBinaryClass(output_values));
  }
  
  // logical XOR.
  {
    const vector<vector<double>> weights_init = { { -2.0f, 1.0f, 1.0f } };
    TEST_CHECK(nn.LoadWeights(weights_init));
    
    // First value is always 1, bias value.
    // 1 & 0 == 0
    vector<double> input_values = { 1, 1, 0 };
    TEST_CHECK(nn.ForwardPropagate(input_values, &output_values));
    TEST_CHECK_(0== VectorToBinaryClass(output_values),
                "Got %s", PrintVector(output_values).c_str());

    // 1 & 1 == 1
    input_values = { 1, 1, 1 };
    TEST_CHECK(nn.ForwardPropagate(input_values, &output_values));
    TEST_CHECK((1 & 1) == VectorToBinaryClass(output_values));
  }
  
}


void test_NeuralNet_2x2x1() {
  // Initialize a neural network with weights for each node.
  // Ie: For network with 2 input nodes, 2 nodes hidden layeer ,1 output,
  // we have: [ [n0.w0, n0.w1, [n1.w0, n1.w1], [n2.w1, n2.w1] 
  // n1.w0, n1.w1, n2.w0, n2.w1
  vector<vector<double>> neural_net_weights = 
      { { 1.0, 1.0 }, {1.0, 1.0 }, 
        { 1.0, 1.0 } };
  NeuralNetwork nn;
  TEST_CHECK(nn.Create(2, 2, 1));
  TEST_CHECK(nn.LoadWeights(neural_net_weights));

  // logical OR
  const vector<double> input_values = { 1, 0 };
  vector<double> output_values;
  vector<double> expected_values = { 0.774004 };
  TEST_CHECK(nn.ForwardPropagate(input_values, &output_values));
  TEST_CHECK(fabs(VectorToDouble(output_values) - 0.774004f) < 0.001f);
  TEST_CHECK_(std::equal(std::begin(output_values), std::end(output_values), std::begin(expected_values)),
              "Expected %s got %s",
              PrintVector(expected_values).c_str(),
              PrintVector(output_values).c_str());
}

TEST_LIST = {
    { "create",  test_create },
    { "UpdateNode",    test_UpdateNode },
    { "NeuralNet_2x1", test_NeuralNet_2x1 },
    { "NeuralNet_2x2x1", test_NeuralNet_2x2x1 },
    { 0 }
};