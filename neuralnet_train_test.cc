// Simple unit testing for NeuralNet class.
// We use the "CuTest" unit test library.
#include <iostream>
#include <memory>
#include <vector>

#include "includes/cutest.h"
#include "neuralnet_train.inl.cc"
#include "test_utils.inl.cc"

using namespace std;

#define FLT_SMALL 0.0001f
#define FLT_MIN -10000000000f

void test_Helpers() {
  vector<double> a = { 1.0, 2.0 };
  vector<double> b = { 0.1, 0.2 };
  vector<double> result;
  VectorDifference(a, b, &result);
  TEST_CHECK(result[0] == 0.9 && result[1] == 1.8);

  vector<double> outputs = { 0.5, 0.51 };
  ApplyDerivativeActivation(&outputs);
  TEST_CHECK(outputs[0] == dSigmoid(0.5));
  TEST_CHECK(outputs[1] == dSigmoid(0.51));
}

void test_VectorSubtract() {
  vector<double> a = { 5.0, 3.0 };
  vector<double> b = { 1.0, 1.0 };
  vector<double> result;
  VectorDifference(a, b, &result);
  vector<double> expected = { 4.0, 2.0 };
  TEST_CHECK(result == expected);
}


// Back propagate a single layer
void test_BackPropagateError() {
  NeuralNetwork nn;
  // Create neira; met wotj 3 input nodes, 2 output nodes.
  TEST_CHECK(nn.Create( { 3, 2 } ));
  // Set up weights connecting the layers together, 3 weights per output node.
  const vector<vector<double>> weights = { { 1.0, 1.1, 1.2 }, { 2.0, 2.1, 2.2 } };
  TEST_CHECK(nn.LoadWeights(weights));
  // Simulate back propagation from a 2 node layer to a 3 node layer.
  vector<double> deltas  = { 2.0, 3.0 };

  /*
  We created the network below. The operations move from bottom to top.
  // N0 = (delta.0*weight.0 + delta.1 * weight.3)

  expected:            8.0  8.5 9.0
  nodes:      N0   N1  N2         N0   N1   N2
  d*w:        2.0, 2.2, 2.4       6.0, 6.3, 6.6
  weights:    1.0, 1.1, 1.2       2.0, 2.1, 2.2
  delta:          (2.0)                 (3.0)
  */
  vector<double> output;
  BackPropagateErrorInLayer(deltas,
                            nn.layers[1].nodes,
                            &output);
  const vector<double> expected_delta = { 8.0, 8.5, 9.0 };
  TEST_CHECK_(std::equal(std::begin(output), std::end(output),
              std::begin(expected_delta)),
              "Expected %s got %s",
              PrintVector(expected_delta).c_str(),
              PrintVector(output).c_str());
}

void test_UpdateWeights() {
  NeuralNetwork nn;
  TEST_CHECK(nn.Create( { 2, 2, 1 } ));
  const vector<vector<double>> init_weights = { { 2.0, 2.1 }, { 3.0, 3.1 },
                                                  { 1.0, 1.1 } };
  TEST_CHECK(nn.LoadWeights(init_weights));

  // Learning rate is 0.1.
  vector< vector<double> > deltas =      { { 20.0, 30.0 },
                                             { 10.0 } };

  vector< vector<double> > expected  =   { { 4.0, 4.1 }, { 6.0, 6.1 },
                                                  { 2.0, 2.1 } };

  TEST_CHECK(UpdateWeights(deltas, &nn.layers));

  int node_counter = 0;
  for (int l = 1; l <  nn.layers.size(); l++) {
    const auto& layer = nn.layers[l];
    for (int n = 0; n < layer.nodes.size(); n++) {
      const auto& weights = layer.nodes[n].weights;

      const vector<double> &expected_node = expected[node_counter];
      TEST_CHECK_(std::equal(std::begin(weights), std::end(weights),
                  std::begin(expected_node)),
                  "Expected %s got %s",
                  PrintVector(expected_node).c_str(),
                  PrintVector(weights).c_str());
      node_counter++;
    }
  }
}

void test_BackPropagateFull() {
  NeuralNetwork nn;
  // Create neira; met wotj 3 input nodes, 1 output nodes.
  TEST_CHECK(nn.Create( { 3, 1 } ));
  // Set up weights connecting the layers together, 3 weights per output node.
  const vector<vector<double>> weights = { { 1.0, 1.1, 1.2 } };
  TEST_CHECK(nn.LoadWeights(weights));

  const vector<double> training_inputs = { 1, 0, 1 };
  const vector<double> training_outputs = { 1 };

  for (int i = 0; i < 100; i++) {
    TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));
  }
}

void RunTest(const vector<double> &input_values, const int expected, const NeuralNetwork& nn) {
  vector<double> output_values;
  // Now test the network
  TEST_CHECK(nn.ForwardPropagate(input_values, &output_values));
  TEST_CHECK_(expected == VectorToBinaryClass(output_values),
              "Input %s Expected %i got %s (precisely %f)",
              PrintVector(input_values).c_str(),
              expected,
              PrintVector(output_values).c_str(),
              VectorToDouble(output_values));
}


// logical OR
// 1 ^ 1 = 0
// 1 ^ 0 = 1
// 0 ^ 1 = 1
// 0 ^ 0 = 0
void test_TrainOR() {
  NeuralNetwork nn;
  // Create neira; met wotj 3 input nodes, 1 output nodes.
  TEST_CHECK(nn.Create( { 3, 1 } ));

  for (int i = 0; i < 100; i++) {
    {
      // third term is BIAS
      const vector<double> training_inputs = { 1, 1, 1 };
      const vector<double> training_outputs = { 0 };
      TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));
    }
    {
      // third term is BIAS
      const vector<double> training_inputs = { 1, 0, 1 };
      const vector<double> training_outputs = { 1 };
      TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));
    }
    {
      // third term is BIAS
      const vector<double> training_inputs = { 0, 1, 1 };
      const vector<double> training_outputs = { 1 };
      TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));
    }
    {
      // third term is BIAS
      const vector<double> training_inputs = { 0, 0, 1 };
      const vector<double> training_outputs = { 0 };
      TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));
    }
  }

  nn.PrintDebug();
  // Now test the network
  // 1 ^ 1 = 0
  // 1 ^ 0 = 1
  // 0 ^ 1 = 1
  // 0 ^ 0 = 0
  vector<double> input_values = { 1, 1, 1 };
  RunTest(input_values, 0, nn);

  input_values = { 1, 0, 1 };
  RunTest(input_values, 1, nn);

  input_values = { 0, 1, 1 };
  RunTest(input_values, 1, nn);

  input_values = { 0, 0, 1 };
  RunTest(input_values, 0, nn);
}


TEST_LIST = {
    { "test_VectorSubtract",  test_VectorSubtract },
    { "test_BackPropagateError", test_BackPropagateError },
    { "test_BackPropagateFull", test_BackPropagateFull },
    { "test_Helpers",  test_Helpers },
    { "test_UpdateWeights", test_UpdateWeights },
    { "test_TrainOR", test_TrainOR },
    { 0 }
};