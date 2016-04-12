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

  float e = (float)ComputeMeanSquaredError(a, b);
  TEST_CHECK_(e > 2.0 && e < 2.3, "%0.19f expected", e);
}

void test_VectorSubtract() {
  vector<double> a = { 5.0, 3.0 };
  vector<double> b = { 1.0, 1.0 };
  vector<double> result;
  VectorDifference(a, b, &result);
  vector<double> expected = { 4.0, 2.0 };
  TEST_CHECK(result == expected);
}

void testInitialDeltas() {
  /*&
  NeuralNetwork nn;
  TEST_CHECK(nn.Create( { 2, 1 } ));
  // Set up weights simple 2 layer network.
  const vector<double> test_data_inputs = { 1.0, 2.0 };
  const vector<vector<double>> weights = { { 0.5, 0.1 } };
  TEST_CHECK(nn.LoadWeights(weights));
  // Compute expected result:
  // 0.5 * 1.0 + 0.1 * 2.0 = 0.7
  // a = sigmoid(0.7)
  // expected_deltas = (0.9 - a)
  const vector<double> test_data_outputs = { 0.9 };
  const vector<double> expected_deltas = { 0.9 - sigmoid(0.7) };
  vector<double> output;
  TEST_CHECK(ComputeInitialDeltas(test_data_inputs, test_data_outputs,
                                  nn.layers, &output));
  TEST_CHECK_(std::equal(std::begin(output), std::end(output),
              std::begin(expected_deltas)),
              "Output: %0.9f expected %0.9f diff %0.9f",
              output[0], expected_deltas[0], output[0] - expected_deltas[0]);
  */
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
  const vector<double> activations = { 0.5, 0.5, 0.5 };
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
  vector<double> output_deltas;
  TEST_CHECK(BackPropagateErrorInLayer(deltas,
      nn.layers[1].nodes,
      activations,
      &output_deltas));
  const vector<double> expected_delta = { 2.0, 2.125, 2.25 };
  TEST_CHECK_(std::equal(std::begin(output_deltas), std::end(output_deltas),
              std::begin(expected_delta)),
              "Expected %s got %s",
              PrintVector(expected_delta).c_str(),
              PrintVector(output_deltas).c_str());
}

void test_UpdateWeights() {
  NeuralNetwork nn;
  TEST_CHECK(nn.Create( { 2, 2, 1 } ));
  const vector<vector<double>> init_weights = { { 2.0, 2.1 }, { 3.0, 3.1 },
                                                  { 1.0, 1.1 } };
  TEST_CHECK(nn.LoadWeights(init_weights));

  // Learning rate is 0.1.
  vector< vector<double> > gradients = { { },{ 20.0, 30.0 },
                                            { 10.0 } };

  vector< vector<double> > expected  = {{ },  { 4.0, 4.1 }, { 6.0, 6.1 },
                                               { 2.0, 2.1 } };

  TEST_CHECK(UpdateWeights(gradients, 2, &nn.layers));

  int node_counter = 0;
  for (int l = 1; l <  nn.layers.size(); l++) {
    const auto& layer = nn.layers[l];
    for (int n = 0; n < layer.nodes.size(); n++) {
      const auto& weights = layer.nodes[n].weights;

      assert(node_counter < expected.size());
      const vector<double> &expected_node = expected[node_counter];
      TEST_CHECK_(VecSimilar(expected_node, weights),
                  "Expected %s got %s diff %0.9f",
                  PrintVector(expected_node).c_str(),
                  PrintVector(weights).c_str(),
                  expected_node[0] - weights[0]);
      node_counter++;
    }
  }
}

void test_AccumulateGradients() {
  vector< vector<double> > activations = { { 0.0, 0.0 },
                                           { 1.1, 1.2 },
                                           { 2.0 } };
  vector< vector<double> > deltas = { { },
                                      { 2.0, 2.1 },
                                      { 3.0 }};
  vector< vector<double> > gradients = { { },
                                         { 0.1, 0.2 },
                                         { 0.3 } };
  vector< vector<double> > expected { { },
                                      { 2.3, 2.72 },
                                      { 6.3 } };

  // Computes: g(l) += d(l+1) * a(l)
  // let l = 0
  // g(0) = {3.0 * 1.1}, { 3.0 * 1.2 }
  // g(0) = {3.3, 3.6}
  TEST_CHECK(AccumulateGradients(activations, deltas, &gradients));
  for (int i = 0; i < gradients.size(); i++) {
    auto &gradient = gradients[i];
    auto &expectedval = expected[i];
    TEST_CHECK_(std::equal(std::begin(gradient), std::end(gradient),
                std::begin(expectedval)),
                "Got %s expected %s",
                PrintVector(gradient).c_str(),
                PrintVector(expectedval).c_str());
  }
}

void test_BackPropagateFull() {
  NeuralNetwork nn;
  // 3 input nodes, 1 output node.
  TEST_CHECK(nn.Create( { 3, 1 } ));
  // Set up weights connecting the layers together, 3 weights per output node.
  const vector<vector<double>> weights = { { 1.0, 1.1, 1.2 } };
  TEST_CHECK(nn.LoadWeights(weights));

  const vector<double> training_inputs = { 1, 0, 1 };
  const vector<double> training_outputs = { 1 };

  TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));
}

void RunTest(const vector<double> &input_values, const int expected, const NeuralNetwork& nn) {
  vector<double> output_values;
  // Now test the network
  TEST_CHECK(nn.ForwardPropagate(input_values, &output_values));
  printf("Ran test: Input %s Expected %i got %s (precisely %f)\n",
         PrintVector(input_values).c_str(),
         expected,
         PrintVector(output_values).c_str(),
         VectorToDouble(output_values));
  TEST_CHECK_(expected == VectorToBinaryClass(output_values),
              "Input %s Expected %i got %s (precisely %f)",
              PrintVector(input_values).c_str(),
              expected,
              PrintVector(output_values).c_str(),
              VectorToDouble(output_values));
}

// logical AND truth table
// 1 & 1 = 1
// 1 & 0 = 0
// 1 & 1 = 1
// 0 & 0 = 0
void test_TrainAND() {
  NeuralNetwork nn;
  // Create neira; met wotj 3 input nodes, 1 output nodes.
  TEST_CHECK(nn.Create( { 3, 1 } ));
  //const vector<vector<double>> weights_init = { { 20.0, 20.00, -30.0 } };
  //TEST_CHECK(nn.LoadWeights(weights_init));
  nn.PrintDebug();

  vector<double> training_inputs;
  vector<double> training_outputs;

  for (int i = 0; i < 1000; i++) {
    training_inputs = { 1, 1, 1 };
    training_outputs = { 1 };
    TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));

    training_inputs = { 1, 0, 1 };
    training_outputs = { 0 };
    TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));

    training_inputs = { 0, 1, 1 };
    training_outputs = { 0 };
    TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));

    training_inputs = { 0, 0, 1 };
    training_outputs = { 0 };
    TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));
    TEST_CHECK(nn.UpdateWeights());
  }

  nn.PrintDebug();
  // Now test the network
  // logical AND truth table
  // 1 & 1 = 1
  // 1 & 0 = 0
  // 1 & 1 = 1
  // 0 & 0 = 0
  vector<double> input_values;
  input_values = { 1, 1, 1 };
  RunTest(input_values, 1, nn);

  input_values = { 1, 0, 1 };
  RunTest(input_values, 0, nn);

  input_values = { 0, 1, 1 };
  RunTest(input_values, 0, nn);

  input_values = { 0, 0, 1 };
  RunTest(input_values, 0, nn);
}


// logical XOR
// 1 ^ 1 = 0
// 1 ^ 0 = 1
// 0 ^ 1 = 1
// 0 ^ 0 = 0
void test_TrainXOR() {
  NeuralNetwork nn;
  // Create neira; met wotj 3 input nodes, 1 output nodes.
  TEST_CHECK(nn.Create( { 3, 3, 1 } ));
  nn.PrintDebug();

  for (int i = 0; i < 1000; i++) {
    {
      const vector<double> training_inputs = { 1, 1, 1 };
      const vector<double> training_outputs = { 0 };
      TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));
    }
    {
      const vector<double> training_inputs = { 1, 0, 1 };
      const vector<double> training_outputs = { 1 };
      TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));
    }
    {
      const vector<double> training_inputs = { 0, 1, 1 };
      const vector<double> training_outputs = { 1 };
      TEST_CHECK(nn.BackPropagate(training_inputs, training_outputs));
    }
    {
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
      { "test_Helpers",  test_Helpers },
      { "test_VectorSubtract",  test_VectorSubtract },
      { "testInitialDeltas", testInitialDeltas },
      { "test_BackPropagateError", test_BackPropagateError },
      { "test_UpdateWeights", test_UpdateWeights },
      { "test_AccumulateGradients", test_AccumulateGradients },
      { "test_BackPropagateFull", test_BackPropagateFull },
      //{ "test_TrainAND", test_TrainAND },
      { "test_TrainXOR", test_TrainXOR },
    { 0 }
};