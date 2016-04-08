// Neural back-propagation training.
// See header file for documentation.
//
//                      . .
//                    '.-:-.`
//                    '  :  `
//                 .-----:
//               .'       `.
//         ,    /       (o) \
//     jgs \`._/          ,__)
//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>

#include "neuralnet.inl.cc"
//#include "neuralnet_math.inl.cc"

using namespace std;

// Simple backpropagation training algorithm.
// Supervised learning strategy that requires labeled training data.
//
// 1. Forward-propagate sample data, and compute difference (error) against
// expected output.
// 2. Backwards-propagate the error through each node.
// 3. Compute difference and modify weights of each node to reduce error.

constexpr float LEARNING_RATE = 0.1F;
constexpr float MOMENTUM = 0.1f;

void VectorDifference(const vector<double> &vec_1,
                      const vector<double> &vec_2,
                      vector<double> *result);

// Compute vec_1 - vec_2
void VectorDifference(const vector<double> &vec_1,
                      const vector<double> &vec_2,
                      vector<double> *result) {
  std::transform(vec_1.begin(), vec_1.end(), vec_2.begin(),
	    std::back_inserter(*result),
	    std::minus<double>());
}

/*
// Back propagation uses the derivative of the sigmoid function:
// Sigmoid and derivative of sigmoid
// f = 1/(1+exp(-x))
// df = f * (1 - f)
double dSigmoid(const double val);
double dSigmoid(const double val) {
  return val * (1 - val);
}*/



  // See https://www.coursera.org/learn/machine-learning/lecture/1z9WW/backpropagation-algorithm
  // Back propagation is forward propagation with these differences:
  // 1. We start from the output nodes and move left
  // 2. The "input" is the difference (error) between the forward propagation
  // and expected result, called the "delta term".
  // TO get the "delta term" we take "sigmoid derivative". It's the slope
  // of the activation function (since we want to know which direction to
  // move the theta to arrive at a smaller error).
  //
  // For example with a 2x2x1 network, we do this:
  // 1. Forward-propagate 2 inputs to get 1 output
  // 2. Take derivative sigmoid of output, we call this Delta term
  //    this is the error of output node. THe goal is to approach 0.
  // 3. Propagate the delta term to each node just like fw prop but
  //    backwards, and using dSigmoid instead of Sigmoid for activation
  //    After this phase you'll have a delta term for each node.
  //    for each node compute:
  //    node_error = SUM(connection_theta*layer_next.delta)
  //    delta = dSigmoid(node_error)
  // 4. For each neuron, generate a new bias term using the computed
  //    delta and weights (thetas) of connections.
  //    new_bias = old_bias + learning_rate * delta
  //    ** note: learning_rate is constant (like 0.2)
  //    ** note, we can leave bias as 1.0 and skip to step 5.
  // 5. Update weights (thetas) to "correct" for error (delta).
  //    new_theta = old_theta + learning_rate * delta
  // Repeat until mean_error and error approaches 0.
  //
  // δ₂ = (Ω) ...

  /*
    For backprop we can use same weights vector. i tink??
    Let Nln b a matrix of Neurons where l = layer and n = neuron

    delta_term for layer 1, node 0:
    d = layer_2_deltas * node.weights
    vec_errors = SUM_MUL(deltas_layer2, )
    */

// For each node, does a SUM(delta * node.weight[weightINdex])
double SumMulWeight(const int weightIndex,
                    const double delta,
                    const vector<Node> &nodes) {
    double val = 0.0f;
    for (const auto &node : nodes) {
        val += node.weights[weightIndex];
    }
    return val;
}

// Computes delta terms for previous layer in back propagation.
// Since we store weights as "incoming" must re-index since we want "outgoing".
//
// For each node, delta is computed as  F(SUM(outgoing_weights*next_layer_deltas))
// Where F is dSigmoid.
bool BackPropagateErrorInLayer(const vector<double> &delta_next_layer,
                               const vector<Node> &nodes_next_layer,
                               vector<double> *output_error_term) {
    double sum_deltas = 0.0;
    const int num_weights = nodes_next_layer[0].weights.size();
    vector<double> error_term;
    error_term.resize(num_weights, 0.0);

    // Loop over each output node and accumulate error.
    for (int w = 0; w < num_weights; w++) {
        const double delta = delta_next_layer[w];
        for (int d = 0; d < delta_next_layer.size(); d++) {
            error_term[d] += SumMulWeight(w, delta, nodes_next_layer);
        }
    }

    *output_error_term = error_term;
    return true;
}

bool func(double a) { return 1; }
bool ApplyDerivativeActivation(vector<double> *vec) {
    assert(!vec->empty());
/*    // Activation dSigmoid term
    for (auto *value : *vec) {
      *value = dSigmoid(*value);
    }
*/
    std::transform(vec->begin(), vec->end(), vec->begin(),
                   [](double x) { return dSigmoid(x); });
}

// new_theta = old_theta + learning_rate * delta
bool UpdateNodeWeights(const double delta, vector<double> *weights) {
  for (int i = 0; i < weights->size(); i++) {
    const double new_theta = (*weights)[i] + LEARNING_RATE * delta;
    (*weights)[i] = new_theta;
  }
  return true;
}

// Forward propagate training set and generate detlas (errors) from expected output..
bool ComputeInitialDeltas(const vector<double> &labeled_data_inputs,
                          const vector<double> &labeled_data_outputs,
                          const vector<Layer> &layers,
                          vector<double> *output_deltas) {
  vector<double> computed_values;
  DoForwardPropagate(labeled_data_inputs, layers, &computed_values);
  vector<double> deltas;

  // Compute difference from labeled (expected) output
  VectorDifference(computed_values, labeled_data_outputs, &deltas);
  ApplyDerivativeActivation(&deltas);
  *output_deltas = deltas;
}

bool NeuralNetwork::BackPropagate(const vector<double> &labeled_data_inputs,
                                  const vector<double> &labeled_data_outputs) {

  vector<double> initial_deltas;
  bool f = ComputeInitialDeltas(labeled_data_inputs, labeled_data_outputs, layers, &initial_deltas);
  assert(f);
  vector< vector<double> > output_layers;
  vector<double> *deltas = &initial_deltas;
  // Accumulate
  for (int i = layers.size() - 1; i > 0; i--) {
      const auto &layer = layers[i];
      vector<double> output_deltas;
      BackPropagateErrorInLayer(*deltas,
                                layer.nodes,
                                &output_deltas);
      output_layers.push_back(output_deltas);
      deltas = &output_layers.back();
  }
  std::reverse(output_layers.begin(), output_layers.end());

  // Update weights of each neuron (reverse direction)
  for (int l = output_layers.size(); l > 1; l++) {
    const vector<double> &layer_deltas = output_layers[l];
    for (int i = 0; i < layers[l].nodes.size(); i++) {
      auto *node = &layers[l].nodes[i];
      const double delta = layer_deltas[l];
      UpdateNodeWeights(delta, &node->weights);
    }
  }

}
