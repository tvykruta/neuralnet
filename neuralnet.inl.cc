// Neural Net C++ implementation by Tomas Vykruta (2016).
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

#include <iostream>
#include <vector>
#include <cassert>

#include "neuralnet.h"
#include "neuralnet_math.inl.cc"

using namespace std;

// Generates value for Node. Applies activation function to previous layer's node
// thetas and incoming weights.
bool UpdateNode(const vector<double> &thetas,
                const vector<double> &weights,
                double *output_value) {
  assert(thetas.size() == weights.size());
  double val = ComputeNode(weights, thetas);
  val = sigmoid(val);
  *output_value = val;
  return true;
}

// Updates each node in this layer. Generates a new node_values_computed array.
// note: Thetas are from previous layer.
bool UpdateLayer(const vector<double> &node_values_previous_layer,
                 const vector<Node> &nodes,
                 vector<double> *node_values_computed) {
  for (int i = 0; i < nodes.size(); i++) {
    const auto &node = nodes[i];
    double value = -999.0;
    if (!UpdateNode(node_values_previous_layer, node.weights, &value)) {
      return false;
    }

    node_values_computed->push_back((double)value);
  }
  return true;
}

// Forward propagate the network and compute output values.
// Variables:
// let x = input, T = thetas (weights), z = SUM(Ta), a = activation (sigmoid(z))
// Algorithm as per Andrew Ng's course see "Gradient Computation" slide:
// https://www.coursera.org/learn/machine-learning/lecture/1z9WW/backpropagation-algorithm
// and wiki: https://share.coursera.org/wiki/index.php/ML:Neural_Networks:_Learning
//
// For a 4 layer network
// a(1) = x
// z(2) = T(1)a(1)
// a(2) = g( z(2) )
// z(3) = T(2)a(2)
// a(3) = g( z(3) )
// z(4) = T(3)a(3)
// a(4) = hT(x) = g( z(4) )
//
// Textual description
// Let activation a of first layer simply be the inputs
// Let layer 2 computed node values z be sum of (thetas * inputs)
// let layer 2 activations be the sigmoid of the z values
// let layer 3 computed node values again be SUM(T*a)
// let layer 4 activation be the sigmoid of the z values
// let the hypotenus be simply be activation of layer 4
bool DoForwardPropagate(const vector<double> &input_values,
                        const vector<Layer> &layers,
                        vector< vector<double> > *activation_values) {
  assert(activation_values->empty());
  assert(input_values.size() == layers[0].nodes.size() - 1);

  // Start with 1st hidden layer and compute values for all nodes
  // Continue through all layers to output nodes
  // Normalize output values through sigmoid function.
  //
  // node_values = values from previous layer
  // nodes = nodes in current layer we';re computing for

  // Insert bias node value.
  vector<double> node_values = { 1.0 };
  node_values.insert(node_values.end(), input_values.begin(), input_values.end());
  activation_values->push_back(node_values);

  vector<double> computed_values;
  for (int i = 1; i < layers.size(); i++) {
    // node_values from previous later
    const auto &nodes = layers[i].nodes;
    if (!UpdateLayer(node_values, nodes, &computed_values)) {
      return false;
    }
    if (i < layers.size() - 1) {
      // Set bias node actiavtion to 1.0 in all but output layer.
      computed_values[0] = 1.0;
    }
    activation_values->push_back(computed_values);
    node_values = computed_values;
    computed_values.clear();
  }
  return true;
}

// Adds layer with N nodes. Each node has array of X weights for incoming
// nodes.
//
//   A  A  A
//     B  B
// Above example if we add layer B, each node B would have 3 weights.
bool AddLayer(const int num_nodes, const int num_nodes_previous_layer,
              vector<Layer> *layers) {
    layers->emplace_back(num_nodes, num_nodes_previous_layer);
}


bool NeuralNetwork::Create(const vector<int> &nodes_per_layer) {
  // Need at least 2 layers (input, output)
  if (nodes_per_layer.size() < 2) return false;

  int node_count_previous_layer = 0;
  for (int l = 0; l < nodes_per_layer.size(); l++) {
    int node_count = nodes_per_layer[l];
    if (node_count < 1 ) return false;
    if (l < nodes_per_layer.size() - 1) {
      node_count++; // bias unit
    }
    AddLayer(node_count, node_count_previous_layer, &layers);
    node_count_previous_layer = node_count;
  }
  return true;
}

// Simple 3 layer creation.
bool NeuralNetwork::Create(const int input_nodes, const int hidden_nodes, const int output_nodes) {
  vector<int> layers = {input_nodes, hidden_nodes, output_nodes};
  return Create(layers);
}

// Initialize a neural network with weights for each node.
// Ie: For network with 2 input nodes, 2 nodes hidden layeer ,1 output,
// we have: [ [n0.w0, n0.w1, [n1.w0, n1.w1], [n2.w1, n2.w1]
// NOTE: Must include weight for BIAS node. BIAS node is node 0.
bool NeuralNetwork::LoadWeights(const vector<vector<double>> &weights) {
  int node_global_count = 0;
  for (int l = 1; l < layers.size(); l++) {
    auto *layer = &layers[l];
    // Skip loading weights for bias node.
    int start_index = 1;
    if (l == layers.size() - 1) {
      start_index = 0;
    }
    for (int n = start_index; n < layer->nodes.size(); n++) {
      auto *node = &layer->nodes[n];
      if (node_global_count >= weights.size()) {
        printf("LoadWeights insufficient weights. Expected %i got %lu.\n",
            node_global_count, weights.size());
        return false;
      }
      if (node->weights.size() != weights[node_global_count].size()) {
        printf("LoadWeights count mismatch. Layer %i node %i expected %lu got %lu  // 0 XNOR 1 = 0",
            l, n, node->weights.size(), weights[node_global_count].size());
        return false;
      }
      node->weights = weights[node_global_count++];
    }
  }
  if (node_global_count != weights.size()) {
    printf("Too many weights expected %i got %lu\n", node_global_count - 1, weights.size());
    return false;
  }
  return true;
}


// Forward propagate the network and compute output values
const bool NeuralNetwork::ForwardPropagate(const vector<double> &input_values,
                                           vector<double> *output_values) const {
  if (layers.size() < 2) {
    cout << "Invalid network, needs 2 or more layers.\n";
    return false;
  }
  if (input_values.size() != layers[0].nodes.size() - 1) {
    printf("Inputs mismatch, got %lu but network expects %lu input nodes\n",
        input_values.size(), layers[0].nodes.size() - 1);
    return false;
  }
  if (!output_values->empty()) {
    output_values->clear();
  }
  vector< vector<double> > activation_values;
  if (!DoForwardPropagate(input_values, layers, &activation_values)) {
    return false;
  }
  assert(activation_values.size() == layers.size());
  *output_values = activation_values.back();
  return true;
}

