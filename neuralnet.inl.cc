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

#include "neuralnet.h"
#include "neuralnet_math.inl.cc"

using namespace std;

bool NeuralNetwork::Create(const vector<int> &nodes_per_layer) {
  // Need at least 2 layers (input, output)
  if (nodes_per_layer.size() < 2) return false;

  int num_nodes_previous_layer = 0;
  for (int l : nodes_per_layer) {
    AddLayer(l, num_nodes_previous_layer);
    num_nodes_previous_layer = l;
  }
  return true;
}

// Simple 3 layer creation.
bool NeuralNetwork::Create(const int input_nodes, const int hidden_nodes, const int output_nodes) {
  vector<int> layers = {input_nodes, hidden_nodes, output_nodes};
  return Create(layers);
}

// Forward propagate the network and compute output values
bool NeuralNetwork::ForwardPropagate(const vector<double> &input_values,
                                     vector<double> *output_values) {
  if (input_values.size() != layers[0].nodes.size() ) {
    return false;
  }
  // Start with 1st hidden layer and compute values for all nodes
  // Continue through all layers to output nodes
  // Normalize output values through sigmoid function.
  vector<double> thetas = input_values;
  vector<double> *node_values_computed;
  const int nlayers = layers.size();
  for (int i = 1; i < nlayers; i++) {
    // thetas from previous later
    const auto &nodes = layers[i].nodes;
    node_values_computed = &layers[i].node_values_computed;
    
    if (!UpdateLayer(thetas, nodes, node_values_computed)) return false;
    thetas = *node_values_computed;
  }
  *output_values = *node_values_computed;
  return true;
}

// Initialize a neural network with weights for each node.
// Ie: For network with 2 input nodes, 2 nodes hidden layeer ,1 output,
// we have: [ [n0.w0, n0.w1, [n1.w0, n1.w1], [n2.w1, n2.w1] 
bool NeuralNetwork::LoadWeights(const vector<vector<double>> &weights) {
  int node_global_count = 0;
  for (int l = 1; l < layers.size(); l++) {
    auto *layer = &layers[l];
    for (int n = 0; n < layer->nodes.size(); n++) {
      auto *node = &layer->nodes[n];
      if (node_global_count >= weights.size()) {
        std::cout << "Weights mismatch count.";
        return false;
      }
      if (node->weights.size() != weights[node_global_count].size()) {
        std::cout << "Weights mismatch count.";
        return false;
      }
      node->weights = weights[node_global_count++];
    }
  }
  if (node_global_count < weights.size()) {
    std::cout << "LoadWeights mismatch count, loaded " << node_global_count
              << " weights expected " << weights.size();
    return false;
  }
  return true;
}

// Generates value for Node. Applies activation function to previous layer's node
// thetas and incoming weights.
bool UpdateNode(const vector<double> &thetas,
                const vector<double> &weights, 
                float *output_value) {
  if (thetas.size() != weights.size()) {
    cout << "Error, thetas weights mismatch.";
    return false;
  }
  float val = ComputeNode(weights, thetas);
  val = sigmoid(val);
  *output_value = val;
  return true;
}
  
// Updates each node in this layer. Generates a new node_values_computed arra.
// note: Thetas are from previous layer.
bool UpdateLayer(const vector<double> &thetas,
                 const vector<Node> &nodes,
                 vector<double> *node_values_computed) {
  for (int i = 0; i < nodes.size(); i++) {
    const auto &node = nodes[i];
    float value = -9999999.0;
    if (!UpdateNode(thetas, node.weights, &value)) {
      return false;
    }
    
    node_values_computed->at(i) = (double)value;
  }
  return true;
}

