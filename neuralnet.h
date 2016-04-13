// NEURAL NET IMPLEMENTATION C++
// (c) Tomas Vykruta
//
//           NeuralNetwork
//            **********
//            *        *
//            *        *
//            *        *
//            *        *
//            **********
//                 *
//                  **   Layer
//                ***********
//
// Flexible, fast Neural Net implementation.
// Architecture diagram:
//
// NeuralNetwork
//   -> Layer
//     -> Node
//        : Weights
//
// Notes:
// Uses standard forward propagation with a sigmoid function.
// Does not implicitly insert a bias node.
//

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <vector>

using namespace std;

class GlobalInitialize {
public:
  GlobalInitialize() {
    srand (time(NULL));
  }
};
GlobalInitialize g;

// Neuron inside a Neural Net.
struct Node {
  Node(const int num_weights) {
    weights.resize(num_weights);
    gradients.resize(num_weights, 0.0);
    // Seed random weight.
    const float RAND_EPSILON = 20.0f;
    for (int w = 0; w < num_weights; w++) {
        weights[w] = 2.0f * RAND_EPSILON * ((float)rand() / RAND_MAX) - RAND_EPSILON;
    }
  }
  // Incoming weights. Each weight corresponds to a node in previous layer.
  vector<double> weights;
  // Gradients used for back-propagation. Accumulated for each training set.
  vector<double> gradients;
};

// Layer inside the neural network composed of one or more nodes.
// Note the input nodes and output node(s) are layers.
struct Layer {
  Layer(const int num_nodes, const int num_weights) {
    Init(num_nodes, num_weights);
  }

  // Creates array of num_nodes Nodes,  each with num_weights weights.
  // Also resizes array of thetas, one per node.
  bool Init(const int num_nodes, const int num_weights) {
    for (int i = 0; i < num_nodes; i++) {
      nodes.emplace_back(num_weights);
    }
    return true;
  }

  vector<Node> nodes;  // Array of Nodes in this layer
};


// Primary implementation.
class NeuralNetwork {
public:
  NeuralNetwork() : last_mean_square_error(1.0), num_trained_samples(0) {};
  // Construct simple 3 layer neural network (most common).
  bool Create(const int input_nodes, const int hidden_nodes, const int output_nodes);
  // Construct full neural network with arbitrary # of hidden layers.
  bool Create(const vector<int> &nodes_per_layer);

  // For debugging only, seed weights. Must match node structure.
  bool LoadWeights(const vector<vector<double>> &weights);
  // For debugging only, seed weights. Must match node structure.
  bool LoadGradients(const vector<vector<double>> &gradients);
  // Run forward propagation and update output values.
  const bool ForwardPropagate(const vector<double> &input_values,
                        vector<double> *output_values) const;

  // For debugging, draws the network as ASCII.
  void PrintDebug() const {
    cout << "=== NEURAL NET WITH " << layers.size() << " LAYERS ===\n";
    // TODO: Add code to draw ascii graph
    for (int l = 0; l < layers.size(); l++) {
      printf("layer %i: ", l);
      const auto &layer = layers[l];
      for (int n = 0; n < layer.nodes.size(); n++) {
        const auto &node = layer.nodes[n];
        if (node.weights.size() >= 0) {
          cout << " [";
          for (int w = 0; w < node.weights.size(); w++) {
            printf("%0.3f ", node.weights[w]);
          }
          cout << "] ";
          }
      }
      cout << "\n";
    }
  }

public:
  // training
  bool BackPropagate(const vector<double> &labeled_data_inputs,
                     const vector<double> &labeled_data_outputs);
  // Update weights. Call after all training samples.
  bool UpdateWeights();

public:
  vector<Layer> layers;
  // For training only.
  int num_trained_samples;
  double last_mean_square_error;
};

// Forward propagation: Updates weights and transforms through sigmoid.
bool UpdateNode(const vector<double> &thetas,
                const vector<double> &weights,
                float *output_value);
// Updates all nodes in a layer.
bool UpdateLayer(const vector<double> &node_values_previous_layer,
                 const vector<Node> &nodes,
                 vector<double> *node_values_computed);
bool DoForwardPropagate(const vector<double> &input_values,
                        const vector<Layer> &layers,
                        vector< vector<double> > *activation_values = NULL);
#endif // NEURALNET_H_