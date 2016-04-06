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

// Neuron inside a Neural Net.
struct Node {
  Node(const int num_weights) {
    weights.resize(num_weights);
  }
  // Incoming weights. Each weight corresponds to a node in previous layer.
  vector<double> weights;
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
    node_thetas.resize(num_nodes);
    node_values_computed.resize(num_nodes);
    return true;
  }

  vector<Node> nodes;  // Array of Nodes in this layer 
  vector<double> node_thetas;  // Array of thetas corresponding to nodes above.
  vector<double> node_values_computed;  // Computed values in forward propagation.
};

// Primary implementation.
class NeuralNetwork {
public:
  // Construct full neural network from # of nodes per layer.
  bool Create(const vector<int> &layer_node_count);
  // Simple 3 layer creation.
  bool Create(const int input_nodes, const int hidden_nodes, const int output_nodes);
  // For debugging only, seed weights.
  bool LoadWeights(const vector<vector<double>> &weights);
  // Run forward propagation and update output values.
  bool ForwardPropagate(const vector<double> &input_values,
                        vector<double> *output_values);
  int NumLayers() const {
    return layers.size();
  }
private:
  // Adds layer with N nodes. Each node has array of X weights for incoming
  // nodes.
  // 
  //   A  A  A
  //     B  B
  // Above example if we add layer B, each node B would have 3 weights.
  bool AddLayer(const int num_nodes, const int num_nodes_previous_layer) {
    layers.emplace_back(num_nodes, num_nodes_previous_layer);
  }
  
public:
  vector<Layer> layers;
};

// Forward propagation: Updates weights and transforms through sigmoid.
bool UpdateNode(const vector<double> &thetas,
                const vector<double> &weights, 
                float *output_value);
// Updates all nodes in a layer.
bool UpdateLayer(const vector<double> &thetas,
                 const vector<Node> &nodes,
                 vector<double> *node_values_computed);

#endif // NEURALNET_H_