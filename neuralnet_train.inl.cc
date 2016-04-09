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
//#define  _PRINT_DEBUG_TEXT

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


// Computes delta terms for previous layer in back propagation.
// Since we store weights as "incoming" must re-index since we want "outgoing".
//
// For each node, delta is computed as  F(SUM(outgoing_weights*next_layer_deltas))
// Where F is dSigmoid.
bool BackPropagateErrorInLayer(const vector<double> &deltas,
                               const vector<Node> &nodes,
                               vector<double> *output) {
    const int num_weights = nodes[0].weights.size();
    vector<double> error_term;
    error_term.resize(num_weights, 0.0);

    // For each weight, loop over each node nad accumulate.
    // Ie: Accumulate weight 0 for all nodes, then 1, then 2..
    for (int w = 0; w < num_weights; w++) {
        for (int n = 0; n < nodes.size(); n++) {
            const double delta = deltas[n];
            const auto &node = nodes[n];
            error_term[w] += node.weights[w] * delta;
        }
    }

    *output = error_term;
    return true;
}

// Does an in-place modification of data. Computes dSigmoid of each value.
bool ApplyDerivativeActivation(vector<double> *vec) {
    assert(!vec->empty());
    std::transform(vec->begin(), vec->end(), vec->begin(),
                   [](double x) { return dSigmoid(x); });
}

// new_theta = old_theta + learning_rate * delta
bool UpdateNodeWeights(const double delta, vector<double> *weights) {
  for (int i = 0; i < weights->size(); i++) {
    const double new_theta = (*weights)[i] + LEARNING_RATE * delta;
#ifdef _PRINT_DEBUG_TEXT
    printf("updating weight %i, w[%f] + LEARNING_RATE * delta[%f]=%f\n",
        i, (*weights)[i], delta, new_theta);
#endif
    (*weights)[i] = new_theta;
  }
  return true;
}

bool ComputeInitialDeltas(const vector<double> &labeled_data_inputs,
                          const vector<double> &labeled_data_outputs,
                          const vector<Layer> &layers,
                          vector<double> *output_deltas);
// Forward propagate training set and generate detlas (errors) from expected output..
bool ComputeInitialDeltas(const vector<double> &labeled_data_inputs,
                          const vector<double> &labeled_data_outputs,
                          const vector<Layer> &layers,
                          vector<double> *output_deltas) {
  vector<double> computed_values;
  if (!DoForwardPropagate(labeled_data_inputs, layers, &computed_values)) {
    return false;
  }
  assert(computed_values.size() == labeled_data_outputs.size());

  vector<double> deltas;
  // Compute difference from labeled (expected) output
  VectorDifference(labeled_data_outputs, computed_values, &deltas);

  // Compute mean of error.
  double sum = std::accumulate(deltas.begin(), deltas.end(), 0.0);
  double mean = sum / deltas.size();
  printf("Mean training error : %f\n", mean);

  ApplyDerivativeActivation(&deltas);
  *output_deltas = deltas;
  return true;
}

// Update weights of each neuron (reverse direction) by subtracting the delta
// term (computed error).
bool UpdateWeights(const vector< vector<double> > &deltas_mat,
                   vector<Layer> *layers) {
  assert(deltas_mat.size() == layers->size() - 1);
  // Note, that there is no first layer for deltas.
  for (int l = deltas_mat.size() - 1; l >= 0; l--) {
    const vector<double> &deltas = deltas_mat[l];
    auto *layer = &layers->at(l+1);
    if(deltas.size() != layer->nodes.size() ) {
      cout << "tvykruta: layer # " << l <<  " deltas " << deltas.size() << " nodes " << layer->nodes.size() << "\n";
    }

    assert(deltas.size() == layer->nodes.size());
    for (int n = 0; n < layer->nodes.size(); n++) {
      auto *node = &layer->nodes[n];
      if (!UpdateNodeWeights(deltas[n], &node->weights)) {
        return false;
      }
    }
  }
  return true;
}

bool BackPropagate(const vector<double> &labeled_data_inputs,
                   const vector<double> &labeled_data_outputs,
                   const vector<Layer> &layers,
                   vector< vector<double> > *output_deltas_mat) {
  vector<double> initial_deltas;
  if (!ComputeInitialDeltas(labeled_data_inputs, labeled_data_outputs,
                                  layers, &initial_deltas) ) return false;
  vector< vector<double> > new_deltas;
  vector<double> *deltas = &initial_deltas;
  new_deltas.push_back(initial_deltas);
  // Accumulate
  for (int i = layers.size() - 1; i > 1; i--) {
      const auto &layer = layers[i];
      vector<double> output_deltas;
      BackPropagateErrorInLayer(*deltas,
                                layer.nodes,
                                &output_deltas);
      ApplyDerivativeActivation(&output_deltas);
      new_deltas.push_back(output_deltas);
      deltas = &new_deltas.back();
  }
  std::reverse(new_deltas.begin(), new_deltas.end());
  *output_deltas_mat = new_deltas;
  return true;
}

bool NeuralNetwork::BackPropagate(const vector<double> &labeled_data_inputs,
                                  const vector<double> &labeled_data_outputs) {
  vector< vector<double> > output_deltas;
  bool ret = ::BackPropagate(labeled_data_inputs, labeled_data_outputs,
                             layers, &output_deltas);
  if (!ret) return ret;

  ret = UpdateWeights(output_deltas, &layers);
  if (!ret) return ret;

  return true;
}

