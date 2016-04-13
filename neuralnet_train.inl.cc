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
#include "test_utils.inl.cc"
//#include "neuralnet_math.inl.cc"

using namespace std;

// Simple backpropagation training algorithm.
// Supervised learning strategy that requires labeled training data.
//
// 1. Forward-propagate sample data, and compute difference (error) against
// expected output.
// 2. Backwards-propagate the error through each node.
// 3. Compute deltas and adjust weights of each node to reduce errors.

constexpr float LEARNING_RATE = 0.2F;
//#define  _PRINT_DEBUG_TEXT


// Compute vec_1 - vec_2
void VectorDifference(const vector<double> &vec_1,
                      const vector<double> &vec_2,
                      vector<double> *result) {
  result->resize(vec_1.size());
  assert(vec_1.size() == vec_2.size());
  for (int i = 0; i < vec_1.size(); i++) {
    (*result)[i] = vec_1[i] - vec_2[i];
  }
}

  // See https://www.coursera.org/learn/machine-learning/lecture/1z9WW/backpropagation-algorithm
  // http://www.codeproject.com/Articles/14342/Designing-And-Implementing-A-Neural-Network-Librar
  // BEST ALGORITHM DESCRPTION: http://outlace.com/Beginner-Tutorial-Backpropagation/
  //
  // Definitions for algorithm
  // x = input, a = node activation (sigmoid(z)), theta = weights, z = theta*a
  // d = delta or error, D = accumulated delta, F = sigmoid func
  // # subscript = layer, where 1 = first layer, 3 = 3rd layer (Output)
  // z = computed node value before activation function
  // First I wil show algorithm, the describe each setp
  //
  // LOOP ALL TRAINING SAMPLES
  // 1. a(2) = F(SUM(theta(1)*x)), then a(l+1) = F(SUM(theta(l)a(l))
  // 2. d3 = a3 - training_set_expected
  // 3. d2 = SUM(d3 * theta_2) * dSigmoid(a2))
  // 4. g(l) = d(l+1) * a(l)
  // 5. D(l) += g(l)
  // END LOOP
  // 6. theta1 += D2 * learning_rate
  // *8. special: d3 = e3
  //
  // LOOP
  // 1. Forward-propagate training set input x to generate activation (a) terms for
  // all layers.
  // 2. Generate error term e3 for each node in output layer as difference between
  // forward propgate output a3 and expected output from training set.
  // 3. Back-propagate error e3 to previous layer just like forward propagation
  // by multiplying it by weights (in reverse) and taking SUM, Take derivative sgmoid
  // of the resultant SUM.. Do this for all layers. Now you have vectors of error
  // (delta) terms e2 and e3 (we don't compute e1).
  // 4. Generate gradients g2 for each hidden layer node by multiplying
  // original activation value a2 by the error term.
  // Why does this work? We're taking the derivative (slope) of the Cpst
  // function. Recall Cost is the difference between activation and expected result.
  // 5. Accumulate the gradient terms in each node. Repeat for
  // all training samples.
  // END LOOP
  // 6. Finally "correct" weights of each node by subtracting accumulated gradients.
  //
  // 7. Special - the delta term for the output layer is computed directly
  // because we have the expected output from training data. In hidden layers
  // it's computed differently based on acviation value.
  //
  // Repeat until mean_error and error approaches 0.

/* intuitions

  // in a balanced, trained network, say you upset one of the weights buy
  // adjusting it. it propagates this error to all nodes in output
  layer. how do you rebelance the network?

  for many training samples, we must apply the weights as a sum of Detlas
  neuron.weight += SUM(deltas)

  and propagating the error requires thought too. the error we get is the
  error = desired_output - computed_output;

  neurons with higher weights will "received" more of the error via back
  propagation. a single "low weight" neuron will block almost all the
  weight from propagating.

  the derivative of sigmoid intuitively is the slope, which is always positive
  and which becomes 0 if < 0 and > 1. so a neuron is "trained" when it reaches
  0 or 1 on the error.
*/

// Back propagate layer L to L - 1;
// deltas: layer L computed deltas
// nodes: Incoming weights from layer L - 1
// activations: activations from L
// output_deltas: new deltas for Layer L - 1
// node_start_index: starting node. use 1 to skip bias propagating bias node.
//
// d(l) = SUM(d(l+1) * theta(l)) * dSigmoid(a(l))
bool BackPropagateDelta(const vector<double> &deltas,
                        const vector<Node> &nodes,
                        const vector<double> &activations,
                        const int node_start_index,
                        vector<double> *output_deltas) {
    assert(output_deltas->empty());
    const int num_weights = nodes[0].weights.size();
    assert(num_weights > 0);
    vector<double> output_delta;
    output_delta.resize(num_weights, 0.0);

    // For each theta, sum up delta * theta, then multiply by activation.
    // For each weight, loop over each node nad accumulate.
    // Ie: Accumulate weight 0 for all nodes, then 1, then 2..
    for (int w = 0; w < num_weights; w++) {
        for (int n = node_start_index; n < nodes.size(); n++) {
            const double delta = deltas[n];
            const auto &node = nodes[n];
            output_delta[w] += node.weights[w] * delta;
        }
    }

    for (int w = 0; w < num_weights; w++) {
      output_delta[w] *= dSigmoid(activations[w]);
    }
    *output_deltas = output_delta;
    return true;
}

// Mean squared error: ABS((VEC_1 - VEC_2))^2
double ComputeMeanSquaredError(const vector<double> &vec1,
                               const vector<double> &vec2) {
  assert(vec1.size() == vec2.size());
  // Compute mean square error, note, this does not take ABSOLUTE value!
  vector<double> diff;
  double sum = 0.0;
  for (int i = 0; i < vec1.size(); i++) {
    double abs_diff = (fabs(vec1[i] - vec2[i]));
    sum += abs_diff * abs_diff;
  }
  double mean = sum / (double)vec1.size();
  return mean;
}

// g(l) += d(l+1) * a(l)
// deltas = layer L
// activations = layer L-1
// gradients = for thetas from L-1 to L
bool AccumulateGradient(const double delta,
                        const vector<double> &activations,
                        vector<double> *gradients) {
  // Loop over incoming weights.
  assert(activations.size() == gradients->size());
  for (int w = 0; w < gradients->size(); w++) {
    double gradient = delta * activations[w];
    gradients->at(w) += gradient;
  }
  return true;
}

// Computes gradients and accumulates into gradients matrix.
// For each theta, we accumulate the next layer node's delta term.
// layers contains node,each node contains incoming gradients.
// g(l) += d(l+1) * a(l)
bool AccumulateGradients(const vector< vector<double> > &activations,
                         const vector< vector<double> > &deltas,
                         vector<Layer> *layers) {
  for (int l = layers->size() - 1; l >= 1; l--) {
    int start_index = 1;
    if (l == layers->size() - 1) {
      start_index = 0; // skip bias nodes, except in output layer.
      // IDEA: add an bias node in output too, then this code goes away.
    }
    Layer &layer = layers->at(l);
    assert(layer.nodes.size() == deltas[l].size());
    for (int n = start_index; n < layer.nodes.size(); n++) {
      Node &node = layer.nodes[n];
      if (activations[l-1].size() != node.gradients.size())
      printf("layer %i node %i activations[l-1].size() %lu node.gradients.size %lu\n",
         l, n, activations[l-1].size(), node.gradients.size());
      assert(activations[l-1].size() == node.gradients.size());
      assert(activations[l-1][0] == 1.0); // bias node is always 1.0.
      AccumulateGradient(deltas[l][n], activations[l-1], &node.gradients);
    }
  }
  return true;
}

// Run back propagation algorihtm. Accumulates into gradients. Call repeatedly
// for each training sample.
bool BackPropagate(const vector<double> &labeled_data_inputs,
                   const vector<double> &labeled_data_outputs,
                   const vector<Layer> &layers,
                   vector<Layer> *gradients) {
  // First forward propagate, generate 'a' terms.
  vector< vector<double> > activations;
  if (!DoForwardPropagate(labeled_data_inputs, layers, &activations)) {
    return false;
  }
  assert(activations.back().size() == labeled_data_outputs.size());
  assert(activations.size() == layers.size());

  // Generate deltas of output layer.
  vector< vector<double> > deltas;
  deltas.resize(1);
  deltas.back().resize(labeled_data_outputs.size());
  VectorDifference(activations.back(), labeled_data_outputs, &deltas.back());
  assert(deltas.back().size() == activations.back().size());

  // Back propagate deltas to each hidden. Do not compute for first layer.
  for (int i = layers.size() - 1; i > 1; i--) {
      int skip_bias_node = 1;
      if (i == layers.size() - 1) {
        skip_bias_node = 0;
      }
      const auto &layer = layers[i];
      vector<double> output_deltas;
      output_deltas.reserve(layer.nodes[0].weights.size());
      BackPropagateDelta(deltas.back(),
                         layer.nodes,
                         activations[i],
                         skip_bias_node,
                         &output_deltas);
      deltas.push_back(output_deltas);
  }
  // Insert empty placeholder to make deltas and layers array same size.
  deltas.push_back(vector<double>());
  // Reverse deltas to match layers.
  std::reverse(deltas.begin(), deltas.end());

  if (!AccumulateGradients(activations, deltas, gradients)) {
    return false;
  }
  return true;
}

bool NeuralNetwork::BackPropagate(const vector<double> &labeled_data_inputs,
                                  const vector<double> &labeled_data_outputs) {
  if (labeled_data_inputs.size() != layers[0].nodes.size() - 1) {
    return false;
  }
  bool ret = ::BackPropagate(labeled_data_inputs, labeled_data_outputs,
                             layers, &layers);

  // Compute mean square error.
  vector< vector<double> > activations;
  if (!::DoForwardPropagate(labeled_data_inputs, layers, &activations)) {
    return false;
  }
  last_mean_square_error = ComputeMeanSquaredError(labeled_data_outputs,
      activations.back());
  printf("Training mean squareerror for %s : %f\n",
      PrintVector(labeled_data_inputs).c_str(), last_mean_square_error);
  num_trained_samples++;
  return ret;
}

// Update thetas of each node by adding bias*gradient.
bool UpdateWeights(const int num_trained_samples, vector<Layer> *layers) {
  double d1_trainingsamples = 1.0 / (double)num_trained_samples;
  // Note, that there is no first layer for gradients.
  for (int l = 1; l < layers->size(); l++) {
    auto &layer = layers->at(l);
    for (int n = 0; n < layer.nodes.size(); n++) {
      auto &node = layer.nodes[n];
      for (int w = 0; w < node.weights.size(); w++) {
        assert(node.gradients[w] != 0.0);
        node.weights[w] += (-1.0 / (double)num_trained_samples) * node.gradients[w] * LEARNING_RATE;
        node.gradients[w] = 0.0;
      }
    }
  }
  return true;
}


// Update weights, call after all training samples have been run.
bool NeuralNetwork::UpdateWeights() {
  assert(num_trained_samples > 0);
  if (!::UpdateWeights(num_trained_samples, &layers)) {
    return false;
  }

  num_trained_samples = 0;
  return true;
}