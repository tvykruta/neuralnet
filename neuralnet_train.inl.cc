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

constexpr float LEARNING_RATE = 0.1F;
constexpr float MOMENTUM = 0.1f;
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
// output: new deltas for Layer L - 1
//
// d(l) = SUM(d(l+1) * theta(l)) * dSigmoid(a(l))
bool BackPropagateErrorInLayer(const vector<double> &deltas,
                               const vector<Node> &nodes,
                               const vector<double> &activations,
                               vector<double> *output_deltas) {
    assert(output_deltas->empty());
    const int num_weights = nodes[0].weights.size();
    assert(num_weights > 0);
    vector<double> output;
    output.resize(num_weights, 0.0);

    // For each weight, loop over each node nad accumulate.
    // Ie: Accumulate weight 0 for all nodes, then 1, then 2..
    for (int w = 0; w < num_weights; w++) {
        for (int n = 0; n < nodes.size(); n++) {
            const double delta = deltas[n];
            const auto &node = nodes[n];
            output[w] += node.weights[w] * delta;
        }
    }

    for (int w = 0; w < num_weights; w++) {
      output[w] *= dSigmoid(activations[w]);
    }
    *output_deltas = output;
    return true;
}

// Does an in-place modification of data. Computes dSigmoid of each value.
bool ApplyDerivativeActivation(vector<double> *vec) {
    assert(!vec->empty());
    std::transform(vec->begin(), vec->end(), vec->begin(),
                   [](double x) { return dSigmoid(x); });
}

// new_theta = old_theta + learning_rate * average_gradient
bool UpdateNodeWeights(const double gradient, vector<double> *weights) {
  for (int i = 0; i < weights->size(); i++) {
    const double new_theta = (*weights)[i] - LEARNING_RATE * gradient;
#ifdef _PRINT_DEBUG_TEXT
    printf("updating weight %i, w[%f] + LEARNING_RATE * gradient[%f]=%f\n",
        i, (*weights)[i], gradient, new_theta);
#endif
    (*weights)[i] = new_theta;
  }
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

// Update weights of each neuron (reverse direction) by subtracting the delta
// term (computed error).
bool UpdateWeights(const vector< vector<double> > &gradients_mat,
    const int num_trained_samples, vector<Layer> *layers) {
  assert(gradients_mat.size() == layers->size());
  // Note, that there is no first layer for gradients.
  for (int l = gradients_mat.size() - 1; l >= 1; l--) {
    const vector<double> &gradients = gradients_mat[l];
    auto *layer = &layers->at(l);
    if(gradients.size() != layer->nodes.size() ) {
      cout << "tvykruta: layer # " << l <<  " gradients " << gradients.size() << " nodes " << layer->nodes.size() << "\n";
    }

    assert(gradients.size() == layer->nodes.size());
    double d1_trainingsamples = 1.0 / (double)num_trained_samples;
    for (int n = 0; n < layer->nodes.size(); n++) {
      auto *node = &layer->nodes[n];
      if (!UpdateNodeWeights(gradients[n] * d1_trainingsamples, &node->weights)) {
        return false;
      }
    }
  }
  return true;
}



// g(l) += d(l+1) * a(l)
bool AccumulateGradient(const vector<double> &deltas,
                        const vector<double> &activations,
                        vector<double> *gradients) {
  assert(deltas.size() == activations.size());
  // gradient = deltas * activations
  vector<double> temp (4);
  std::transform(deltas.begin(), deltas.end(), activations.begin(),
                 temp.begin(), std::multiplies<double>());
  // gradients += gradient
  std::transform(gradients->begin(), gradients->end(), temp.begin(),
                 gradients->begin(), std::plus<double>());
}

// Computes gradients and accuulates into gradients matrix.
// g(l) += d(l+1) * a(l)
bool AccumulateGradients(const vector< vector<double> > &activations,
                         const vector< vector<double> > &deltas,
                         vector< vector<double> > *gradients) {
  assert(activations.size() == gradients->size());
  assert(activations.size() == deltas.size());

  for (int l = 1; l < activations.size(); l++) {
   //printf("l %i gradients %lu activations %lu  \n", l, gradients->at(l).size(), activations[l].size());
    assert(gradients->at(l).size() == activations[l].size());
    //printf("l %i deltas %lu activations %lu  \n", l, deltas[l].size(), activations[l].size());
    assert( deltas[l].size() == activations[l].size());
    vector<double> *gradient = &(*gradients)[l];
    AccumulateGradient(deltas[l], activations[l], gradient);
  }
  return true;
}

// Run back propagation algorihtm. Accumulates into gradients. Call repeatedly
// for each training sample.
bool BackPropagate(const vector<double> &labeled_data_inputs,
                   const vector<double> &labeled_data_outputs,
                   const vector<Layer> &layers,
                   vector< vector<double> > *gradients) {
  // First forward propagate, generate 'a' terms.
  vector< vector<double> > activations;
  if (!DoForwardPropagate(labeled_data_inputs, layers, &activations)) {
    return false;
  }
  assert(activations.back().size() == labeled_data_outputs.size());
  assert(activations.size() == layers.size()); // First layer has no activations.

  // Generate deltas of output layer.
  vector< vector<double> > deltas;
  deltas.resize(1);
  deltas.back().resize(labeled_data_outputs.size());
  VectorDifference(activations.back(), labeled_data_outputs, &deltas.back());
  assert(deltas.back().size() == activations.back().size());

  // Back propgate deltas to each hidden. Do not compute for first layer.
  for (int i = layers.size() - 1; i > 1; i--) {
      const auto &layer = layers[i];
      vector<double> output_deltas;
      output_deltas.reserve(layer.nodes[0].weights.size());
      BackPropagateErrorInLayer(deltas.back(),
                                layer.nodes,
                                activations[i],
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
  bool ret = ::BackPropagate(labeled_data_inputs, labeled_data_outputs,
                             layers, &gradients);

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

void ZeroMatrix(vector< vector<double> > *mat) {
  for (auto &row : *mat) {
    std::fill(row.begin(), row.end(), 0.0);
  }
}

// Update weights, call after all training samples have been run.
bool NeuralNetwork::UpdateWeights() {
  assert(num_trained_samples > 0);
  if (!::UpdateWeights(gradients, num_trained_samples, &layers)) {
    return false;
  }

  num_trained_samples = 0;
  ZeroMatrix(&gradients);
  return true;
  // TODO: Do another forwadr propagation and compute new MSE.
}