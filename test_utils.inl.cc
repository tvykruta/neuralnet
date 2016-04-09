// Simple unit testing for NeuralNet class.
// We use the "CuTest" unit test library.
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

const string PrintVector(const vector<double> &v);
int VectorToBinaryClass(const vector<double> &vec);
double VectorToDouble(const vector<double> &vec);

const string PrintVector(const vector<double> &v) {
  string s = "{";
  string separator = "";
  for (auto i = v.begin(); i != v.end(); ++i) {
    s += separator + to_string(*i);
    separator = ",";
  }
  s += "}";
  return s;
}

// Helper sum vector and converts to 0 or 1 where < 0.5 = {0} and > 0.5 = {1}.
int VectorToBinaryClass(const vector<double> &vec) {
  double sum = std::accumulate(vec.begin(), vec.end(), 0.0f);
  return (int) (sum + 0.5f);
}

// Helper sum vector and converts to 0 or 1 where < 0.5 = {0} and > 0.5 = {1}.
double VectorToDouble(const vector<double> &vec) {
  double sum = std::accumulate(vec.begin(), vec.end(), 0.0f);
  return sum;
}
