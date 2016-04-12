#ifndef _TEST_UTILS_INL_CC
#define _TEST_UTILS_INL_CC
// Simple unit testing for NeuralNet class.
// We use the "CuTest" unit test library.
#include <algorithm>
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
// Returns TRUE if vectors are almost same. { SUM(vec_1 - vec_2) < threshold }
bool VecSimilar(const vector<double> &vec_1, const vector<double> &vec_2);
const string PrintVector(const vector<double> &v) {
  string s = "{";
  string separator = "";
  for (auto i = v.begin(); i != v.end(); ++i) {
    //s += separator + to_string(*i);
    char buff[256];
    snprintf(buff, sizeof(buff), "%0.9f", *i);
    s += separator + buff;
    separator = ",";
  }
  s += "}";
  return s;
}

// Helper sum vector and converts to 0 or 1 where < 0.5 = {0} and > 0.5 = {1}.
int VectorToBinaryClass(const vector<double> &vec) {
  double sum = std::accumulate(vec.begin(), vec.end(), 0.0f);
  assert(sum >= 0.0 && sum <= 1.0);
  return (sum >= 0.5);
}

// Helper sum vector and converts to 0 or 1 where < 0.5 = {0} and > 0.5 = {1}.
double VectorToDouble(const vector<double> &vec) {
  double sum = std::accumulate(vec.begin(), vec.end(), 0.0f);
  return sum;
}

bool VecSimilar(const vector<double> &vec_1,
                      const vector<double> &vec_2) {
  if (vec_1.size() != vec_2.size()) {
    return false;
  }
  const double THRESHOLD = 0.000f;
  vector<double> r;
  std::transform(vec_1.begin(), vec_1.end(), vec_2.begin(),
	    std::back_inserter(r),
	    std::minus<double>());
	return std::accumulate(r.begin(), r.end(), 0.0f) < THRESHOLD;
}

#endif  // _TEST_UTILS_INL_CC