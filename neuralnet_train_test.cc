// Simple unit testing for NeuralNet class.
// We use the "CuTest" unit test library.
#include <iostream>
#include <memory>
#include <vector>

#include "includes/cutest.h"
#include "neuralnet_train.inl.cc"
#include "test_utils.inl.cc"

using namespace std;

#define FLT_SMALL 0.0001f
#define FLT_MIN -10000000000f

void test_Create() {

}

void test_VectorSubtract() {
  vector<double> a = { 5.0, 3.0 };
  vector<double> b = { 1.0, 1.0 };
  vector<double> result;
  VectorDifference(a, b, &result);
  vector<double> expected = { 4.0, 2.0 };
  TEST_CHECK(result == expected);
}

TEST_LIST = {
    { "test_VectorSubtract",  test_VectorSubtract },
    { "Create",  test_Create },
    { 0 }
};