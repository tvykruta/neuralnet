
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

#include "neuralnet.inl.cc"
#include "draw_util.h"

using namespace std;

#define TEST_DATA_FILENAME "training_data.txt"
#define NN_LAYERS 3
const int NN_NODES[NN_LAYERS] = { 2, 3, 1 };

// Loads a file.
void loadFile(string name) {
  string line;
  ifstream myfile (name);
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      cout << line << '\n';
    }
    myfile.close();
  }
  else cout << "Unable to open file";     
}

// Gets input from keyboard.
void getInput(int n) {
    char key = 0;
    cin >> key;
    cout << "You pressed " << key;
}

// Main function.
main() {
    loadFile(TEST_DATA_FILENAME);
    
    NeuralNetwork n;
    n.Create(2,3,1);
    std::cout << "Neural Network created." << std::endl;
}
