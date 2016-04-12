// https://www.raspberrypi.org/forums/viewtopic.php?t=23077&p=216391
// c++ demo_generate_png.cc -o demo_generate_png -lmgl -lGL -lpng

#include <mgl2/mgl.h>
#include "draw_util.h"

#include <vector>

#include "includes/cutest.h"
#include "neuralnet_math.inl.cc"

using namespace std;

void test_GrapXForm() {
}

void setupGraph(mglGraph *gr, const float funcWidth) {
  gr->Title("Sigmoid function and derivative");
  //  gr.Clf(0.7, 0.7, 0.7);
  //  gr.SubPlot(1,1,0, "#");
  //gr.SetOrigin(0,-1);
  //gr.Box();
  gr->SetRanges(-funcWidth, funcWidth, 0.0f, 1.0f);
  gr->Axis();
  gr->Grid();
}

void test_DrawSigmoid() {
  // Test sigmoid function.
  printf("x, sigmoid, dSigmoid\n");

  const float funcWidth = 5.0f;
  const float funcStep = 0.25f;
  const int funcSteps = (funcWidth * 2.0f) / funcStep;
  const float sigmoidScale = 3.0f;
  printf("steps %i", funcSteps);

  // Plot sigmoid and derivative sigmoid function.
  mglData dat(funcSteps);
  mglData dat2(funcSteps);
  int index = 0;
  for (float x = -funcWidth; x < funcWidth; x += funcStep) {
    const double weight = -2.0;
    double s = sigmoid(weight * x, sigmoidScale);
    double ds = dSigmoid(s);
    printf("setting dat %i to %f for x=%f\n", index, s, x);
    dat[index] = s;
    dat2[index] = ds;
    index++;
  }

  mglGraph gr;
  setupGraph(&gr, funcWidth);
  gr.Plot(dat2, "1b");
  gr.Plot(dat, "1r");
  gr.WriteFrame("sigmoid.png");	// save it
}


TEST_LIST = {
    { "test_DrawSigmoid",  test_DrawSigmoid },
    { "test_GrapXForm", test_GrapXForm },
    { 0 }
};