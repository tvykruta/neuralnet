// https://www.raspberrypi.org/forums/viewtopic.php?t=23077&p=216391
// c++ demo_generate_png.cc -o demo_generate_png -lmgl -lGL -lpng

#include <mgl2/mgl.h>
#include "draw_util.h"

int draw_stuff() {
    mglGraph gr;
    gr.FPlot("sin(pi*x)+abs");
    gr.WriteFrame("test.png");
}
