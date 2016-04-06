// Linux graphics drawing
// see http://askubuntu.com/questions/525051/how-do-i-use-graphics-h-in-ubuntu
//  g++ draw.cc -o draw.o -lmgl-wnd

#include <mgl2/window.h>


int sample(mglGraph *gr)
{
  gr->Rotate(60,40);
  gr->Box();
  return 0;
}

//-----------------------------------------------------

int draw_stuff()
{
  mglWindow gr(sample,"MathGL examples");
  return gr.Run();
}
