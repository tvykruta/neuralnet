LIBS=-lmgl -lGL -lpng

all: main

%: %.cc draw_util.o
	g++ -std=c++11 $< -o $@ draw_util.o $(LIBS)

draw_util.o: draw_util.cc
	g++ -std=c++11 -c draw_util.cc $(LIBS)
  
%: %.c
	gcc $< -o $@

clean:
	rm *.o main