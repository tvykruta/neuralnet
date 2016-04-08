LIBS=-lmgl -lGL -lpng

# File lists, keep updated.
TEST_FILES := neuralnet_test neuralnet_math_test neuralnet_train_test draw_util_test
SOURCE_FILES := main draw_util

# run: make test
# Builds and executes all tests listed in the file list below.
test: $(TEST_FILES)
	for FILE in $(TEST_FILES) ; do \
		./$$FILE ; \
	done

# run: make all
all: main
%: %.cc draw_util.o
	g++ -std=c++11 $< -o $@ $(LIBS)

draw_util.o: draw_util.cc
	g++ -std=c++11 -c draw_util.cc $(LIBS)

%: %.c
	gcc $< -o $@

clean:
	rm *.o main