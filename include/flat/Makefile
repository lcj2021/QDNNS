.PHONY: all test

all: test

test:
	nvcc -o test test.cpp knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets -Xcompiler -fopenmp 
clean:
	rm test
