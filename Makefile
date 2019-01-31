.PHONY: all clean

CUC=clang++
CU_FLAGS= -xcuda --cuda-path=/Developer/NVIDIA/CUDA-9.2 --cuda-gpu-arch=sm_30

all: lib/liblwps.dylib

clean:
	rm lib/liblwps.dylib obj/{kron,blas,vector,matrix}.o

lib/liblwps.dylib: obj/kron.o obj/blas.o obj/vector.o obj/matrix.o
	@mkdir -p lib
	ar cr lib/liblwps.dylib obj/kron.o obj/blas.o obj/vector.o obj/matrix.o

obj/kron.o: src/kron.cu
	@mkdir -p obj
	$(CUC) $(CU_FLAGS) -std=c++17 -Iinclude -I/Users/atkassen/projects/thrust -O3 -o obj/kron.o -c src/kron.cu

obj/blas.o: src/blas.cu
	@mkdir -p obj
	$(CUC) $(CU_FLAGS) -std=c++17 -Iinclude -I/Users/atkassen/projects/thrust -O3 -o obj/blas.o -c src/blas.cu

obj/vector.o: src/vector.cu
	@mkdir -p obj
	$(CUC) $(CU_FLAGS) -std=c++17 -Iinclude -I/Users/atkassen/projects/thrust -O3 -o obj/vector.o -c src/vector.cu

obj/matrix.o: src/matrix.cu
	@mkdir -p obj
	$(CUC) $(CU_FLAGS) -std=c++17 -Iinclude -I/Users/atkassen/projects/thrust -O3 -o obj/matrix.o -c src/matrix.cu
