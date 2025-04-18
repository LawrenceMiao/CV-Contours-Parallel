module load xl_r spectrum-mpi cuda/11.2
nvcc -O3 -arch=sm_70 -c io.cu -o io.o
mpicxx io.o -o io -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ -lpng