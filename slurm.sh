#!/bin/bash -x

module load xl_r spectrum-mpi cuda/11.2

mpirun --bind-to core --map-by node -np 2 ./io eatingbaby.png output.png