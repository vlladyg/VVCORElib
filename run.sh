#!/bin/bash

module load PrgEnv-nvidia cray-mpich cudatoolkit craype-accel-nvidia80 python
conda activate gpu-aware-mpi

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1
export MPICH_GPU_SUPPORT_ENABLED=1 


srun -n 128 -G 4 --cpu-bind=cores --gpu-bind=none python VVCORE.py 10000
