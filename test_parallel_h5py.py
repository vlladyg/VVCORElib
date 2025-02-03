from mpi4py import MPI
import h5py

rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for a 4-process job)

with h5py.File('test.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
  dset = f.create_dataset('test', (4,), dtype='i')
  dset[rank] = rank
