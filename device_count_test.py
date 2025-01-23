import cupy as cp
from mpi4py import MPI
import nvidia_smi

comm = MPI.COMM_WORLD
print(comm.rank)


cp.cuda.Device(comm.rank).use()
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(comm.rank)
x = cp.array([1, 2, 3]*(4-comm.rank)*10**8)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print(x.device)
print("Total memory:", info.total/2**20)
print("Free memory:", info.free/2**20)
print("Used memory:", info.used/2**20)

nvidia_smi.nvmlShutdown()
