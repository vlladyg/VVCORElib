import cupy as cp
import h5py
from datetime import datetime
from mpi4py import MPI
import nvidia_smi

from trajectory_reader import trajectory_cp

comm = MPI.COMM_WORLD
num_of_devices = cp.cuda.runtime.getDeviceCount()
if comm.rank == 0:
    print(num_of_devices)
############### Cur calculation #####################
opts = ['cur']

path = "/pscratch/sd/v/vladygin/VVCORE_benchmark/data/hdf5/"
traj_file = f"{path}/NaBr_T300.h5"

N = 10000
traj = trajectory_cp(traj_file, vel_flag = not (len(opts) == 1 and opts[0] == 'dens'))


cp.cuda.Device(comm.rank).use()
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(comm.rank)

if comm.rank == 0:
    start_time = datetime.now()
tot = 1
for i in range(tot):
    pos, vel = traj.get_slice(0, N//num_of_devices//tot)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    #if comm.rank == 0:
    #    print(pos.device)
    #    print("Total memory:", info.total/2**20)
    #    print("Free memory:", info.free/2**20)
    #    print("Used memory:", info.used/2**20)

if comm.rank == 0:
    end_time = datetime.now()
    print(f"Total time of reading is {end_time - start_time}")
nvidia_smi.nvmlShutdown()
