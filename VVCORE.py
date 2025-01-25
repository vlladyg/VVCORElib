import time

import cupy as cp
import numpy as np
import nvidia_smi

from signals import compute_signal
from utils import stack, dict_from_device, save_signal
import sys

if __name__ == "__main__":
    
    start_time_program = time.time()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    nvidia_smi.nvmlInit()

    num_of_devices = nvidia_smi.nvmlDeviceGetCount()
    cp.cuda.Device(comm.rank%num_of_devices).use()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(comm.rank%num_of_devices)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    ############### Cur calculation #####################
    opts = ['cur', 'cur_T', 'cur_L']
    
    
    path = "./data"
    traj_file = f"{path}/NaBr_T300K.h5"
    
    N = int(sys.argv[1])
    
    # Initializing k points grid
    
    lattice = 'fcc'
    a = 6.125
    Nq = 30
    
    time_init = time.time() - start_time_program
    if comm.rank == 0:
        print(time_init)
    start_time_signal = time.time()
    #################  Compute signal part  #############################################
    res_cp, read_time, compute_time = compute_signal(traj_file, N, Nq, lattice, a, opts, comm, num_of_devices, handle, info)
    
    time_signal = time.time() - start_time_signal
    start_time_comm_1 = time.time()
    
    # Getting data from device
    res = dict_from_device(res_cp)
    cp.cuda.get_current_stream().synchronize()
    
    ################### Saving current ############################### 
    for opt in opts:
        res_cur = comm.gather(res[opt], root = 0)
    
        if comm.rank == 0:
            res_cur = stack(res_cur)
            save_signal(res_cur, f"{path}/{opt}.h5")
            del res_cur
        res.pop(opt)
    
    
    comm.Barrier()
    time_comm_1 = time.time() - start_time_comm_1
    
    time_program = time.time() - start_time_program
    
    time_proc = np.array([time_comm_1, time_signal, time_program, read_time, compute_time], dtype='d')
    if comm.rank == 0:
        time_total = np.zeros_like(time_proc)
    else:
        time_total = None

    nvidia_smi.nvmlShutdown()
    
    comm.Reduce([time_proc, MPI.DOUBLE], [time_total, MPI.DOUBLE], op=MPI.SUM, root=0)
    
    if comm.rank == 0:
        time_total /= comm.size
        print(f"Total time SIGNAL = {time_total[1]}")
        print(f"\nTotal time READ = {time_total[-2]}")
        print(f"Total time COMPUTE = {time_total[-1]}\n")
        print(f"Total time PROGRAM = {time_total[2]}")
        print(f"Total time COMM = {time_total[0]}")

