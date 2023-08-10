import numpy as np
import os
import psutil
import time
from .utils import get_types_ind, get_egv, gen_output_dict, get_num_iter, signal_mem, get_start_end
from .qgrids import grids
from .trajectory_reader import trajectory_h5

import ctypes
import distutils.sysconfig

np_pointer = np.ctypeslib.ndpointer

pointer_vel = np_pointer(dtype=np.float64, ndim=2,
                           flags='f_contiguous, aligned')
pointer_egv = np_pointer(dtype=np.complex128, ndim=2,
                           flags='f_contiguous, aligned')

pointer_k = np_pointer(dtype=np.complex128, ndim=2,
                            flags='f_contiguous, aligned')
pointer_v_k = np_pointer(dtype=np.complex128, ndim=2,
                            flags='f_contiguous, aligned, writeable')


c_ext = ctypes.cdll.LoadLibrary(f"{os.path.dirname(__file__)}/../_rho_j_k_d{distutils.sysconfig.get_config_var('EXT_SUFFIX')}")

c_ext.v_k.argtypes = (pointer_vel, ctypes.c_int,
                                pointer_k, ctypes.c_int, 
                      pointer_v_k)

def frame_to_signal(vel, egv):
    np.require(vel, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    np.require(egv, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])

    res = {'v_k': None}
    Natoms = vel.shape[1]
    Nq = egv.shape[1]
    v_k = np.zeros((3,Nq), dtype = np.complex128, order = 'F')
    c_ext.v_k(vel, Natoms, egv, Nq, v_k)
    res['v_k'] = v_k.T

    return res

def compute_signal(traj_file, N, opts, comm):
    """Initializes computations for signal"""
    np.random.seed(comm.rank)
    
    egv = get_egv()
            
    Nq = egv['1'].shape[1]
    frame_ind = get_start_end(N, comm.size, comm.rank)
    ind = get_types_ind()
    traj = trajectory_h5(traj_file, ind, vel_flag = True)
    mem = psutil.virtual_memory().available/comm.size/2**20
    Nsplit, Qsplit = signal_mem(mem, max([ind[key].size for key in ind]), Nq, opts)
    
    if Qsplit < 1:
        print("WARNING: Your setup takes too much memory on a single process, job might cancel due to out of memory error")
        Qsplit = 1
    
    return signal(traj, egv, frame_ind, Nsplit, Qsplit, opts)


def signal(traj, egv, frame_ind, Nsplit, Qsplit, opts):
    """Computes signal splitted according to memory requirenemnts"""
    Nq = egv['1'].shape[1]
    res = gen_output_dict(['1'], frame_ind.size, Nq, opts)
    Q_ind = np.arange(Nq)
    
    Niter, Qiter = get_num_iter(frame_ind.size, Nsplit), get_num_iter(Nq, Qsplit)
    Niter = frame_ind.size; Nsplit = 1
    read_time = 0
    compute_time = 0
    for i in range(frame_ind[0], frame_ind[-1]+1):
        
        start_time_read = time.time()
        pos, vel = traj.read_one_frame(i)
        
        read_time += time.time() - start_time_read
        
        start_time_compute = time.time()
        for j in range(Qiter):
            cur_ind_q = Q_ind[j*Qsplit:(j+1)*Qsplit]
            for k in traj.ind:
                res_tmp = frame_to_signal(vel[traj.ind[k]].transpose(), egv[k][:, cur_ind_q[0]:cur_ind_q[-1]+1])
                res['v_k']['1'][i-frame_ind[0], cur_ind_q[0]:cur_ind_q[-1]+1] += res_tmp['v_k']
        compute_time += time.time() - start_time_compute
        
    traj.close()
    return res, read_time, compute_time
