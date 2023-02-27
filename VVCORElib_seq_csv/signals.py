import numpy as np
import os
import psutil
import time

from .utils import get_types_ind, gen_output_dict, get_num_iter, signal_mem
from .qgrids import grids
from .trajectory_reader import trajectory_csv

import ctypes
import distutils.sysconfig

np_pointer = np.ctypeslib.ndpointer

pointer_pos = np_pointer(dtype=np.float64, ndim=2,
                           flags='f_contiguous, aligned')
pointer_vel = np_pointer(dtype=np.float64, ndim=2,
                           flags='f_contiguous, aligned')
pointer_k = np_pointer(dtype=np.float64, ndim=2,
                           flags='f_contiguous, aligned')

pointer_dens = np_pointer(dtype=np.complex128, ndim=1,
                            flags='f_contiguous, aligned, writeable')
pointer_cur = np_pointer(dtype=np.complex128, ndim=2,
                            flags='f_contiguous, aligned, writeable')


c_ext = ctypes.cdll.LoadLibrary(f"{os.path.dirname(__file__)}/../_rho_j_k_d{distutils.sysconfig.get_config_var('EXT_SUFFIX')}")

c_ext.rho_k.argtypes = (pointer_pos, ctypes.c_int,
                                pointer_k, ctypes.c_int,
                                pointer_dens)

c_ext.rho_j_k.argtypes = (pointer_pos, pointer_vel, ctypes.c_int,
                                pointer_k, ctypes.c_int,
                                pointer_dens, pointer_cur)

def _j_L(Qn, cur):
    """Computes longtitugonal projected current"""
    return (cur*Qn).sum(axis = 1)[..., np.newaxis]*Qn

def _j_T(Qn, cur, cur_L):
    """Computes transversivel projected current"""
    if not isinstance(cur, type(None)) and not isinstance(cur_L, type(None)):
        return cur - cur_L
    elif not isinstance(cur, type(None)) and isinstance(cur_L, type(None)):
        return cur - _j_L(Qn, cur)

def frame_to_signal(pos, vel, Q, Qn, opts):
    np.require(pos, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    np.require(Q, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])
    if not isinstance(vel, type(None)):
        np.require(vel, np.float64, ['F_CONTIGUOUS', 'ALIGNED'])

    res = {'cur_L': None, 'cur': None}
    Natoms = pos.shape[1]
    Nq = Q.shape[1]
    dens = np.zeros(Nq, dtype = np.complex128, order = 'F')
    if len(opts) == 1 and opts[0] == 'dens':
        c_ext.rho_k(pos, Natoms, Q, Nq, dens)
        res['dens'] = dens

    if "cur" in opts or "cur_L" in opts or "cur_T" in opts:
        cur = np.zeros((3,Nq), dtype = np.complex128, order = 'F')
        c_ext.rho_j_k(pos, vel, Natoms, Q, Nq, dens, cur)
        res['dens'] = dens
        res['cur'] = cur.T
    
    if 'cur_L' in opts:
        res['cur_L'] = _j_L(Qn, res['cur'])
    if 'cur_T' in opts:
        res['cur_T'] = _j_T(Qn, res['cur'], res['cur_L'])

    return res

def compute_signal(traj_file, N, Nq_path, Nq, lattice, a, c, M, opts, partial):
    """Initializes computations for signal"""
    
    if os.path.exists(f"qgrid.h5"):
        Q, Qn = grids['file']()
    else:
        if lattice == 'hcp':
            Q, Qn = grids[lattice](Nq_path-1, a, c)
        else:
            Q, Qn = grids[lattice](Nq_path-1, a)
            
    Q = Q[:, :Nq]; Qn = Qn[:Nq]
    Nq = Q.shape[1]
    frame_ind = np.array(range(N))
    ind = get_types_ind()
    if isinstance(M, type(None)):
        M = {key: np.float64(1) for key in ind}
    traj = trajectory_csv(traj_file, ind, M, vel_flag = not (len(opts) == 1 and opts[0] == 'dens'))
    mem = psutil.virtual_memory().available/1/2**20
    Nsplit, Qsplit = signal_mem(mem, max([ind[key].size for key in ind]), Nq, opts)
    
    if Qsplit < 1:
        print("WARNING: Your setup takes too much memory on a single process, job might cancel due to out of memory error")
        Qsplit = 1
    
    return signal(traj, Q, Qn, frame_ind, Nsplit, Qsplit, opts, partial)


def signal(traj, Q, Qn, frame_ind, Nsplit, Qsplit, opts, partial):
    """Computes signal splitted according to memory requirenemnts"""
    Nq = Q.shape[1]
    if partial:
        res = gen_output_dict(traj.ind.keys(), frame_ind.size, Nq, opts)
    else:
        res = gen_output_dict(['1'], frame_ind.size, Nq, opts)
    Q_ind = np.arange(Nq)
    
    Niter, Qiter = get_num_iter(frame_ind.size, Nsplit), get_num_iter(Nq, Qsplit)
    Niter = frame_ind.size; Nsplit = 1
    read_time = 0
    compute_time = 0
    for i in range(frame_ind[0], frame_ind[-1]+1):
        
        start_time_read = time.time()
        pos, vel = traj.read_one_frame()
        read_time += time.time() - start_time_read
        
        start_time_compute = time.time()
        for j in range(Qiter):
            cur_ind_q = Q_ind[j*Qsplit:(j+1)*Qsplit]
            Qcur = Q[:, cur_ind_q[0]:cur_ind_q[-1]+1]
            Qncur = Qn[cur_ind_q[0]:cur_ind_q[-1]+1]
            for k in traj.ind:
                res_tmp = frame_to_signal(pos[traj.ind[k]].transpose(), vel[traj.ind[k]].transpose(), Qcur, Qncur, opts)
                for opt in opts:
                    if partial:
                        res[opt][k][i-frame_ind[0], cur_ind_q[0]:cur_ind_q[-1]+1] = traj.M[k]**(1/2.0)*res_tmp[opt]
                    else:
                        res[opt]['1'][i-frame_ind[0], cur_ind_q[0]:cur_ind_q[-1]+1] += traj.M[k]**(1/2.0)*res_tmp[opt]
        compute_time += time.time() - start_time_compute
        
    traj.close()
    return res, read_time, compute_time
