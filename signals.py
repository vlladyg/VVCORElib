import cupy as cp
import numpy as np
import os
import time

from utils import get_types_ind, gen_output_dict, get_num_iter, signal_mem, get_start_end
from qgrids import grids
from trajectory_reader import trajectory_h5

def _dens(p_proj):
    """Computes projected density"""
    return p_proj.sum(axis = 1)

def _j(p_proj, vel):
    """Computes full projected current"""
    return (vel[:, :, cp.newaxis] * (p_proj[..., cp.newaxis])).sum(axis = 1)

def _j_L(Qn, cur):
    """Computes longtitugonal projected current"""
    return (cur*Qn).sum(axis = 2)[..., cp.newaxis]*Qn

def _j_T(Qn, cur, cur_L):
    """Computes transversivel projected current"""
    if not isinstance(cur, type(None)) and not isinstance(cur_L, type(None)):
        return cur - cur_L
    elif not isinstance(cur, type(None)) and isinstance(cur_L, type(None)):
        return cur - _j_L(Qn, cur)

def frame_to_signal(p_proj, vel, Qn, opts):
    v_proj = None
    
    res = {'cur_L': None, 'cur': None}
    if "cur" in opts or "cur_L" in opts or "cur_T" in opts:
        res['cur'] = _j(p_proj, vel)
    
    if 'cur_L' in opts:
        res['cur_L'] = _j_L(Qn, res['cur'])
    if 'cur_T' in opts:
        res['cur_T'] = _j_T(Qn, res['cur'], res['cur_L'])
    if 'dens' in opts:
        res['dens'] = _dens(p_proj)

    return res

def compute_signal(traj_file, N, Nq, lattice, a, opts, comm, num_of_devices, handle, info):
    """Initializes computations for signal"""
    traj = trajectory_h5(traj_file, vel_flag = not (len(opts) == 1 and opts[0] == 'dens'))
    
    if os.path.exists(f"qrid.h5"):
        Q, Qn = grids['file']()
    else:
        Q, Qn = grids[lattice](Nq, a)
    
    Q = cp.array(Q[:Nq]); Qn = cp.array(Qn[:Nq])
    frame_ind = get_start_end(N, comm.size, comm.rank)
    ind = get_types_ind("ind.h5")


    start_time_init = time.time()
    
    mem = info.free/2**20/(comm.size//num_of_devices)

    
    Nsplit, Qsplit = signal_mem(mem, max([ind[key].size for key in ind]), Nq, opts)
    
    if Qsplit < 1:
        print("WARNING: Your setup takes too much memory on a single process, job might cancel due to out of memory error")
        Qsplit = 1
    res, read_time, compute_time = signal(traj, Q, Qn, ind, frame_ind, Nsplit, Qsplit, opts)
    return res, read_time, compute_time


def signal(traj, Q, Qn, ind, frame_ind, Nsplit, Qsplit, opts):
    """Computes signal splitted according to memory requirenemnts"""
    Nq = Q.shape[0]
    res = gen_output_dict(ind.keys(), frame_ind.size, Nq, opts)
    Q_ind = np.arange(Nq)
    Niter, Qiter = get_num_iter(frame_ind.size, Nsplit), get_num_iter(Nq, Qsplit)
    #Niter = frame_ind.size; Nsplit = 1
    read_time = 0
    compute_time = 0
    for i in range(Niter):
        cur_ind = frame_ind[Nsplit*i:Nsplit*(i+1)]
        
        start_time_read = time.time()
        pos, vel = traj.get_slice(cur_ind[0], cur_ind[-1]+1)
        read_time += time.time() - start_time_read        

        start_time_compute = time.time()
        for j in range(Qiter):
            cur_ind_q = Q_ind[j*Qsplit:(j+1)*Qsplit]

            Qcur = Q[cur_ind_q[0]:cur_ind_q[-1]+1]
            Qncur = Qn[cur_ind_q[0]:cur_ind_q[-1]+1]
            p_proj = cp.exp(1.0j*(pos@Qcur.T))
            for k in ind:
                res_tmp = frame_to_signal(p_proj[:, ind[k]], vel[:, ind[k]], Qncur, opts)
                for opt in opts:
                    res[opt][k][cur_ind[0]-frame_ind[0]:cur_ind[-1]-frame_ind[0]+1, cur_ind_q[0]:cur_ind_q[-1]+1] = res_tmp[opt]
        compute_time += time.time() - start_time_compute
    traj.close()
    return res, read_time, compute_time
