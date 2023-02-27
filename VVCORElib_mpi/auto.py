import numpy as np
from scipy import signal
import h5py

from .utils import get_start_end_2D, get_start_end

def compute_auto(path, N, Nq, chunk_size, opts, comm):
    assert(N % chunk_size == 0)
    Nchunks = N//chunk_size
    q_ind, chunk_ind = get_start_end_2D(Nq, Nchunks, comm.size, comm.rank)
    res = {}
    for opt in opts:
        signal_f = h5py.File(f"{path}/{opt}.h5", 'r')
        signal = {key: np.array(signal_f[key][chunk_ind[0]*chunk_size:(chunk_ind[-1]+1)*chunk_size, q_ind[0]:q_ind[-1]+1]) for key in signal_f.keys()}
        
        res[opt] = auto_transform(signal, chunk_size)
    
        signal_f.close()
    return res

def auto_transform(s, chunk_size):
    signal_auto = {}
    for t1 in s.keys():
        for t2 in s.keys():
            key1 = t1 + '_' + t2
            key2 = t2 + '_' + t1
            if key1 not in signal_auto.keys() and key2 not in signal_auto.keys():
                signal_auto[key1] = np.zeros((chunk_size, s[t1].shape[1]), dtype = np.complex128)
                for k in range(s[t1].shape[0]//chunk_size):
                    for i in range(s[t1].shape[1]):
                        if len(s[t1].shape) == 3:
                            for m in range(s[t1].shape[2]):
                                signal_auto[key1][:, i] += signal.correlate(s[t1][k*chunk_size:(k+1)*chunk_size, i, m], s[t2][k*chunk_size:(k+1)*chunk_size, i, m], mode = "full")[-chunk_size:]
                        elif len(s[t1].shape) == 2:
                            signal_auto[key1][:, i] += signal.correlate(s[t1][k*chunk_size:(k+1)*chunk_size, i], s[t2][k*chunk_size:(k+1)*chunk_size, i], mode = "full")[-chunk_size:]

    return signal_auto


def stack_auto(Nq, Nchunks, s):
    """Converts list of dicts to dict of stacked lists"""
    s_final = {}
    for k in s[0].keys():
        s_final[k] = np.zeros((s[0][k].shape[0], Nq), dtype = np.complex128)
        for i in range(len(s)):
            s_final[k][:, i%Nq] += s[i][k][:, 0]
        s_final[k] /= Nchunks
    return s_final

def collect_auto(path, folders, Nq, opts, comm):
    """Averages autocorrelated signal from multiple trajectories"""
    q_ind = get_start_end(Nq, comm.size, comm.rank)
    
    res = {opt: {} for opt in opts}
    
    N_fold = len(folders)
    for el in folders:
        for opt in opts:
            signal_f = h5py.File(f"{path}/{el}/auto_{opt}.h5", 'r')
            if len(res[opt]) == 0:
                res[opt] = {key: np.array(signal_f[key][:, q_ind[0]:q_ind[-1]+1])/(N_fold) for key in signal_f.keys()}
            else:
                for key in signal_f.keys():
                    res[opt][key] += np.array(signal_f[key][:, q_ind[0]:q_ind[-1]+1])/(N_fold)
            signal_f.close()

    return res