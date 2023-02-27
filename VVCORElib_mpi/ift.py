import numpy as np
import h5py
import psutil

from .utils import get_num_iter, get_start_end_2D

def ift_mem(mem, N, N_freq):
    """Returns optimal partitioning on Nq and Number of frames for the process"""
    itemsize = 8 # size of element in the array
    alpha = 0.7 # the part of memory allowed to use

    max_mem = 4*itemsize*N*N_freq/2**20
    # Checks if there is enough memory for all Nq
    if max_mem < mem*alpha:
        return N_freq
    else:
        return int(N_freq*mem/(max_mem*alpha))

def compute_ift(path, N, Nq, ts, wmax, dw, opts, comm):
    """Computes ift for the signal"""
    freq = np.arange(0, wmax, dw)
    time = np.arange(0, N)*ts/1000.

    q_ind, freq_ind = get_start_end_2D(Nq, len(freq), comm.size, comm.rank)
    mem = psutil.virtual_memory().available/comm.size/2**20
    N_freq_split = ift_mem(mem, N, len(freq_ind))

    res = {}
    for opt in opts:
        signal_f = h5py.File(f"{path}/auto_{opt}.h5", 'r')
        signal = {key: np.array(signal_f[key][:, q_ind[0]:q_ind[-1]+1]) for key in signal_f.keys()}
        
        res[opt] = ift(signal, freq[freq_ind[0]:freq_ind[-1]+1], time, N_freq_split)

        signal_f.close()
    return res

def ift(s, freq, time, N_freq_split):
    """Performs inverse Furior transform by simple summation"""
    N_freq_iter = get_num_iter(len(freq), N_freq_split)
    spec = {}
    for key in s.keys(): 
        spec[key] = np.zeros((len(freq), s[key].shape[1]), dtype = np.complex128)
    for k in range(N_freq_iter):
        W, T = np.meshgrid(freq[k*N_freq_split:(k+1)*N_freq_split], time)
        precomp = np.exp(-2.0j*np.pi*T*W)
        for key in s.keys():
            for i in range(s[key].shape[1]):
                spec[key][k*N_freq_split:(k+1)*N_freq_split, i] = np.mean(s[key][:, i][:, np.newaxis]*precomp, axis = 0)
    return spec

def stack_ift(Nq, Nfreq, s, comm):
    """Converts list of dicts to dict of stacked lists in the case of 2D net"""
    s_final = {}
    for k in s[0].keys():
        s_final[k] = np.zeros((Nfreq, Nq), dtype = np.complex128)
        for i in range(len(s)):
            freq_ind = get_start_end_2D(Nq, Nfreq, comm.size, i)[1]
            s_final[k][freq_ind[0]:freq_ind[-1]+1, i%Nq] = s[i][k][:, 0]
    return s_final
