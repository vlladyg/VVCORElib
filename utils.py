import cupy as cp
import numpy as np
import h5py

def get_num_iter(n, nsplit):
    """Computes number of iterations needed to iterate through n using nsplit chunks"""
    if n % nsplit == 0:
        return n // nsplit
    else:
        return n // nsplit + 1

def get_start_end(N, size, rank):
    """Computes equal partitioning of N along mpi processes"""
    if size <= N:
        if rank < N%size:
            start = N//size*rank + rank
            end = N//size*(rank+1) + rank + 1
        else:
            start = N//size*rank + N%size
            end = N//size*(rank+1) + N%size
    else:
        print("WARNING: number of processes > than number of chunks\nPart of the processes is doing nothing")
        if rank < N:
            start = rank
            end = rank + 1
        else:
            start = None
            end = None
        if isinstance(start, type(None)):
            return None
    return np.array(range(start, end))

def get_types_ind(ind_file):
    """Generates index dict from h5 file"""
    ind = {}
    g = h5py.File(ind_file, 'r')
    for key in list(g.keys()):
        ind[key] = cp.array(g[key])

    g.close()
    return ind

def signal_mem(mem, Natoms, Nq, opts):
    """Returns optimal partitioning on Nq and Number of frames for the process"""
    itemsize = 16 # size of element in the array
    alpha = 0.50 # the part of memory allowed to use

    if 'cur_T' in opts:
        max_mem = 2*itemsize*Natoms*Nq*3/2**20
    else:
        max_mem = 2*itemsize*Natoms*Nq/2**20
    # Checks if there is enough memory for all Nq
    #print(f"max_mem - {max_mem}")
    if max_mem < mem*alpha:
        return int(mem*alpha)//int(max_mem), Nq
    else:
        return 1, int(Nq*mem*alpha/max_mem)

def gen_output_dict(types, Nproc, Nq, opts = 'dens'):
    """Generates random(for memory estimate perposes) output dict"""
    res = {}
    for opt in opts:
        res[opt] = {}
        for t in types:
            if opt == 'dens':
                res[opt][t] = cp.zeros((Nproc, Nq), dtype = cp.complex128)
            else:
                res[opt][t] = cp.zeros((Nproc, Nq, 3), dtype = cp.complex128)
    return res


def stack(s, dim = 0):
    """Converts list of dicts to dict of stacked lists"""
    s_final = {}
    for k in s[0].keys():
        s_final[k] = []
        for i in range(len(s)):
            s_final[k].append(s[i][k])
        if dim == 0:
            s_final[k] = np.vstack(s_final[k])
        elif dim == 1:
            s_final[k] = np.hstack(s_final[k])
    return s_final

def save_signal(s, out):
    """Saves signal to the file out"""
    out_file = h5py.File(out, "w")
    for key in s.keys():
        out_file.create_dataset(key, data = s[key])
    out_file.close()
    pass

def get_start_end_2D(N1, N2, size, rank):
    if N1 < size:
        N1l = range(rank%N1, rank%N1+1)
        if rank % N1 < size%N1:
            splits = size // N1 + 1
        else:
            splits = size // N1
        nsplit = N2 // splits
        if (rank // N1) < (N2 % splits):
            N2l = range(nsplit*(rank // N1) + (rank // N1), nsplit*(rank // N1 + 1) + (rank // N1) + 1)
        else:
            N2l = range(nsplit*(rank // N1) + N2 % splits, nsplit*(rank // N1 + 1) + N2 % splits)
    else:
        N1l = get_start_end(N1, size, rank)
        N2l = range(N2)
    return N1l, N2l

def dict_from_device(data):
    out = {}
    for opt in data.keys():
        out[opt] = {}
        for key in data[opt].keys():
            out[opt][key] = data[opt][key].get()

    return out
