import numpy as np
import h5py

def get_num_iter(n, nsplit):
    """Computes number of iterations needed to iterate through n using nsplit chunks"""
    if n % nsplit == 0:
        return n // nsplit
    else:
        return n // nsplit + 1

def get_types_ind():
    """Generates index dict from h5 file"""
    g = h5py.File('ind.h5', 'r')
    ind = {key: np.array(g[key]) for key in g.keys()}
    g.close()
    return ind

def signal_mem(mem, Natoms, Nq, opts):
    """Returns optimal partitioning on Nq and Number of frames for the process"""
    itemsize = 8 # size of element in the array
    alpha = 1.0 # the part of memory allowed to use

    if 'cur_T' in opts:
        max_mem = 4*itemsize*Natoms*Nq*3/2**20
    else:
        max_mem = 4*itemsize*Natoms*Nq/2**20
    # Checks if there is enough memory for all Nq
    if max_mem < mem*alpha:
        return int(mem*alpha)//int(max_mem), Nq
    else:
        return 1, int(Nq*mem*alpha/(max_mem))

def gen_output_dict(types, Nproc, Nq, opts = 'dens'):
    """Generates random(for memory estimate perposes) output dict"""
    res = {}
    for opt in opts:
        if opt != 'dens':
            res[opt] = {t: np.zeros((Nproc, Nq, 3), dtype = np.complex128) for t in types}
        else:
            res[opt] = {t: np.zeros((Nproc, Nq), dtype = np.complex128) for t in types}
    return res

def save_signal(s, out):
    """Saves signal to the file out"""
    out_file = h5py.File(out, "w")
    for key in s.keys():
        out_file.create_dataset(key, data = s[key])
    out_file.close()
    pass

