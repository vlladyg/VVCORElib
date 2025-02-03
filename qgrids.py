import numpy as np
import h5py

def fcc(Nq, a):
    """Generates high symmetry paths for the face-centered cubic lattice"""
    G_X = [[0.0, x/Nq*2*np.pi/a, 0.0] for x in range(Nq+1)]
    X_G = [[(Nq - x)/Nq*2*np.pi/a, (Nq-x)/Nq*2*np.pi/a, 0.0] for x in range(Nq+1)]
    G_L = [[x/Nq*np.pi/a, x/Nq*np.pi/a, x/Nq*np.pi/a] for x in range(Nq+1)]

    Q = np.array(G_X + X_G + G_L, dtype = np.float64) + 1e-4
    Qn = Q/np.linalg.norm(Q, axis = 1)[:, np.newaxis]
    return Q, Qn



def sc(Nq, a):
    """Generates high symmetry paths for the simple cubic lattice"""
    G_X = [[0.0, x/Nq*np.pi/a, 0.0] for x in range(Nq+1)]
    X_M = [[x/Nq*np.pi/a, np.pi/a, 0.0] for x in range(Nq+1)]
    M_G = [[(Nq - x)/Nq*np.pi/a, (Nq-x)/Nq*np.pi/a, 0.0] for x in range(Nq+1)]
    G_R = [[x/Nq*np.pi/a, x/Nq*pi/a, x/Nq*np.pi/a] for x in range(Nq+1)]
    R_M = [[np.pi/a, np.pi/a, (Nq - x)/Nq*np.pi/a] for x in range(Nq+1)]

    Q = np.array(G_X + X_M + M_G + G_R + R_M, dtype = np.float64)
   
    Qn = Q/np.linalg.norm(Q, axis = 1)[:, np.newaxis]
    return Q, Qn

def bcc(Nq, a):
    """Generates high symmetry path for the body-centered cubic lattice"""
    G_H = [[0.0, 0.0, x/Nq*2*np.pi/a] for x in range(Nq+1)]
    H_G = [[(Nq - x)/Nq*2*np.pi/a, (Nq-x)/Nq*2*np.pi/a, (Nq-x)/Nq*2*np.pi/a] for x in range(Nq+1)]
    G_N = [[0.0, x/Nq*np.pi/a, x/Nq*np.pi/a] for x in range(Nq+1)]

    Q = np.array(G_H + H_G + G_N, dtype = np.float64) + 1e-4
    Qn = Q/np.linalg.norm(Q, axis = 1)[:, np.newaxis]
    return Q, Qn


def from_file():
    qgrid_f = hdf5.File("qgrid.h5")
    Q = [qgrid[key] for key in qgrid.keys()]
    Q = np.vstack(Q)
    Qn = Q/norm(Q, axis = 1)[:, np.newaxis]
    return Q, Qn


grids = {'fcc': fcc, 'bcc': bcc, 'sc': sc, 'file': from_file}
