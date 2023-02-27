import numpy as np
import h5py


def hcp(Nq, a, c):
    """Generates high symmetry path for the body-centered cubic lattice"""
    G_M = [[x/Nq*np.pi/a, x/Nq*np.pi/a/3**(1/2.), 0.0] for x in range(Nq+1)]
    M_G = [[(Nq - x)/Nq*2*np.pi/a, 0.0, 0.0] for x in range(Nq+1)]
    G_A = [[0.0, 0.0, x/Nq*np.pi/c] for x in range(Nq+1)]
    A_L = [[x/Nq*np.pi/a, -x/Nq*np.pi/a/3**(1/2.), np.pi/c] for x in range(Nq+1)]
    
    Q = np.array(G_M + M_G + G_A + A_L, dtype = np.float64, order = 'C') + 1e-4
    Qn = Q/np.linalg.norm(Q, axis = 1)[:, np.newaxis]
    return Q.transpose(), Qn

def fcc(Nq, a):
    """Generates high symmetry paths for the face-centered cubic lattice"""
    G_X = [[0.0, x/Nq*2*np.pi/a, 0.0] for x in range(Nq+1)]
    X_G = [[(Nq - x)/Nq*2*np.pi/a, (Nq-x)/Nq*2*np.pi/a, 0.0] for x in range(Nq+1)]
    G_L = [[x/(Nq//2)*np.pi/a, x/(Nq//2)*np.pi/a, x/(Nq//2)*np.pi/a] for x in range((Nq//2)+1)]

    Q = np.array(G_X + X_G + G_L, dtype = np.float64, order = 'C') + 1e-4
    Qn = Q/np.linalg.norm(Q, axis = 1)[:, np.newaxis]
    return Q.transpose(), Qn



def sc(Nq, a):
    """Generates high symmetry paths for the simple cubic lattice"""
    G_X = [[0.0, x/Nq*np.pi/a, 0.0] for x in range(Nq+1)]
    X_M = [[x/Nq*np.pi/a, np.pi/a, 0.0] for x in range(Nq+1)]
    M_G = [[(Nq - x)/Nq*np.pi/a, (Nq-x)/Nq*np.pi/a, 0.0] for x in range(Nq+1)]
    G_R = [[x/Nq*np.pi/a, x/Nq*np.pi/a, x/Nq*np.pi/a] for x in range(Nq+1)]
    R_M = [[np.pi/a, np.pi/a, (Nq - x)/Nq*np.pi/a] for x in range(Nq+1)]

    Q = np.array(G_X + X_M + M_G + G_R + R_M, dtype = np.float64, order = 'C') + 1e-4
   
    Qn = Q/np.linalg.norm(Q, axis = 1)[:, np.newaxis]
    return Q.transpose(), Qn

def bcc(Nq, a):
    """Generates high symmetry path for the body-centered cubic lattice"""
    G_H = [[0.0, 0.0, x/Nq*2*np.pi/a] for x in range(Nq+1)]
    H_G = [[(Nq - x)/Nq*2*np.pi/a, (Nq-x)/Nq*2*np.pi/a, (Nq-x)/Nq*2*np.pi/a] for x in range(Nq+1)]
    G_N = [[0.0, x/(Nq//2)*np.pi/a, x/(Nq//2)*np.pi/a] for x in range((Nq//2)+1)]

    Q = np.array(G_H + H_G + G_N, dtype = np.float64, order = 'C') + 1e-4
    Qn = Q/np.linalg.norm(Q, axis = 1)[:, np.newaxis]
    return Q.transpose(), Qn


def from_file():
    qgrid_f = h5py.File("qgrid.h5")
    Q = [qgrid_f[key] for key in qgrid_f.keys()]
    Q = np.array(np.vstack(Q), order = 'C')
    Qn = Q/np.linalg.norm(Q, axis = 1)[:, np.newaxis]
    return Q.transpose(), Qn


grids = {'hcp': hcp, 'fcc': fcc, 'bcc': bcc, 'sc': sc, 'file': from_file}
