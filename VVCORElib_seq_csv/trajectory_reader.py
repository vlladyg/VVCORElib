import h5py
import numpy as np


class trajectory_h5():
    """Gets pointer to trajectory file
    Saves pointer to a file"""

    def __init__(self, traj_file, ind, M, vel_flag = True):
        if not (traj_file[-3:] == '.h5'):
            raise ValueError("Wrong file formal (not hdf5)")
        self.mode = 'h5md'
        self.file = File(traj_file, 'r')
        self.ind = ind
        self.M = M
        self.vel_flag = vel_flag   

    def get_slice(self, N1, N2=None, axis = 0):
        """Gets slice of trajectory along axis"""
        if self.mode == 'h5md':
            if axis == 0:
                pos = array(self.file['particles']['all']['position']['value'][N1:N2])
                vel = None
                if self.vel_flag:
                    vel = array(self.file['particles']['all']['velocity']['value'][N1:N2])
            elif axis == 1:
                pos = array(self.file['particles']['all']['position']['value'][:, N1:N2])
                vel = None
                if self.vel_flag:
                    vel = array(self.file['particles']['all']['velocity']['value'][:, N1:N2])
            else:
                raise AttributeError
       
        return pos, vel
        
    def close(self):
        self.file.close()


def get_lammps_info(traj_file):
    """Gets number of atoms and dymentions of cell from the lammps dump file"""
    with open(traj_file, 'r') as f:
        for i in range(3):
            f.readline()
        # Reading number of atoms in the sym from the first frame    
        N = int(f.readline().rstrip())
        f.readline()
        # Reading lattice from the simulation from the first frame
        lattice = np.array([float(f.readline().rstrip().split()[1]) for i in range(3)])

    return N, lattice[np.newaxis, :]

class trajectory_csv(trajectory_h5):
    """Gets pointer to trajectory file
    Saves pointer to a file"""

    def __init__(self, traj_file, ind, M, vel_flag = True):
        if not (traj_file[-3:] == 'dat'):   
            raise ValueError("Wrong file formal (not csv)")
        self.mode = 'lammps.dump'
        self.Natoms, self.lattice = get_lammps_info(traj_file)
        self.file = open(traj_file, 'r')
        self.ind = ind
        self.M = M
        self.vel_flag = vel_flag

    def read_one_frame(self):
        """Reads one frame using pointer to the file"""
        for i in range(9):
            self.file.readline()
        data = np.array([self.file.readline().split() for i in range(self.Natoms)], dtype = np.float64, order = 'C')

        pos = data[:, 2:5]*self.lattice
        vel = None
        if self.vel_flag:
            vel = data[:, 5:]
        return pos, vel

    def get_slice(self, N):
        """Gets slice of trajectory along axis"""
        pos = np.zeros((N, self.Natoms, 3), order = 'C')
        vel = None
        if self.vel_flag:
            vel = np.zeros((N, self.Natoms, 3), order = 'C')
        for i in range(N):
            if self.vel_flag:
                pos[i, :, :], vel[i, :, :] = self.read_one_frame()
            else:
                pos[i, :, :], _ = self.read_one_frame()

        return pos, vel
