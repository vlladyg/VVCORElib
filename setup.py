#!/usr/bin/env python

from distutils.core import setup, Extension

rho_j_k_d_ext = Extension('_rho_j_k_d',
                          sources=['src/_rho_j_k.c'],
                          define_macros=[('RHOPREC', 'double')],
                          extra_compile_args=[],
                          extra_link_args=[],
                          )

rho_j_k_s_ext = Extension('_rho_j_k_s',
                          sources=['src/_rho_j_k.c'],
                          define_macros=[('RHOPREC', 'float')],
                          extra_compile_args=[],
                          extra_link_args=[],
                          )

setup (name = 'VVCORElib_mpi',
       version = '1.0',
       description = 'This is version of VVCORE with sequential reading of csv LAMMPS outputs',
       packages=['VVCORElib_mpi'],
       ext_modules = [rho_j_k_d_ext, rho_j_k_s_ext],
       scripts=['VVCORE_mpi', 'VVCORE_mpi_reduce', 'VVCORE_mpi_auto', 'VVCORE_mpi_collect', 'VVCORE_mpi_ift'],
       install_requires=['numpy', 'scipy', 'psutil', 'h5py', 'mpi4py'])
