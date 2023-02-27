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

setup (name = 'VVCORElib_seq_csv',
       version = '1.0',
       description = 'This is version of VVCORE with sequential reading of csv LAMMPS outputs',
       packages=['VVCORElib_seq_csv'],
       ext_modules = [rho_j_k_d_ext, rho_j_k_s_ext],
       scripts=['VVCORE_seq_csv', 'VVCORE_seq_csv_reduce', 'VVCORE_seq_csv_auto', 'VVCORE_seq_csv_collect', 'VVCORE_seq_csv_ift'],
       install_requires=['numpy', 'scipy', 'psutil', 'csvpy'])
