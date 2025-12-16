# VVCORElib

**Projected Velocity-Velocity Autocorrelation Function Analysis Library**

A high-performance, MPI-parallelized Python library for computing velocity-velocity autocorrelation functions (VACF) and spectral properties from molecular dynamics (MD) simulations. Designed for extracting phonon dispersion relations and dynamical structure factors from atomistic trajectory data.

---

## Features

- **MPI Parallelization** — Efficiently scales across multiple cores for large trajectory processing
- **Multiple Lattice Types** — Built-in support for FCC, BCC, SC, and HCP crystal structures
- **Custom Q-Grids** — Load arbitrary wavevector grids from HDF5 files
- **Partial Contributions** — Compute per-species contributions in multi-component systems
- **Current Projections** — Longitudinal (L), transverse (T), and total current correlations
- **Normal Mode Projections** — Project velocities onto phonon eigenvectors (TDEP integration)
- **Memory-Aware Processing** — Automatic memory management for large datasets
- **C Extension Core** — Optimized C routines for computing density and current in reciprocal space

---

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.7+
- C compiler (gcc, clang, or compatible)
- MPI implementation (OpenMPI, MPICH, etc.)

### Install from source

```bash
git clone https://github.com/username/VVCORElib.git
cd VVCORElib
pip install .
```

Or install in development mode:

```bash
pip install -e .
```

---

## Dependencies

| Package   | Purpose                                    |
|-----------|--------------------------------------------|
| `numpy`   | Numerical array operations                 |
| `scipy`   | Signal correlation functions               |
| `h5py`    | HDF5 file I/O for trajectories and results |
| `mpi4py`  | MPI parallelization                        |
| `psutil`  | Memory management                          |

---

## Quick Start

### Full Pipeline (Signal → Autocorrelation → Spectrum)

```bash
mpirun -np 16 VVCORE_mpi \
    -path ./trajectory \
    -i trajectory.h5 \
    -N_frames 10000 \
    -N_auto 1000 \
    -lattice fcc \
    -a 4.05 \
    -Nq_path 50 \
    -Nq 150 \
    -M "26.98" \
    -opt "dens cur_L cur_T" \
    -ts 25 \
    -wmax 40 \
    -dw 0.05
```

### Step-by-Step Execution

For more control, run each stage separately:

**1. Compute Signal (density/current in reciprocal space):**
```bash
mpirun -np 16 VVCORE_mpi_reduce \
    -path ./trajectory \
    -i trajectory.h5 \
    -N_frames 10000 \
    -lattice fcc \
    -a 4.05 \
    -Nq_path 50 \
    -Nq 150 \
    -M "26.98" \
    -opt "dens cur_L cur_T"
```

**2. Compute Autocorrelation:**
```bash
mpirun -np 16 VVCORE_mpi_auto \
    -path ./trajectory \
    -N_frames 10000 \
    -N_auto 1000 \
    -Nq 150 \
    -opt "dens cur_L cur_T"
```

**3. Compute Spectrum (Inverse Fourier Transform):**
```bash
mpirun -np 16 VVCORE_mpi_ift \
    -path ./trajectory \
    -N_auto 1000 \
    -Nq 150 \
    -opt "dens cur_L cur_T" \
    -ts 25 \
    -wmax 40 \
    -dw 0.05
```

**4. Collect/Average Multiple Trajectories:**
```bash
mpirun -np 16 VVCORE_mpi_collect \
    -path ./data \
    -folders "traj1 traj2 traj3 traj4" \
    -Nq 150 \
    -opt "dens cur_L cur_T"
```

---

## Command-Line Interface

### VVCORE_mpi

Full analysis pipeline: signal → autocorrelation → spectrum.

| Argument          | Default            | Description                                           |
|-------------------|--------------------|-------------------------------------------------------|
| `-path`           | `./`               | Path to trajectory folder                             |
| `-i`              | `data.dat`         | Input trajectory file (HDF5 format)                   |
| `-opt`            | `dens cur_T cur_L` | Quantities to compute (space-separated)               |
| `-N_frames`       | `10000`            | Number of frames to process                           |
| `-N_auto`         | `1000`             | Correlation window length                             |
| `-lattice`        | `fcc`              | Lattice type: `fcc`, `bcc`, `sc`, `hcp`, `file`       |
| `-Nq_path`        | `50`               | Points per high-symmetry segment                      |
| `-Nq`             | `50`               | Total number of q-points                              |
| `-a`              | —                  | Lattice constant (Å)                                  |
| `-c`              | —                  | c-axis lattice constant (HCP only)                    |
| `-M`              | —                  | Atomic masses (space-separated string)                |
| `-ts`             | `25`               | Timestep (fs)                                         |
| `-wmax`           | `40`               | Maximum frequency (THz)                               |
| `-dw`             | `0.05`             | Frequency resolution (THz)                            |
| `--partial`       | `False`            | Compute partial (per-species) contributions           |
| `--normal_modes`  | `False`            | Project onto phonon normal modes                      |
| `--phase_noise`   | `False`            | Add velocity noise for testing                        |
| `-noise_ind`      | `0`                | Species index for noise injection                     |
| `-noise_ratio`    | `0.0`              | Fraction of atoms with flipped velocities             |

### Observable Options (`-opt`)

| Option   | Description                                      |
|----------|--------------------------------------------------|
| `dens`   | Density correlation function S(q,t)              |
| `cur_L`  | Longitudinal current correlation C_L(q,t)        |
| `cur_T`  | Transverse current correlation C_T(q,t)          |
| `cur`    | Total current (before projection)                |
| `v_k`    | Normal mode projected velocities (requires TDEP) |

---

## Input Files

### Trajectory File (HDF5)

The library expects trajectories in **H5MD** format:

```
trajectory.h5
├── particles/
│   └── all/
│       ├── position/
│       │   └── value   [N_frames × N_atoms × 3]
│       └── velocity/
│           └── value   [N_frames × N_atoms × 3]
```

### Atom Type Indices (`ind.h5`)

Required file specifying atom type indices:

```
ind.h5
├── 1   [indices of type 1 atoms]
├── 2   [indices of type 2 atoms]
└── ...
```

### Custom Q-Grid (`qgrid.h5`, optional)

For arbitrary wavevector grids (use `-lattice file`):

```
qgrid.h5
├── qx   [N_q array of q_x components]
├── qy   [N_q array of q_y components]
└── qz   [N_q array of q_z components]
```

### Normal Mode Eigenvectors (TDEP format, optional)

For normal mode projections (`--normal_modes`):

```
outfile.dispersion_relations.hdf5
├── eigenvectors_re   [N_q × N_modes × N_atoms×3]
└── eigenvectors_im   [N_q × N_modes × N_atoms×3]
```

---

## Output Files

| File              | Description                                         |
|-------------------|-----------------------------------------------------|
| `dens.h5`         | Density signal ρ(q,t)                               |
| `cur_L.h5`        | Longitudinal current j_L(q,t)                       |
| `cur_T.h5`        | Transverse current j_T(q,t)                         |
| `auto_dens.h5`    | Density autocorrelation ⟨ρ(q,t)ρ*(q,0)⟩             |
| `auto_cur_L.h5`   | Longitudinal current autocorrelation                |
| `auto_cur_T.h5`   | Transverse current autocorrelation                  |
| `spec_dens.h5`    | Dynamic structure factor S(q,ω)                     |
| `spec_cur_L.h5`   | Longitudinal current spectrum                       |
| `spec_cur_T.h5`   | Transverse current spectrum                         |

---

## Python API

### Core Functions

```python
from VVCORElib_mpi import (
    compute_signal,              # Compute density/current in reciprocal space
    compute_signal_normal_modes, # Project onto phonon eigenvectors
    compute_auto,                # Compute autocorrelation functions
    compute_ift,                 # Inverse Fourier transform to frequency domain
    collect_auto,                # Average autocorrelations across trajectories
    save_signal,                 # Save results to HDF5
    stack,                       # Stack distributed results
    grids,                       # Q-grid generators
)
```

### Example: Custom Analysis Script

```python
from mpi4py import MPI
import numpy as np
from VVCORElib_mpi import compute_signal, compute_auto, compute_ift
from VVCORElib_mpi import save_signal, stack

comm = MPI.COMM_WORLD

# Parameters
traj_file = "trajectory.h5"
N_frames = 10000
N_auto = 1000
Nq_path = 50
Nq = 150
a = 4.05  # Lattice constant
M = {'1': np.float64(26.98)}  # Aluminum
opts = ['dens', 'cur_L', 'cur_T']

# 1. Compute signals
res, _, _ = compute_signal(
    traj_file, N_frames, Nq_path, Nq, 
    lattice='fcc', a=a, c=None, M=M, 
    opts=opts, partial=False,
    phase_noise=False, noise_ind=0, noise_ratio=0.0,
    comm=comm
)

# Gather and save
for opt in opts:
    gathered = comm.gather(res[opt], root=0)
    if comm.rank == 0:
        stacked = stack(gathered)
        save_signal(stacked, f"{opt}.h5")

# 2. Compute autocorrelation
auto_res = compute_auto('./', N_frames, Nq, N_auto, opts, comm)

# 3. Compute spectrum
spec_res = compute_ift('./', N_auto, Nq, ts=25, wmax=40, dw=0.05, opts=opts, comm=comm)
```

### Q-Grid Generators

```python
from VVCORElib_mpi import grids

# FCC high-symmetry path: Γ-X-Γ-L
Q, Qn = grids['fcc'](Nq=50, a=4.05)

# BCC high-symmetry path: Γ-H-Γ-N  
Q, Qn = grids['bcc'](Nq=50, a=2.87)

# Simple cubic: Γ-X-M-Γ-R-M
Q, Qn = grids['sc'](Nq=50, a=3.0)

# HCP: Γ-M-Γ-A-L
Q, Qn = grids['hcp'](Nq=50, a=3.2, c=5.2)

# Load from file
Q, Qn = grids['file']()  # Reads qgrid.h5
```

---

## High-Symmetry Paths

### FCC (Face-Centered Cubic)
```
Γ → X → Γ → L
(0,0,0) → (0,2π/a,0) → (2π/a,2π/a,0) → (π/a,π/a,π/a)
```

### BCC (Body-Centered Cubic)
```
Γ → H → Γ → N
(0,0,0) → (0,0,2π/a) → (2π/a,2π/a,2π/a) → (0,π/a,π/a)
```

### SC (Simple Cubic)
```
Γ → X → M → Γ → R → M
(0,0,0) → (0,π/a,0) → (π/a,π/a,0) → (0,0,0) → (π/a,π/a,π/a) → (π/a,π/a,0)
```

### HCP (Hexagonal Close-Packed)
```
Γ → M → Γ → A → L
```

---

## Theory

### Density Correlation

The instantaneous density in reciprocal space:

$$\rho(\mathbf{q}, t) = \sum_{j=1}^{N} e^{i\mathbf{q} \cdot \mathbf{r}_j(t)}$$

### Current Correlation

The mass current density:

$$\mathbf{j}(\mathbf{q}, t) = \sum_{j=1}^{N} \sqrt{m_j} \, \mathbf{v}_j(t) \, e^{i\mathbf{q} \cdot \mathbf{r}_j(t)}$$

**Longitudinal projection:**
$$j_L(\mathbf{q}, t) = \hat{\mathbf{q}} \cdot \mathbf{j}(\mathbf{q}, t)$$

**Transverse projection:**
$$\mathbf{j}_T(\mathbf{q}, t) = \mathbf{j}(\mathbf{q}, t) - j_L(\mathbf{q}, t) \hat{\mathbf{q}}$$

### Autocorrelation and Spectrum

The autocorrelation function:

$$C(\mathbf{q}, t) = \langle A(\mathbf{q}, t) A^*(\mathbf{q}, 0) \rangle$$

The dynamic spectrum via Fourier transform:

$$S(\mathbf{q}, \omega) = \int_{-\infty}^{\infty} C(\mathbf{q}, t) \, e^{-i\omega t} \, dt$$

---

## Performance Tips

1. **Optimal MPI Ranks**: Use `N_ranks ≈ N_q` for best load balancing
2. **Memory**: The library auto-partitions based on available RAM
3. **Chunk Size**: Set `-N_auto` to divide evenly into `-N_frames`
4. **I/O**: HDF5 trajectories significantly outperform ASCII formats
5. **Large Systems**: Increase ranks rather than per-rank memory

---

## Troubleshooting

### "WARNING: number of processes > than number of chunks"
Reduce MPI ranks or increase workload (more q-points or frames).

### Memory errors
The library attempts automatic memory management. For very large systems:
- Reduce `-Nq` and run in batches
- Increase available RAM per process
- Use fewer MPI ranks with more memory each

### Missing `ind.h5`
Create an index file mapping atom types:
```python
import h5py
import numpy as np

with h5py.File('ind.h5', 'w') as f:
    f.create_dataset('1', data=np.arange(0, 500))    # Type 1: atoms 0-499
    f.create_dataset('2', data=np.arange(500, 1000)) # Type 2: atoms 500-999
```

---

## Citation

If you use VVCORElib in your research, please cite:

```bibtex
@software{vvcorelib,
  title = {VVCORElib: Projected Velocity-Velocity Autocorrelation Analysis},
  author = {Author Name},
  year = {2025},
  url = {https://github.com/username/VVCORElib}
}
```

---

## License

[Add license information here]

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
