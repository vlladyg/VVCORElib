# VVCORElib — GPU Branch

**GPU-Accelerated Projected Velocity-Velocity Autocorrelation Function Analysis**

A high-performance, GPU-accelerated Python library for computing velocity-velocity autocorrelation functions (VACF) and current correlation functions from molecular dynamics simulations. This branch leverages NVIDIA GPUs via CuPy for massively parallel computation, with MPI support for multi-GPU scaling.

---

## Features

- **GPU Acceleration** — CUDA-powered computations via CuPy for 10-100× speedups
- **Multi-GPU Support** — MPI parallelization across multiple GPUs
- **Automatic GPU Memory Management** — Smart partitioning based on available VRAM
- **Multiple Lattice Types** — Built-in support for FCC, BCC, and SC crystal structures
- **Custom Q-Grids** — Load arbitrary wavevector grids from HDF5 files
- **Current Projections** — Longitudinal (L), transverse (T), and total current correlations
- **HPC Ready** — Designed for GPU clusters with CUDA-aware MPI

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- Recommended: A100, V100, or newer for optimal performance

### Software
- Python 3.8+
- CUDA Toolkit 11.0+
- CUDA-aware MPI implementation (recommended)

---

## Installation

### 1. Set up a GPU-enabled environment

```bash
# Create conda environment
conda create -n vvcore-gpu python=3.10
conda activate vvcore-gpu

# Install CUDA-aware mpi4py (for multi-GPU)
conda install -c conda-forge mpi4py

# Or build from source with CUDA support:
# MPICC="mpicc" pip install mpi4py --no-cache-dir
```

### 2. Install dependencies

```bash
pip install cupy-cuda11x  # Match your CUDA version (cuda11x, cuda12x)
pip install numpy h5py nvidia-ml-py3
```

### 3. Clone the repository

```bash
git clone https://github.com/username/VVCORElib.git
cd VVCORElib
git checkout VVCORE_gpu
```

---

## Dependencies

| Package        | Purpose                                    |
|----------------|--------------------------------------------|
| `cupy`         | GPU array operations (NumPy API on CUDA)   |
| `numpy`        | CPU array operations and I/O               |
| `h5py`         | HDF5 file I/O for trajectories             |
| `mpi4py`       | MPI parallelization for multi-GPU          |
| `nvidia-ml-py3`| GPU memory monitoring (`nvidia_smi`)       |

---

## Quick Start

### Single GPU

```bash
python VVCORE.py 10000  # Process 10000 frames
```

### Multi-GPU with MPI

```bash
mpirun -np 4 python VVCORE.py 10000  # 4 processes across available GPUs
```

### HPC Cluster (SLURM + Cray MPI)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=128
#SBATCH --time=01:00:00

module load PrgEnv-nvidia cray-mpich cudatoolkit craype-accel-nvidia80 python
conda activate vvcore-gpu

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1
export MPICH_GPU_SUPPORT_ENABLED=1

srun -n 128 -G 4 --cpu-bind=cores --gpu-bind=none python VVCORE.py 10000
```

---

## Configuration

Edit `VVCORE.py` to configure your analysis:

```python
# Output options
opts = ['cur', 'cur_T', 'cur_L']  # Current components to compute

# Paths
path = "./data"
traj_file = f"{path}/trajectory.h5"

# Number of frames
N = 10000  # Or pass via command line: sys.argv[1]

# Crystal structure
lattice = 'fcc'
a = 6.125  # Lattice constant in Å
Nq = 30    # Number of q-points along path
```

---

## Observable Options

| Option   | Description                                 |
|----------|---------------------------------------------|
| `dens`   | Density correlation function ρ(q,t)         |
| `cur`    | Total current j(q,t)                        |
| `cur_L`  | Longitudinal current j_L(q,t)               |
| `cur_T`  | Transverse current j_T(q,t)                 |

---

## Input Files

### Trajectory File (HDF5)

Trajectories must be in **H5MD** format:

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

**Creating an index file:**

```python
import h5py
import numpy as np

with h5py.File('ind.h5', 'w') as f:
    f.create_dataset('1', data=np.arange(0, 500))    # Na atoms: 0-499
    f.create_dataset('2', data=np.arange(500, 1000)) # Br atoms: 500-999
```

### Custom Q-Grid (`qgrid.h5`, optional)

For arbitrary wavevector grids:

```python
import h5py
import numpy as np

# Custom q-points array [N_q × 3]
q_points = np.array([...])

with h5py.File('qgrid.h5', 'w') as f:
    f.create_dataset('qx', data=q_points[:, 0])
    f.create_dataset('qy', data=q_points[:, 1])
    f.create_dataset('qz', data=q_points[:, 2])
```

---

## Output Files

| File        | Description                           |
|-------------|---------------------------------------|
| `cur.h5`    | Total current j(q,t)                  |
| `cur_L.h5`  | Longitudinal current j_L(q,t)         |
| `cur_T.h5`  | Transverse current j_T(q,t)           |
| `dens.h5`   | Density ρ(q,t) (if computed)          |

Each file contains datasets keyed by atom type (e.g., `'1'`, `'2'`).

---

## Python API

### Core Modules

| Module                | Purpose                                        |
|-----------------------|------------------------------------------------|
| `signals.py`          | Main signal computation (GPU)                  |
| `signals_cp.py`       | Alternative CuPy implementations               |
| `qgrids.py`           | Q-vector grid generators                       |
| `trajectory_reader.py`| HDF5 trajectory I/O with GPU transfer          |
| `utils.py`            | Memory management, MPI utilities, I/O          |

### Example: Custom Analysis

```python
import cupy as cp
import numpy as np
from mpi4py import MPI
import nvidia_smi

from signals import compute_signal
from utils import stack, dict_from_device, save_signal

comm = MPI.COMM_WORLD

# Initialize GPU
nvidia_smi.nvmlInit()
num_gpus = nvidia_smi.nvmlDeviceGetCount()
cp.cuda.Device(comm.rank % num_gpus).use()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(comm.rank % num_gpus)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

# Compute signals
opts = ['cur', 'cur_L', 'cur_T']
res_gpu, read_time, compute_time = compute_signal(
    traj_file="./data/trajectory.h5",
    N=10000,
    Nq=50,
    lattice='fcc',
    a=4.05,
    opts=opts,
    comm=comm,
    num_of_devices=num_gpus,
    handle=handle,
    info=info
)

# Transfer from GPU to CPU
res = dict_from_device(res_gpu)
cp.cuda.get_current_stream().synchronize()

# Gather and save
for opt in opts:
    gathered = comm.gather(res[opt], root=0)
    if comm.rank == 0:
        stacked = stack(gathered)
        save_signal(stacked, f"{opt}.h5")

nvidia_smi.nvmlShutdown()
```

### Q-Grid Generators

```python
from qgrids import grids

# FCC high-symmetry path: Γ-X-Γ-L
Q, Qn = grids['fcc'](Nq=50, a=4.05)

# BCC high-symmetry path: Γ-H-Γ-N  
Q, Qn = grids['bcc'](Nq=50, a=2.87)

# Simple cubic: Γ-X-M-Γ-R-M
Q, Qn = grids['sc'](Nq=50, a=3.0)

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
```

---

## Theory

### Density in Reciprocal Space

$$\rho(\mathbf{q}, t) = \sum_{j=1}^{N} e^{i\mathbf{q} \cdot \mathbf{r}_j(t)}$$

### Current Density

$$\mathbf{j}(\mathbf{q}, t) = \sum_{j=1}^{N} \mathbf{v}_j(t) \, e^{i\mathbf{q} \cdot \mathbf{r}_j(t)}$$

### Longitudinal Current

$$j_L(\mathbf{q}, t) = \hat{\mathbf{q}} \cdot \mathbf{j}(\mathbf{q}, t)$$

### Transverse Current

$$\mathbf{j}_T(\mathbf{q}, t) = \mathbf{j}(\mathbf{q}, t) - j_L(\mathbf{q}, t) \, \hat{\mathbf{q}}$$

---

## GPU Memory Management

The library automatically partitions work based on available GPU memory:

```python
# From utils.py
def signal_mem(mem, Natoms, Nq, opts):
    """Returns optimal partitioning based on available VRAM"""
    itemsize = 16  # complex128
    alpha = 0.50   # Use 50% of available memory
    
    if 'cur_T' in opts:
        max_mem = 2 * itemsize * Natoms * Nq * 3 / 2**20
    else:
        max_mem = 2 * itemsize * Natoms * Nq / 2**20
    
    if max_mem < mem * alpha:
        return int(mem * alpha) // int(max_mem), Nq
    else:
        return 1, int(Nq * mem * alpha / max_mem)
```

---

## Multi-GPU Scaling

The library supports multiple MPI ranks per GPU for CPU-bound I/O overlap:

```bash
# 4 GPUs, 32 MPI ranks (8 ranks per GPU)
mpirun -np 32 python VVCORE.py 10000
```

GPU assignment is automatic:
```python
cp.cuda.Device(comm.rank % num_of_devices).use()
```

---

## Performance Tips

1. **GPU Memory**: Larger `Nq` values require more VRAM; the library auto-partitions
2. **MPI Ranks**: Use 4-8 ranks per GPU for I/O overlap
3. **CUDA-Aware MPI**: Enable for direct GPU-GPU communication
4. **HDF5 I/O**: Keep trajectories on fast storage (NVMe, parallel FS)
5. **Batch Size**: Larger frame chunks (`Nsplit`) reduce kernel launch overhead

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `Nq` or number of atoms
- Check `alpha` parameter in `signal_mem()` (default 0.5)
- Use fewer MPI ranks per GPU

### Slow Performance
- Ensure CUDA-aware MPI is enabled
- Check GPU utilization with `nvidia-smi`
- Profile with `nvprof` or `nsys`

### MPI Errors with GPU Arrays
- Ensure `dict_from_device()` is called before `comm.gather()`
- Synchronize streams: `cp.cuda.get_current_stream().synchronize()`

---

## Comparison with MPI Branch

| Feature              | GPU Branch          | MPI Branch               |
|----------------------|---------------------|--------------------------|
| Compute Backend      | CuPy (CUDA)         | NumPy + C Extension      |
| Parallelization      | MPI + Multi-GPU     | MPI only                 |
| Memory              | GPU VRAM             | System RAM               |
| Best for            | Large systems        | CPU clusters             |
| Autocorrelation     | Not yet implemented  | ✓ Implemented            |
| Fourier Transform   | Not yet implemented  | ✓ Implemented            |
| Normal Modes        | Not yet implemented  | ✓ Implemented            |

---

## Roadmap

- [ ] GPU-accelerated autocorrelation (`compute_auto`)
- [ ] GPU-accelerated inverse Fourier transform (`compute_ift`)
- [ ] cuFFT integration for spectral analysis
- [ ] Normal mode projections on GPU
- [ ] Multi-node GPU support

---

## File Structure

```
VVCORElib/
├── VVCORE.py              # Main entry point
├── signals.py             # GPU signal computation
├── signals_cp.py          # Alternative CuPy implementations
├── qgrids.py              # Q-vector grid generators
├── trajectory_reader.py   # HDF5 trajectory reader with GPU transfer
├── utils.py               # Utilities (memory, MPI, I/O)
├── run.sh                 # Example SLURM/Cray submission script
├── data/
│   └── NaBr_T300K.h5      # Example trajectory
└── ind.h5                 # Atom type indices
```

---

## Citation

If you use VVCORElib in your research, please cite:

```bibtex
@software{vvcorelib_gpu,
  title = {VVCORElib: GPU-Accelerated Velocity Autocorrelation Analysis},
  author = {Author Name},
  year = {2025},
  url = {https://github.com/username/VVCORElib},
  note = {GPU Branch}
}
```

---

## License

[Add license information here]

---

## Contributing

Contributions are welcome! Key areas for improvement:
- GPU autocorrelation implementation
- cuFFT-based spectral analysis
- Additional lattice types (HCP)
- Performance benchmarks

