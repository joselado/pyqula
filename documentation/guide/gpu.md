# Running on a GPU

Some of the heaviest calculations can run on a GPU: the Chebyshev moments of
the [kernel polynomial method](kpm.md), the sum over states of the
[spin and charge response](spin-response.md), and the diagonalization of the
Bloch Hamiltonian on a k-mesh, which is behind band structures, densities of
states and Berry curvatures. One switch decides where they run, and by
default everything runs on the CPU:

```python
from pyqula import gpu
gpu.set_gpu(True) # every routine with a GPU path now uses the device
```

`gpu.set_gpu(False)` goes back to the CPU, and `gpu.is_gpu_available()` says
if a GPU is found, so `gpu.set_gpu(gpu.is_gpu_available())` uses the GPU when
there is one. A script that never calls `gpu.set_gpu()` gives the same
result on every machine. If you ask for a GPU and there is none, the same
code runs on the CPU with a warning, and gives the same numbers.

## When the GPU helps

The gain depends on the kind of calculation:

- The response functions (the Lindhard sum) gain the most. Their cost grows
  as the fourth power of the number of sites in the unit cell, and the whole
  k-mesh is computed at once.
- The diagonalization on a k-mesh only gains for Hamiltonians with about 32
  orbitals or more; for smaller ones it stays on the CPU even with the switch
  on.
- The kernel polynomial density of states gains little, a factor 1 to 3.
- The KPM mean field gains more: on a consumer card, an iteration for a
  unit cell with one to three thousand orbitals is 6 to 10 times faster than
  on the CPU.
- The static polarizability of the screened interaction
  (`get_polarizability`, `get_screened_interaction`, and
  `get_bse(screening=...)`) is faster on a consumer card only in single
  precision, from about 36 orbitals on, by about 7 times. In double
  precision it is slower than the CPU at every size.

## Precision

The precision is chosen in each call, since it changes the numbers and not
only the place where they are computed: `kpm_prec` for the Chebyshev
moments, `chi_prec` for the response functions, and `eigh_prec` for the
diagonalization, each `"single"` or `"double"`. The default is double
precision, except for the response functions and the KPM mean field on the
GPU, which use single precision, because a consumer GPU can be ten times
slower in double precision. Single precision keeps about six digits. This is
enough for a density of states or a band structure, but not for a quantity
computed as the difference of two numbers that are almost equal.

A few routines exist only in jax, for example the mean-field solvers with
`use_jax=True`, the [graphene relaxation](relaxation.md) and the jax Keldysh
current. They follow the same switch, so they also run on the CPU by
default.

See
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/22_gpu_execution.ipynb`
for an executed notebook that compares the CPU and the GPU for each case.
