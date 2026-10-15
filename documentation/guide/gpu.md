# Running on a GPU

Three of the heaviest kernels of the library have a second implementation
written in jax, which runs on a GPU: the Chebyshev moments behind the KPM
density of states, the Lindhard sum behind the RPA response functions, and
the batched diagonalization of the Bloch Hamiltonian over a k-mesh, which
band structures, densities of states and Berry curvatures all bottom out
in. Which implementation runs is decided by one switch, and the package
stays on the CPU until that switch is set

```python
from pyqula import gpu
gpu.set_gpu(True) # every routine with a GPU path now uses the device
```

so that a script that never calls it computes what it computed before, on
whichever machine it lands on. `gpu.set_gpu(False)` goes back to the CPU,
and `gpu.is_gpu_available()` says whether jax sees a device at all, which
is what to pass when a script should simply take what the machine offers.
Asking for a GPU where there is none is not an error: the same jax kernels
then run on jax's own CPU backend and give the same numbers, with a warning
saying that the speedup is not there.

How much the device buys depends on the shape of the calculation more than
on its size. The Lindhard sum gains the most, since its cost grows as the
fourth power of the number of sites in the cell and the whole k-mesh is
contracted on the device at once. The Chebyshev moments gain little, a
factor of one to three, because a sparse matrix-vector product is limited
by memory bandwidth rather than by arithmetic. The batched diagonalization
sits in between, and only pays from matrices of about 32 orbitals upward,
so below that size it stays on the CPU even with the switch set. The KPM
mean field is the exception among the Chebyshev routines: its density
matrix needs one recursion per orbital, which runs as a single block over
every orbital and every k-point, so the card works on a dense block rather
than on one vector at a time, and on a consumer card an iteration of a cell
with one to three thousand orbitals is six to ten times faster than on the
processor.

One sum runs against that pattern and is worth knowing about before
setting the switch. The static polarizability behind the screened
interaction (`get_polarizability`, `get_screened_interaction` and any
`get_bse(screening=...)`) produces only a small matrix per q-point out of
a very long sum, so what it asks of the device is a great many small
products rather than a few large ones. On a consumer card it is faster
than the processor in single precision from about 36 orbitals upward, by
around seven times, and slower than the processor in double precision at
every size, which is the one place in the library where asking for double
precision on a GPU costs rather than buys. Leaving `chi_prec` unset gives
single precision there, which is the case that pays.

Precision is a separate choice, made per call rather than globally, because
it changes the numbers and not only where they are computed. The KPM
moments take `kpm_prec`, the Lindhard sum takes `chi_prec` and the batched
diagonalization takes `eigh_prec`, each of them `"single"` or `"double"`.
Double precision is the default everywhere except the Lindhard sum and the
KPM mean field on the GPU, where single precision is the default because a
consumer card can be an order of magnitude slower in double precision than
in single. Single
precision costs about six digits, which a density of states or a band
structure does not notice, and which a quantity built from differences of
nearly equal numbers does.

A handful of routines written in jax have no CPU implementation of their
own, among them the mean-field solvers reached with `use_jax=True`, the
graphene relaxation and the jax Keldysh current. Those follow the same
switch, so they too run on the CPU by default.

See
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/22_gpu_execution.ipynb`
for an executed notebook comparing the two paths on each kernel.
