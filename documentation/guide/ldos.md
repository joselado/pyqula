# Local density of states

In this page we resolve the density of states in space. This is the
quantity that STM measures: the differential conductance $dI/dV$ at bias $V$
and tip position $\mathbf r$ is, in a first approximation, proportional to
the local density of states at energy $eV$ and position $\mathbf r$.

## Local density of states at one energy

The local density of states on site $n$ is

$$
D(\omega,n) = \frac{1}{N_k}\sum_{k,\alpha}
\delta(\omega-\epsilon_{k,\alpha})\, | \langle n | \Psi_{k,\alpha} \rangle |^2 ,
$$

where $\epsilon_{k,\alpha}$ are the eigenenergies,
$\langle n | \Psi_{k,\alpha}\rangle$ is the amplitude of the eigenstate on
site $n$, and the delta function is broadened with a Lorentzian of width
$\delta$. It counts the states at energy $\omega$, each one weighted by its
probability on site $n$. The sum over all the sites gives back the density of
states of [Band structures and DOS](bands-and-dos.md). We compute it with
`h.get_ldos()`:

```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(x,y,d) = h.get_ldos()
```

`h.get_ldos()` returns three arrays: the $x$ and $y$ coordinates of the sites,
and the local density of states on each site, at the energy `e` (zero by
default). For a periodic system the unit cell is repeated a few times, so
the map is easier to see. For the zigzag ribbon at zero energy, about three
quarters of the weight is on the outermost sites of the two edges. These are
the zigzag edge states, and this profile is what STM measures at a zigzag
edge of graphene.

The keywords are

- `e`: the energy $\omega$.
- `delta`: the broadening $\delta$.
- `operator`: weight each state with an operator, for example `"sz"` for a
  map of the spin density of states instead of the charge one.
- `mode`: `"arpack"` (default, diagonalization on a k-mesh), `"green"`
  (Green's function, 2D only) or `"KPM"` (Chebyshev expansion, for large
  systems, see [Kernel polynomial method](kpm.md)).
- `projection`: `"TB"` (default, one value per site), `"TBRS"` (the same
  values, interpolated on a continuous grid in real space) or `"atomic"`
  (each site replaced by an atomic orbital, which gives a smooth map closer
  to an STM image).
- `num_bands`: compute only this many states around `e`, for large
  Hamiltonians.

With an operator $A$, the two modes do not compute exactly the same thing.
`mode="arpack"` weights each eigenstate with its expectation value,
$\langle\Psi|A|\Psi\rangle\,|\Psi(n)|^2$. `mode="green"` and `mode="KPM"` use
the local matrix element $\textrm{Re}\,\Psi^*(n)(A\Psi)(n)$. The two agree
for a state that is an eigenstate of $A$, and summed over the sites both
give the same total. For a state whose spin, say, changes from site to site,
only the local matrix element shows this change in the map. An operator that depends on momentum, such as `"valley"`, only
works with `mode="arpack"`, because the Green's function has already summed
over the Brillouin zone.

For a smoother map than one value per site, use `projection="TBRS"`:

```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(x,y,d) = h.get_ldos(e=0.0,projection="TBRS") # interpolated real-space map
```

To follow how the map changes with energy, as in a series of STM conductance
maps at different biases, `h.get_multildos()` computes the local density of
states at many energies at once:

```python
import numpy as np
(x,y,es,ldos) = h.get_multildos(energies=np.linspace(-2.0,2.0,100),projection="atomic") # one map per energy
```

`x` and `y` are the positions, `es` the energies, and `ldos` has one row for
each energy, so the rows are the frames of a stack $D(x,y,\omega)$. The maps
are also written to the folder `MULTILDOS/`, one file per energy, together
with the total density of states in `MULTILDOS/DOS.OUT`, unless `write=False`
is given. The maps have the same normalization as `h.get_ldos()`, and
`DOS.OUT` the same as `h.get_dos()`, so they can be compared directly. With
an operator, `h.get_multildos()` uses the expectation value
$\langle\Psi|A|\Psi\rangle$ times $|\Psi(n)|^2$, as `mode="arpack"` above.
`projection="atomic"` does not accept an operator.

See `examples/0d/island/main.py` for a runnable script on a superconducting
island with `projection="TBRS"`.

## Local density of states of a finite island

On a finite island the states are discrete, and the map at each energy shows
one or a few of them. For a triangular island of the honeycomb lattice, we
compute the maps at many energies with `h.get_multildos()`:

```python
from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=3,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
h.get_multildos(projection="atomic") # get the LDOS
```

The map at each energy is what STM measures on the island at that bias, and
`DOS.OUT` is the spectrum of the whole island. See
`examples/readme_examples/ldos_island/main.py` for a runnable script that
plots the maps.
