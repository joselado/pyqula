# Local density of states

## Local density of states

We will now resolve the density of states in space, with `h.get_ldos()`, which gives the
quantity that STM measures. The local density of states resolves the density of states by
site: it counts how many states are in a certain energy window, weighted by how much of each
state sits on site $n$. It is defined as

$$
D(\omega,n) = \int \delta(\omega-\epsilon_k) | \langle \Psi_k | n \rangle |^2 dk
$$

where $\epsilon_k$ are the eigenenergies of the Hamiltonian and $\langle \Psi_k | n \rangle$
is the amplitude of each eigenstate on site $n$, so that summing over the sites gives back
the density of states. It can be used as shown below

```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(x,y,d) = h.get_ldos()
```

`h.get_ldos()` returns three arrays, the x and y coordinates of the sites and the local
density of states on each of them, at the energy `e` (zero by default). For the zigzag
ribbon what you see at zero energy is that three quarters of the weight sits on the outermost
sites of the two edges: these are the zigzag edge states, and this profile is what one
measures with STM at a zigzag edge of graphene.


Optional arguments
- e: energy at which the LDOS is evaluated
- delta: smearing of the LDOS
- operator: operator to which the LDOS is projected (a name such as `"sz"`, a matrix, or an `Operator`), giving e.g. a spin-resolved real-space map instead of the charge one
- mode: `"arpack"` (default, diagonalization on a k-mesh), `"green"` (Green's function, 2D Hamiltonians only) or `"KPM"` (Chebyshev expansion, for large sparse Hamiltonians, see the local density of states of a large system in the KPM chapter)
- projection: `"TB"` (default, one value per lattice site), `"TBRS"` (same, but interpolated onto a continuous real-space map for smoother plotting), or `"atomic"` (projected onto atomic orbitals rather than tight-binding sites)
- num_bands: for large sparse Hamiltonians, only compute this many states around the target energy

The two evaluation modes weight an operator differently, and both integrate over sites to
the same operator-resolved density of states: `mode="arpack"` uses the expectation value
$\langle \Psi | A | \Psi\rangle$ of each eigenstate times its local density $|\Psi(i)|^2$,
while `mode="green"` uses the local matrix element $\mathrm{Re}\,\Psi^*(i)(A\Psi)(i)$, which
is the genuinely local quantity for states that are not eigenstates of the operator, and
which `mode="KPM"` uses too. A momentum-dependent operator (`"valley"`) only works in
`mode="arpack"`, since `mode="green"` has already integrated over the Brillouin zone.

For a smoother map than one value per site, `projection="TBRS"` interpolates the same
weights onto a continuous real-space grid

```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(x,y,d) = h.get_ldos(e=0.0,projection="TBRS") # interpolated real-space map
```

To follow how the map evolves with energy, `h.get_multildos()` computes the LDOS at many
energies at once

```python
import numpy as np
(x,y,es,ldos) = h.get_multildos(energies=np.linspace(-2.0,2.0,100),projection="atomic") # one map per energy
```

where `x` and `y` are the positions and `ldos` holds one map per energy, one row for each
of the energies `es`, so that the rows are the frames of an LDOS(x,y,E) stack. The same maps
are written one file per energy to a `MULTILDOS/` folder, plus a `MULTILDOS/DOS.OUT` holding
the corresponding total DOS, unless `write=False` is passed

The maps and the `DOS.OUT` beside them carry the same normalization as `h.get_ldos()` and
`h.get_dos()` on the same system, so a map can be read against a single-energy LDOS and the
`DOS.OUT` against `h.get_dos()` without rescaling. An `operator=` is honoured here as it is
by `get_ldos` (the older spelling `op=` still works, but passing both is a `TypeError`), and
the weight it applies is the expectation value $\langle\Psi|A|\Psi\rangle$ of the eigenstate,
a gauge-invariant number, times the local density, the same convention
`get_ldos(mode="arpack")` uses. `projection="atomic"` does not accept an operator and says so

See `examples/0d/island/main.py` (single-energy, `projection="TBRS"`, superconducting island) and `examples/readme_examples/ldos_island/main.py` (`get_multildos`, `projection="atomic"`) for runnable versions.


## Spatially resolved density of states

Let us now look at the local density of states of a finite island, resolved in space and
scanned over a whole set of energies at once, with `h.get_multildos()`. For a triangular
island of the honeycomb lattice, projected onto atomic orbitals rather than onto the
tight-binding sites so that the map is smooth, we write

```python
from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=3,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
h.get_multildos(projection="atomic") # get the LDOS
```

The call writes one map per energy to a `MULTILDOS/` folder, with the density of states of the
whole island next to them in `DOS.OUT`, and the map at a given energy is what one measures with
STM at that bias. The arguments, the projections and the single-energy version are addressed in
the section on the local density of states; see `examples/readme_examples/ldos_island/main.py`
for a runnable version ending in the maps.
