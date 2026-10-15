# Entanglement

## Entanglement entropy and entanglement spectrum

We will now see how to compute the entanglement entropy of a region of a ground state, and
its entanglement spectrum, with `h.get_entanglement_entropy()` and
`h.get_entanglement_spectrum()`. The ground state of any non-interacting or mean-field
Hamiltonian is a Slater determinant. For a Slater determinant the reduced density matrix of a
spatial region $A$ is the exponential of a free-fermion operator, so it is completely fixed
by the one-particle correlations inside $A$. The whole calculation therefore reduces to
diagonalizing the restricted correlation matrix $C_{ij} = \langle c_i^\dagger c_j\rangle$
($i,j \in A$). This gives occupations $\zeta_n \in [0,1]$, from which

$$
S = -\sum_n \left[ \zeta_n \ln \zeta_n + (1-\zeta_n)\ln(1-\zeta_n)\right],
\qquad
\xi_n = \ln\frac{1-\zeta_n}{\zeta_n}
$$

are the entanglement entropy and the single-particle entanglement spectrum. An occupation
exactly at 0 or 1 contributes nothing to the entropy, and an occupation at $1/2$ contributes
the most. This is the correlation-matrix method of Peschel: only a matrix the size of the
region is ever diagonalized, never the exponentially large reduced density matrix.

The region is given as a list of site indices, a boolean mask, a callable on positions (the
same convention as `sculpt`), or simply a fraction of the cells. Spin, sublattice and Nambu
components are treated as extra orbitals of a site, so a region is always specified in terms
of *sites*. A periodic Hamiltonian is first folded into a ring of `nsuper` unit cells, so
region $A$ has **two** entanglement boundaries rather than one. Its entropy is therefore
twice that of a single cut, which matters when comparing with a formula written for one
boundary.

The entanglement spectrum is where this becomes a topological probe. For a 2d Hamiltonian the
momentum parallel to the cut remains a good quantum number, and $\xi_n(k_\parallel)$ is the
Li-Haldane entanglement spectrum. For a Chern insulator its in-gap branches flow across
$\xi=0$ and count $2|C|$ ($|C|$ chiral modes per boundary, two boundaries), mirroring the edge
spectrum of the same model, while the entanglement spectrum of a trivial insulator stays
gapped around $\xi=0$. This is entirely a bulk ground-state calculation: no ribbon and no
open boundary is ever constructed, and the edge states are read off from the bulk alone

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # create a honeycomb lattice
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
h.add_haldane(0.1) # Chern insulator, C = 1
print(h.get_chern(nk=20)) # 1.0

# Li-Haldane entanglement spectrum xi_n(k_par) across the BZ
(ks,xis) = h.get_entanglement_spectrum(nsuper=10,nk=101)

# entanglement entropy, per parallel unit cell, averaged over the BZ
print(h.get_entanglement_entropy(nsuper=10,nk=20))
```

`ks` are the `nk` parallel momenta across the Brillouin zone, and `xis` has one row per
momentum and one column per orbital of the region, ten here (half of a ring of ten cells,
with two sites per cell). Plotting every column against `ks` shows the levels of the Chern
insulator crossing $\xi=0$; the same plot for the honeycomb lattice gapped by a sublattice
imbalance shows a spectrum that never comes close to zero. The entropy is per
parallel unit cell, averaged over `nk` momenta, and counts both boundaries of the region.

Optional arguments

- region: the region $A$, in any of the four forms above; `None` takes half of the system
- nsuper: number of unit cells in the ring that is cut, for a periodic Hamiltonian
- direction: lattice direction the cut is normal to, by default the last periodic one
- kpar: a single parallel momentum; `None` on a 2d Hamiltonian sweeps or averages over `nk` of them
- nk: number of parallel momenta
- fermi: occupied states are those with $E<$ `fermi`, zero by default

Nambu/BdG Hamiltonians are handled with the full anomalous correlation matrix, whose basis
doubling is divided out; `fermi` must stay at zero for them, since their chemical potential
is already inside the Hamiltonian. Occupation is a hard $T=0$ cut. A level sitting exactly at
the Fermi energy makes the ground state degenerate, so the routine raises rather than
silently returning the entropy of an arbitrarily chosen determinant. On a ring this happens
for particular values of `nsuper` (a half-filled chain with `nsuper` divisible by four, say),
and changing the ring size by one is the remedy. The normalization is pinned by the critical
chain, which reproduces the $c=1$ conformal-field-theory law
$S = (c/3)\ln[(L/\pi)\sin(\pi l/L)]$ for a region of $l$ cells in a ring of $L$. A gapped 2d
insulator obeys the area law, and the Li-Haldane counting matches the Chern number. See
`examples/1d/entanglement_entropy_chain/main.py` (the central charge from a fit of the
entropy against the region size) and `examples/2d/entanglement_spectrum_haldane/main.py` (the
spectrum of the Haldane model next to that of a trivial insulator) for runnable versions
ending in the plots, and
`jupyter-notebooks/functionalities/topological_characterization/12_entanglement.ipynb`
for an executed notebook.
