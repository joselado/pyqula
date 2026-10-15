# Electronic structure folding and unfolding

We will now see how the band structure of a supercell folds into its smaller Brillouin zone,
and, what matters most, how it can be unfolded back onto the Brillouin zone of the primitive
cell, with `operator="unfold"`.

Building a supercell does not change the electronic structure, but it does change how it is
drawn: the Brillouin zone shrinks, and the bands of the primitive cell fold into it, so that a
supercell of $n\times n$ primitive cells shows $n^2$ times more bands, all of them replicas
of the same dispersion. The inverse operation, unfolding, recovers the primitive-cell spectral
weight of a supercell calculation, and this is what is needed when the supercell is there for
a reason (a defect, a moire pattern, or a non-primitive choice of cell) and the result is to
be compared against a measured band structure, since ARPES sees the spectral weight of the
primitive cell and only a faint trace of the folded replicas when the perturbation is weak.
Unfolding is implemented as a special operator, `"unfold"`, that projects onto the Bloch
states of the primitive cell; it can be passed to any of the k-resolved observables
(`h.get_bands()`, `h.get_kdos_bands()`, `h.get_multi_fermi_surface()`...), and it requires
the supercell to have been built keeping track of the primitive geometry

```python
from pyqula import geometry
import numpy as np
g = geometry.honeycomb_lattice() # primitive geometry
n = 3
gs = g.get_supercell(n,store_primal=True) # supercell, keeping the primitive cell info
h = gs.get_hamiltonian() # Hamiltonian of the supercell
kpath = gs.get_unfolded_kpath(nk=200) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_kdos_bands(operator="unfold",delta=1e-1,kpath=kpath) # unfolded weight
```

`gs.get_unfolded_kpath()` gives the k-path of the primitive cell written in the coordinates of
the supercell, which is the path the unfolded bands have to be drawn along, and the section
"Which k-path to unfold along" below says what it does; `d` holds the unfolded spectral weight
at each `(k,e)`, so that plotting a scatter of `k,e` colored or sized by `d` recovers the
primitive-cell band structure out of the supercell calculation: the replicas that a plain
`h.get_bands()` of the supercell would show carry no weight, and the Dirac cones of the
honeycomb lattice reappear where they belong. The same `operator="unfold"` can be passed to
`h.get_multi_fermi_surface()` to unfold constant-energy cuts, with `primal_mesh=True` to draw
them on the mesh of the primitive cell (see the section on Fermi surfaces). See
`examples/2d/unfolding/main.py`, `examples/1d/unfolding/main.py` and
`examples/readme_examples/unfolding_FS/main.py` for runnable versions, and
`jupyter-notebooks/functionalities/single_particle_hamiltonians/08_unfolding_supercells.ipynb`
for the executed notebook.

Unfolding also works when atoms have been removed from the supercell (for example
`gs = gs.remove([...])` before `gs.get_hamiltonian()`), such as a vacancy or an
irregularly-shaped flake cut out of a supercell: each remaining atom is matched back to its
primitive-cell replica by position, so no extra arguments are needed. The match requires the
remaining atoms to sit exactly where they were in the complete supercell, so do not move them
(with `gs.center()`, or a relaxation) between `gs.remove(...)` and `gs.get_hamiltonian()`.
Doing so raises a `ValueError` rather than unfolding onto the wrong replica.

Unfolding also works for a general, non-diagonal or non-orthogonal supercell, built by passing
a 3x3 integer matrix `M` to `get_supercell` instead of a plain `(n1,n2,...)` size
(`gs.a1,gs.a2,gs.a3` become integer combinations of the primitive vectors, `gs = M @ g`). No
change is needed at the unfolding call site: `get_supercell(M,...)` records, per surviving
atom, which primitive replica it came from, and `operator="unfold"` reads that bookkeeping
directly, both for a complete supercell and after removing atoms

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # primitive geometry
M = [[2,1,0],[0,1,0],[0,0,1]] # non-diagonal supercell matrix, det(M)=2
gs = g.get_supercell(M,store_primal=True) # supercell, keeping the primitive cell info
h = gs.get_hamiltonian() # Hamiltonian of the supercell
kpath = gs.get_unfolded_kpath(nk=200) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_kdos_bands(operator="unfold",delta=1e-1,kpath=kpath) # unfolded weight
```

The projector that the unfolding operator builds is a sum of Bloch phases over the replicas
of the primitive cell, and nothing in it refers to the dimensionality, so all of this works
the same way for a chain, for a two-dimensional lattice and for a bulk three-dimensional
crystal, with a plain size or with a 3x3 `M`

```python
from pyqula import geometry
g = geometry.cubic_lattice() # three dimensional primitive geometry
gs = g.get_supercell([2,2,2],store_primal=True) # supercell, keeping the primitive cell info
h = gs.get_hamiltonian() # Hamiltonian of the supercell
kpath = gs.get_unfolded_kpath(nk=200) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_bands(operator="unfold",kpath=kpath) # unfolded band structure
```

See `examples/2d/unfolding_nonorthogonal/main.py` for a runnable version with a defect in the
non-diagonal supercell, and `examples/3d/unfolding/main.py` for the three-dimensional one.

A cell that is not a multiple of the primitive one along each lattice vector can also be asked
for by its size alone, and the case that comes up most often is the $\sqrt3\times\sqrt3$
supercell of a triangular or a honeycomb lattice, the cell of a three-sublattice charge order

```python
from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # primitive geometry
gs = g.get_supercell(np.sqrt(3),store_primal=True) # sqrt(3) x sqrt(3) supercell
```

A non-integer size is read as a request for a cell larger by that factor in area, three
primitive cells here, and `get_supercell` returns the most compact cell of that size that can
be built out of integer combinations of $\vec a_1$ and $\vec a_2$, which for the triangular
lattice is the cell with $|\vec A_1|=|\vec A_2|=\sqrt3|\vec a_1|$ at 60 degrees from each
other, rotated so that $\vec A_1$ points along $x$. Several integer matrices of determinant
three describe that same cell, `gs.supercell_matrix` says which one was taken, and passing an
explicit `M` is the way to fix both the matrix and the orientation rather than leaving them
to the search. Compactness is what makes the cell usable and not just correct: a cell of the
right volume but with a small angle between its lattice vectors puts the first neighbors of
an atom outside the cells the hopping generator searches, and the Hamiltonian comes out with
hoppings missing.

## Which k-path to unfold along

The unfolded band structure has to be drawn along the Brillouin zone of the primitive cell,
and this is where a general supercell differs from an $n\times n$ one. The unfolding operator
sends a momentum $\vec k$ of the supercell onto the momentum $M^{-1}\vec k$ of the primitive
cell, so tracing the primitive Brillouin zone means feeding it $M\vec k_0$ for every point
$\vec k_0$ of the primitive path. For a diagonal supercell that is the primitive path times
$n$, which is how `examples/2d/unfolding/main.py` and the other diagonal examples write it by
hand, and for a $\sqrt3\times\sqrt3$ cell it is a rotation as well as a rescaling, which no
multiplication by a number can give. `gs.get_unfolded_kpath()` returns it for any supercell,
diagonal or not

```python
from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # primitive geometry
gs = g.get_supercell(np.sqrt(3),store_primal=True) # sqrt(3) x sqrt(3) supercell
h = gs.get_hamiltonian() # Hamiltonian of the supercell
fons = lambda r: (np.sum((r - gs.r[0])**2)<1e-2)*0.6 # onsite in one of the three sites
h.add_onsite(fons) # three-sublattice charge order
kpath = gs.get_unfolded_kpath(nk=200) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_kdos_bands(operator="unfold",delta=1e-1,kpath=kpath) # unfolded weight
```

The k-points come back in the reduced coordinates of the supercell, which is what
`h.get_bands()` and `h.get_kdos_bands()` take, so they go straight into `kpath=`, and the
method takes the same arguments as `g.get_kpath()`, meaning that the high-symmetry points can
be named as well, `gs.get_unfolded_kpath(["G","K","M","G"],nk=400)`, and they are the points of
the primitive cell and not of the supercell. Plotting `k,e` colored by `d` gives the single
band of the triangular lattice along $\Gamma$-K-M-K'-$\Gamma$, with a gap opened at K by the
charge order and a faint shadow band where the folded replica carries the little weight that
the modulation gives it. Along the supercell k-path instead, `gs.get_kpath()`, the same
calculation traces a path through the primitive Brillouin zone that goes nowhere in
particular, and this is the reason for asking for the primitive one. See
`examples/2d/unfolding_sqrt3/main.py` for a runnable version.

## Unfolding a large supercell with the kernel polynomial method

We will now see how to unfold a supercell too large to diagonalize at every k-point, with
`h.get_kdos_bands(operator="unfold",mode="KPM")`. The cases where unfolding matters most, a
moire pattern or a disordered region, are the ones with the largest supercells, and there the
diagonalization that `mode="ED"` performs at each k-point is what limits the calculation. The
unfolding operator projects onto the Bloch states of the primitive cell, one for each orbital
$\alpha$ of the primitive cell, so the unfolded weight is the density of states of those Bloch
states

$$
A(\vec k,\omega) = \sum_\alpha \langle \vec k,\alpha|\delta(\omega-H)|\vec k,\alpha\rangle
$$

where $|\vec k,\alpha\rangle$ is the Bloch state built out of orbital $\alpha$ over all its
replicas in the supercell and $H$ is the Bloch Hamiltonian of the supercell at the momentum
$\vec k$ maps to. Each term is a single Chebyshev expansion of the kernel polynomial method,
started from the Bloch state itself, meaning that no random vectors are needed and the result
carries no stochastic noise

```python
import numpy as np
from pyqula import geometry
from pyqula import potentials
g0 = geometry.triangular_lattice() # primitive geometry
g = g0.get_supercell(12,store_primal=True) # supercell of 144 primitive cells
h = g.get_hamiltonian() # Hamiltonian of the supercell
fmoire = potentials.commensurate_potential(g,n=3,minmax=[0,1]) # moire potential
h.add_onsite(fmoire) # onsite energy following the moire
kpath = g.get_unfolded_kpath(nk=100) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_kdos_bands(operator="unfold",mode="KPM",delta=5e-2,kpath=kpath,
        energies=np.linspace(-3,-1,200)) # unfolded weight, with the KPM
```

The arrays are the ones `mode="ED"` returns, and so is the weight under each band, since the
two modes compute the same unfolded spectral weight; what differs is the line shape, as the
Chebyshev expansion broadens each band with its kernel rather than with a Lorentzian, using a
number of polynomials set by `delta`. The cost of each k-point grows linearly with the number
of sites of the supercell instead of with its cube, which is what makes supercells of
thousands of sites affordable, and a smaller `delta` costs proportionally more polynomials.
