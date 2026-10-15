# Wannierization

We will now see how to build a smaller tight-binding model out of a chosen set of bands of a
Hamiltonian, with `h.get_wannier_hamiltonian()`.

A band, or a group of bands, that is separated from the rest by a gap can be described by
localized orbitals centered on the unit cells, the Wannier functions, whose hoppings reproduce
that band subspace. Wannierization is the construction of those functions with the smallest
possible spread, and what it returns is a new tight-binding Hamiltonian with as many orbitals
per cell as bands were selected, that can be used like any other Hamiltonian in this guide.
`h.get_wannier_hamiltonian()` Wannierizes a contiguous range of a periodic Hamiltonian's bands
and returns a new, smaller multicell Hamiltonian whose real-space hoppings reproduce that band
subspace on the wannierization k-mesh. By default the range is taken as a fixed subspace,
Wannierized jointly as one group, and reproduced exactly; passing a `num_wann` smaller than
the range instead disentangles, which is the subject of the second subsection below.

As an example, consider a staggered honeycomb lattice, where a sublattice potential opens a
gap and gives a genuinely dispersive valence band to Wannierize

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_onsite([0.8,-0.8]) # sublattice potential opens a gap

# Wannierize just the lowest (valence) band: bands=[0,0] selects band
# index 0 at every k-point of a 12x12 Monkhorst-Pack wannierization mesh
hwan = h.get_wannier_hamiltonian(bands=[0,0],nk=12)

print("Number of Wannier functions:",hwan.intra.shape[0])
print("Wannier centres (Cartesian):\n",hwan.wannier_centres)
print("Wannier spreads:",hwan.wannier_spreads)
print("Total spread Omega:",hwan.wannier_spread_total)
```

One band was selected, so the Wannierized model has a single orbital per cell, and its centre
and spread are the ones the minimization arrived at: the total spread $\Omega$ is what a
maximally localized Wannier function has made as small as it can. The minimization starts from
the orbitals of the original model where the selected bands have most of their weight, here the
site with the lower onsite energy, so that the same call always gives the same Wannier function,
and `trial_vectors=` replaces that starting point by one of your choice. The Wannierized Hamiltonian
`hwan` behaves like any other pyqula Hamiltonian, so its bands can be compared directly
against the original model's

```python
(k,e) = h.get_bands(write=False)
(kw,ew) = hwan.get_bands(write=False)
```

The two coincide on the wannierization mesh, and the Wannier model interpolates smoothly
between the mesh points, which is the point of doing this: a one-orbital model that carries the
valence band of the two-orbital one. The mesh fixes each hopping only up to a translation by the
supercell of `nk` cells along each direction, so, as in Wannier90, the hoppings are placed on the
cells of the Wigner-Seitz cell of that supercell, the ones closest to the origin, and a cell on its
boundary carries the hopping divided by the number of equally close copies it has; this set of
cells is symmetric under $R\to-R$, so the Bloch Hamiltonian of the Wannier model is Hermitian at
every k and not only on the mesh. How good the interpolation is between the mesh points depends on
how localized the Wannier functions are, so a band that comes close to another somewhere in the
Brillouin zone needs a denser mesh. See `examples/wannier/get_wannier_hamiltonian/main.py` for
a runnable version, and
`jupyter-notebooks/functionalities/wannierization/01_maximally_localized_wannier.ipynb` and
`02_exact_reproduction.ipynb` in the same folder for the executed notebooks.

## Symmetry-enforced Wannierization

Let us now see how the point-group symmetry of the lattice is enforced on the Wannier
functions, with `symmetries="auto"`. Passing it makes `get_wannier_hamiltonian` check, before
Wannierizing, that the selected band range is a genuine union of point-group-related multiplets
everywhere on the mesh, with the point group auto-detected from the geometry and the
Hamiltonian. A band selection that instead slices through a symmetry-related degeneracy is
rejected with a `ValueError` rather than silently returning a mis-symmetrized model. A list of
explicit `symmetrytk.pointgroup.SymmetryOperation` can be passed instead of `"auto"` to
enforce a specific subgroup.

A good illustration is kagome's flat band, the lowest of its three bands (index 0, at
$E=-2$ in units of the hopping): it touches the dispersive middle band at the $\Gamma$ point, so
no selection containing only the flat band is a union of whole multiplets there. This touching
is why kagome's flat band on its own, which is not an isolated band, has no exponentially
localized Wannier function, symmetric or not, and the check catches it instead of returning a
broken model

```python
from pyqula import geometry
from pyqula.symmetrytk import pointgroup

g = geometry.kagome_lattice()
h = g.get_hamiltonian(has_spin=False)

try:
    h.get_wannier_hamiltonian(bands=[0,0],nk=12,symmetries="auto") # the flat band alone
except ValueError as e:
    print("Flat band alone correctly rejected:",str(e).splitlines()[0])

# the full 3-band manifold has no such obstruction
hwan_sym = h.get_wannier_hamiltonian(bands=[0,2],nk=12,symmetries="auto")
print("Symmetries enforced:",[c.op.name for c in hwan_sym.wannier_symmetries])
```

The first call raises, and the message names the symmetry and the k-point, $\Gamma$, where the
selection is not a union of whole multiplets; the second returns a
three-orbital model whose Wannier functions transform into each other under the operations
listed in `wannier_symmetries`. See `examples/wannier/get_wannier_hamiltonian/main.py` and
`examples/wannier/symmetric_wannierization/main.py` for runnable versions of these two
examples, and
`jupyter-notebooks/functionalities/wannierization/03_symmetry_enforced_wannierization.ipynb`
for the executed notebook.

## Disentanglement

A fixed band range only works when the range is separated from everything else by a gap across
the whole Brillouin zone. Let us now see what to do when it is not, with `num_wann`.
Graphene's two $p_z$ bands are the simplest counterexample: they touch at K, so neither of
them is a smooth subspace on its own. Souza-Marzari-Vanderbilt disentanglement (Phys. Rev. B
65, 035109) handles that case by extracting `num_wann` optimally connected states out of a
larger set of *offered* bands, k-point by k-point. It is switched on by passing a `num_wann`
smaller than the selected range, in which case `bands=[a,b]` only says which bands are offered.

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)

# one Wannier function out of both pz bands, with the deep part of the
# valence band frozen (kept exactly rather than optimized)
hwan = h.get_wannier_hamiltonian(bands=[0,1],num_wann=1,nk=12,
                                 dis_froz_max=-1.0,cutoff=0.0)

print("Wannier functions:",hwan.wannier_num_wann)
print("window used:",hwan.wannier_disentanglement_window)

f0,fw = h.get_hk_gen(),hwan.get_hk_gen()
k = [2./12.,1./12.,0.] # a wannierization-mesh k-point
e0 = np.sort(np.linalg.eigvalsh(f0(k))) ; ew = np.linalg.eigvalsh(fw(k))
print("frozen state reproduced to:",abs(ew[0]-e0[0]))
```

What is reproduced changes with it, and this is the trade rather than a loss of accuracy: a
disentangled Hamiltonian reproduces the states inside the **frozen inner window**
(`dis_froz_min`/`dis_froz_max`) exactly at every mesh k-point, and deliberately does not
reproduce the selected bands outside it. Outside the frozen window the extracted subspace is a
different, smoother one, meaning that the last line above prints a number at machine precision
for the frozen valence state while the conduction band is not in the model at all. Pass
`cutoff=0.0` when testing that exactness, as above: the default `cutoff=1e-6` drops small
real-space hoppings and with them the last few digits. An **outer window**
(`dis_win_min`/`dis_win_max`) narrows which bands are offered at each k-point, so the number
available varies across the mesh, which is the point of a window rather than a band range;
every eigenvalue of the result then lies inside it. `dis_num_iter` (default 200) bounds the
$\Omega_I$ minimization; not reaching convergence within it is a warning rather than an
error, since the frozen window is reproduced either way.

The returned Hamiltonian carries `wannier_num_wann` and `wannier_disentanglement_window` (the
four window values actually used, or `None` when not disentangling) alongside the centres and
spreads.

Combinations that would silently do nothing raise instead: a `dis_*` window without a smaller
`num_wann` (disentanglement is switched on by the band count exceeding `num_wann`, so the
window would be accepted and ignored), `dis_froz_min` without `dis_froz_max`, or a `num_wann`
outside `1..len(bands)`. Disentanglement is not implemented together with a Nambu/BdG
Hamiltonian, with `symmetries=`, or with `auto_split_clusters=True`, each naming the
combination. One gap in that guard is worth knowing about: `win_keywords=` is by design an
unchecked passthrough merged after the check, so a window smuggled in through it can still
reach the engine and be ignored; use the real `dis_froz_max=` argument, which is guarded. See
`jupyter-notebooks/functionalities/wannierization/04_dimensionality_and_bdg.ipynb` for the
executed notebook on Wannierizing in other dimensionalities and in the Nambu basis, and
`jupyter-notebooks/functionalities/wannierization/05_band_disentanglement.ipynb` for the one on
disentanglement.
