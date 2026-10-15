# Spectral functions and Fermi surfaces

## Momentum resolved spectral functions

Apart from the band structure, in certain cases it is interesting to compute the
momentum-resolved spectral function, with `h.get_kdos_bands()`, that takes the form

$$
A(k,\omega) = \delta(\omega-\epsilon_k) | \langle \Psi_k | A | \Psi_k \rangle|^2
$$

where $A$ is a certain operator. The previous quantity allows defining a heatmap of the
momentum-resolved spectral function, the band structure seen the way ARPES sees it, with
each band drawn as a line of finite width and weighted by the operator. For example, in a
superconducting state, if the operator is chosen to be the projection onto the electron
sector, the previous quantity shows the electronic spectral function, with the hole copy of
the bands removed

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
h.get_kdos_bands()
```

`h.get_kdos_bands()` returns three arrays, the position along the k-path, the energy, and
the spectral weight at each pair, on a window from $-3$ to $3$ by default, and writes the
same three columns to `KDOS_BANDS.OUT`; the broadening `delta` sets the width of each line
in the heatmap.

Optional arguments
- energies: array with the energies for which the spectral function is computed
- delta: smearing of the spectral function
- operator: operator to which the spectral function is projected
- kpath, nk: the k-path and the number of k-points along it, as in `h.get_bands()`
- mode: `"ED"` (default, diagonalization), `"green"` (Green's function) or `"KPM"` (Chebyshev expansion, for large sparse systems)

The three modes compute the same quantity, the spectral weight summed over the orbitals of the
unit cell, so that a system too large to diagonalize at every k-point can be moved from
`mode="ED"` to `mode="KPM"` without changing the scale of the result; what changes is the line
shape, since the Chebyshev expansion broadens each band with its kernel rather than with a
Lorentzian of width `delta`.

See `examples/1d/kdos_armchair/main.py` for a runnable version on a wide ribbon with
`mode="KPM"`, and `jupyter-notebooks/functionalities/single_particle_hamiltonians/05_momentum_resolved_spectral_functions.ipynb`
and `jupyter-notebooks/functionalities/spectral_functions/06_operator_momentum_resolved.ipynb`
for executed notebooks.


## Fermi surfaces

Let us now cut the band structure at a single energy rather than follow it along a path.
The Fermi surface is the set of momenta where a band crosses the Fermi energy, and it is
what a metal looks like in reciprocal space. For a 2D periodic Hamiltonian,
`h.get_fermi_surface()` computes the spectral weight on a $(k_x,k_y)$ mesh at a single
energy (by default the Fermi level, `e=0.0`), i.e. a single constant-energy cut

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian() # get the Hamiltonian
(kx,ky,fs) = h.get_fermi_surface(e=0.0,nk=50,delta=1e-1)
```

It returns the two momentum coordinates of the mesh and the spectral weight at each point,
so a scatter plot colored by the weight is the Fermi surface; the broadening `delta` sets how
thick the contour is drawn, and a value below the level spacing of the mesh shows the mesh
rather than the contour.

Optional arguments
- e: energy of the cut
- nk: number of k-points per direction
- delta: broadening
- operator: project/weight the Fermi surface by an operator, giving e.g. a spin- or valley-textured Fermi surface

An operator weights each point of the Fermi surface by its expectation value, which is how
the spin texture of a spin-orbit coupled material is seen: the NbSe2 Hamiltonian below is a
multi-orbital model with spin-orbit coupling, and coloring its Fermi surface by the spin
along z tells the two spin channels apart on each pocket

```python
from pyqula.specialhamiltonian import NbSe2
h = NbSe2(soc=0.9) # multi-orbital spin-orbit-coupled Hamiltonian
(kx,ky,fs) = h.get_fermi_surface(e=0.,nk=100,delta=3e-1,operator="sz")
```

`h.get_multi_fermi_surface()` computes the same kind of map at many energies at once,
writing one file per energy to a `MULTIFERMISURFACE/` folder, which is the way to scan how
the Fermi surface evolves away from the Fermi level

```python
import numpy as np
h.get_multi_fermi_surface(energies=np.linspace(-4,4,100),delta=1e-1)
```

Passing `operator="unfold"` together with `nsuper` unfolds the Fermi surface of a defective
or disordered supercell back onto the primitive Brillouin zone (see the section on
electronic structure folding and unfolding); as with QPI unfolding below, the supercell must
be built with `store_primal=True`

```python
import numpy as np
from pyqula import geometry
g0 = geometry.triangular_lattice()
n = 3 # size of the supercell
g = g0.get_supercell(n,store_primal=True)
h = g.get_hamiltonian()
h.add_onsite(lambda r: 100.0 if np.linalg.norm(r-g.r[0])<1e-1 else 0.0) # a point defect

out = h.get_multi_fermi_surface(nk=50,energies=np.linspace(-4,4,100),delta=0.1,
        nsuper=n,operator="unfold")
```

Here the mesh is drawn in the reciprocal space of the supercell, and `nsuper=n` stretches it
over the primitive Brillouin zone, which works for a supercell with the shape of the primitive
cell, and not for one whose two lattice vectors have different lengths, where the mesh comes
out sheared. For any supercell, pass `primal_mesh=True` instead: the mesh is then drawn in the
Brillouin zone of the primitive cell, exactly as the Fermi surface of the primitive cell is,
and every point of it is computed at its image in the supercell

```python
import numpy as np
from pyqula import geometry
g0 = geometry.triangular_lattice() # primitive geometry
g = g0.get_supercell(np.sqrt(3),store_primal=True) # sqrt(3) x sqrt(3) supercell
h = g.get_hamiltonian() # Hamiltonian of the supercell
h.add_onsite(lambda r: 0.6 if np.linalg.norm(r-g.r[0])<1e-1 else 0.0) # charge order
(kx,ky,fs) = h.get_fermi_surface(e=0.0,nk=50,delta=0.1,operator="unfold",
        primal_mesh=True) # unfolded Fermi surface, on the primitive mesh
```

The unfolded Fermi surface comes out on the same mesh and in the same coordinates as the one
of the primitive cell, so that the two can be compared point by point, and for a supercell
without any modulation they are the same map, times the number of primitive cells in the
supercell. Note that a supercell asked for by a non-integer size comes out rotated so that its
first lattice vector lies along $x$, and the primitive cell it keeps is rotated with it, so its
Fermi surface is drawn in that rotated frame.

See `examples/readme_examples/fermi_surface/main.py`, `examples/2d/operator_fermi_surface/main.py` and `examples/readme_examples/unfolding_FS/main.py` for runnable versions.


## Surface spectral functions

We will now see how to compute the surface spectral function of a semi-infinite system, meaning
a system that is bulk-like far from a boundary but is cleaved along one direction, with
`h.get_surface_kdos()`.

A ribbon of finite width has two edges and a discrete set of bands that depends on the width,
whereas a semi-infinite system has a single edge and a continuum of bulk states behind it,
which is the situation of a cleaved crystal. The way to obtain it is through the surface
Green's function, computed with a renormalization (decimation) technique for the semi-infinite
bulk: the layers behind the surface are folded into it one after another until adding one more
changes nothing, and the imaginary part of the resulting Green's function, at each momentum
along the edge and each energy, is the surface spectral function, what ARPES measures on the
surface of a crystal. For a honeycomb lattice with Kane-Mele spin-orbit coupling we write

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_soc(0.05) # Kane-Mele spin-orbit coupling opens a topological gap
h.add_rashba(0.1) # break z-mirror symmetry to expose the edge states
(k,e,ds,db) = h.get_surface_kdos(delta=1e-2) # surface and bulk spectral functions
```

`ds` and `db` are, respectively, the surface and bulk spectral weight at each `(k,e)`. The
spin-orbit coupling opens a gap in the bulk, so `db` is empty inside it, and plotting `k,e`
colored by `ds` shows the states that cross that gap: the helical edge states of the quantum
spin Hall state, living at the boundary and absent from the bulk spectrum, which is what makes
them topologically protected. See `examples/readme_examples/surface_2dTI/main.py` for a
runnable version, and
`jupyter-notebooks/functionalities/spectral_functions/02_surface_spectral_semiinfinite.ipynb`
for the executed notebook.

The k-integrated counterpart is `dos.surface_dos(h,...)`, which returns the surface and bulk
densities of states at a set of energies. It takes an `operator=` and projects onto it, giving
for example the spin-resolved surface DOS of a magnetized lead rather than the charge one. A
momentum-dependent operator such as `"valley"` is refused with a `NotImplementedError` naming
it, since the Green's function it is built on has already been integrated over the Brillouin
zone.

This whole family, `dos.surface_dos`, `dos.dos_surface`, `dos.bulkandsurface`,
`dos.surface2bulk` and the surface writers in `kdos`, reports $-\mathrm{Im}\,\mathrm{Tr}\,G$
without the $1/\pi$ that `h.get_dos()` applies, so their values are $\pi$ times a density of
states. They agree with each other; compare them among themselves rather than against
`h.get_dos()`. The same decimation of a semi-infinite lead is what the quantum transport
chapter builds on.
