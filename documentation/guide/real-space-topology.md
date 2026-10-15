# Topology in real space

A topological invariant is defined in the Brillouin zone, but its
consequences are local: the bulk carries a quantized Berry flux and the edges
carry the states that cross the gap. In this page we compute the Chern number
from the local marker of a finite sample, also without any lattice, and the
spectral function at the edge of a semi-infinite system.

## Chern number in real space

The local marker $C(\mathbf r)$ of
[Berry curvature density in real space](quantum-geometry.md#berry-curvature-density-in-real-space)
is a Chern number per unit area, so in principle

$$
C = \int C (\mathbf r)\, d^2 \mathbf r ,
$$

with the integral over one unit cell of the bulk. On a finite sample this has
to be read with care. The sum of the marker, times the area of each site,
over the whole island is exactly zero. So summing over all the sites does not
give the invariant. What happens is the following. Deep inside a large enough
island, the surroundings of a site look like the infinite bulk, and there the
marker has a plateau at the bulk Chern number. The sites at the edge carry
the opposite weight, which cancels it. This cancellation is the
bulk-boundary correspondence in real space: the flux carried by the bulk is
given back by the edge states.

```python
from pyqula import islands
from pyqula import topology
import numpy as np
g = islands.get_geometry(name="honeycomb",n=8,nedges=3) # a honeycomb island
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(r,c) = topology.real_space_chern(h) # local marker, per site
r = np.array(r)
bulk = np.argsort(np.linalg.norm(r-r.mean(axis=0),axis=1))[:len(r)//4] # innermost sites
C = np.mean(c[bulk]) # bulk plateau value approximates the total Chern number
```

The marker of each site is divided by the area of the site. This makes it a
density whose value in the bulk is the Chern number. The area of a site is
the area of its Voronoi cell, the region of the plane closer to this site
than to any other. In a crystal this is the area of the unit cell divided by
the number of sites in it. The sites at the edge have open Voronoi cells, and
they take the mean area of the others.

For this island the innermost quarter of the sites gives about $1.95$, close
to the $C=2$ of the spinful Haldane model. The difference of a few percent
comes from the small gap of $t_2=0.05$: with $t_2=0.2$ the innermost sites
give $2.00$. The outermost sites are strongly negative, and the marker times
the area adds up to zero over the whole island.

### An amorphous Chern insulator

The marker also works when there is no lattice at all. We take the amorphous
Chern insulator of Agarwala and Shenoy. The sites are placed at random,
uniformly and without correlations, in a square; this is what
`geometry.amorphous_lattice()` builds. Each site has two orbitals. Every pair
of sites closer than $R=4$ is connected by a hopping that decays
exponentially with the distance and depends on the direction of the bond.
This dependence on the direction makes the model topological in a window of
the mass $M$.

```python
from pyqula import geometry
from pyqula import topology
from pyqula.geometrytk import amorphous
from pyqula.topologytk.realspace import site_areas
import numpy as np
g = geometry.amorphous_lattice(L=36.,density=0.6,seed=0) # 778 sites at random positions
h = amorphous.amorphous_chern_hamiltonian(g,M=-0.5) # Agarwala-Shenoy model, half filling
(r,c) = topology.real_space_chern(h) # local marker, per site
A = site_areas(g) # area of the Voronoi cell of each site
bulk = np.max(np.abs(g.r[:,0:2]),axis=1)<6. # central square, far from the edges
C = np.sum(c[bulk]*A[bulk])/np.sum(A[bulk]) # area-weighted average of the marker
```

The two orbitals of each site play the role of the spin, so for pyqula `h` is
spinful. The Hamiltonian is shifted so that the Fermi energy of half filling
is at zero, which is where `topology.real_space_chern()` puts the Fermi
energy. The marker now changes a lot from site to site, because no two sites
have the same surroundings. And the sites do not have the same area, so the
Chern number of a region is not the plain average of the marker, but its
average weighted by the area of each site,

$$
C = \frac{\sum_{i} C_i A_i}{\sum_i A_i} .
$$

This gives $C\approx-1$. Agarwala and Shenoy find a Bott index of $-1$ at
this density in the window $-2\lesssim M\lesssim1.2$, and the bulk marker
goes to zero on both sides of it: sharply at $M\approx-2$, and slowly at
positive $M$, where it is still about $-1/2$ at $M=1$, because the gap of a
sample of this size closes slowly there.

See
`jupyter-notebooks/functionalities/topological_characterization/05_real_space_chern_amorphous.ipynb`
for the maps of the Chern density and the two transitions.

### Where local markers are useful

The real-space Berry curvature and the Chern marker are examples of a
topological marker: a local quantity, computed from the ground-state
projector only, that gives a bulk invariant without translational symmetry.
So they are the tool for disordered systems, finite flakes, amorphous solids,
or systems where the parameters change in space, for example a Haldane mass
that changes sign across a boundary, or a topological insulator with dilute
vacancies. The map of the marker shows where the invariant is carried, and
where it changes.

## Topological surface states

The invariant predicts states at the edge. The direct way to see them is the
spectral function of a semi-infinite system: a half-plane, periodic along the
edge and infinite away from it. The momentum along the edge is still a good
quantum number, and the states localized at the edge appear as branches that
cross the bulk gap. This is what ARPES sees on a surface, and what STM sees
as in-gap density of states at an edge. For the Haldane model we use
`kdos.surface()`:

```python
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
kdos.surface(h) # surface spectral function
```

The call writes `KDOS.OUT`, with four columns: the momentum along the edge,
the energy, the spectral weight at the surface and the spectral weight in the
bulk. It also returns these four arrays, flat, with one entry per pair of
momentum and energy. By default it uses a hundred energies between $-1$ and
$1$ and a broadening `delta=0.01`. If you plot the surface weight as a
function of momentum and energy, you see the chiral edge states crossing the
gap, one for each unit of the Chern number. They are absent from the bulk
weight: inside the gap the bulk weight is almost zero and the surface weight
is not.

The same calculation is available as `h.get_surface_kdos()`, with the
arguments `energies`, `delta`, `kpath` and `operator`, described in
[Spectral functions](spectral-functions.md), together with the quantum spin
Hall case. See `examples/readme_examples/chern_surface_states/main.py` and
`examples/2d/kdos_edge/main.py` for scripts that make the plot.
