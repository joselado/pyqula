# Topology in real space

## Chern number in real-space

The local marker of the previous section can be summed into an invariant, and we will now see
how. The Chern number in real-space is defined as 

$$
C = \int F (\mathbf r) d^2 \mathbf r
$$

where $C$ is the total Chern number of the occupied bands and $F (\mathbf r)$
is the spatially-resolved Chern number. Note that this object is meaningful for periodic
systems with very large unit cells. $F(\mathbf r)$ is the local marker computed by
`topology.real_space_chern`, and the integral above has to be read with care on a finite
sample: the sum of the marker over the whole island is exactly zero, so summing over every
site is not how the invariant is recovered. Instead, deep inside a large enough island, where
the local environment already looks like the infinite bulk, the marker plateaus at the
quantized bulk Chern number, and the edge sites carry the opposite weight that cancels it.
This exact real-space cancellation is itself a manifestation of the bulk-boundary
correspondence: the flux carried by the bulk is returned by the edge states

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

The marker of each site is divided by the area the site occupies, which is what makes it a
density that reads the Chern number: the area of a site is the area of its Voronoi cell, the
region of the plane closer to it than to any other site, which in a crystal is the area of the
unit cell divided by the number of sites in it, while the sites at the edge, whose cells are
open, take the mean area of the others. For this island the innermost quarter of the sites
average to about $1.95$, close to the $C=2$ of the spinful Haldane model of the Chern number
section; the missing few percent come from the small gap of $t_2=0.05$, and with $t_2=0.2$ the
innermost sites read $2.00$. The outermost sites are strongly negative, and the marker times the
area of each site adds up to zero over the whole island.

The same marker works where there is no lattice at all, and we will now see it on an amorphous
Chern insulator, the model of Agarwala and Shenoy. The sites are placed at random, uniformly and
without any correlation between them, in a square, which is what `geometry.amorphous_lattice()`
builds, each site hosts two orbitals, and every pair of sites closer than $R=4$ is connected by a
hopping that decays exponentially with their distance and depends on the direction of the bond,
which is what makes the model topological in a window of the mass $M$

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

The two orbitals of each site take the place of the spin, so `h` is spinful as far as pyqula is
concerned, and the Hamiltonian comes shifted so that the Fermi energy of half filling is at zero,
which is where `topology.real_space_chern()` takes the Fermi energy to be. The marker of each site now fluctuates
strongly, since no two sites have the same surroundings, and since the sites do not occupy the
same area the Chern number of a region is not the plain average of the marker but its average
weighted by the area of each site. This gives $C\approx-1$, and the bulk marker goes to zero on
both sides of the window $-2\lesssim M\lesssim1.2$ where Agarwala and Shenoy find a Bott index of
$-1$ at this density: sharply at $M\approx-2$, and gradually at positive $M$, where it is already
about $-1/2$ at $M=1$, since the gap of a sample of this size closes slowly there. See
`jupyter-notebooks/functionalities/topological_characterization/05_real_space_chern_amorphous.ipynb`
for an executed notebook with the maps of the Chern density and the two transitions.

## Topological surface states

The invariant predicts states at the edge, and the direct way to see them is the spectral
function of a semi-infinite system, a half-plane that is periodic along the edge and extends
to infinity away from it, so that the momentum along the edge stays a good quantum number and
the states localized at the boundary show up as branches crossing the bulk gap. We will now
see how to compute it for the Haldane model with `kdos.surface()`

```python
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
kdos.surface(h) # surface spectral function
```

The call writes `KDOS.OUT`, with the momentum along the edge, the energy, the surface spectral
weight and the bulk spectral weight on each line, and returns the same four arrays, flat, one
entry per pair of momentum and energy, on a window of a hundred energies between $-1$ and $1$
with a broadening `delta=0.01` by default. What you see when the surface weight is plotted
against momentum and energy is the chiral edge states crossing the gap, one per unit of the
Chern number, which are absent from the bulk weight: inside the gap the bulk weight is
essentially zero and the surface weight is not. The same calculation is available as
`h.get_surface_kdos()`, with the `energies`, `delta`, `kpath` and `operator` arguments
described in the section on surface spectral functions, which also covers the quantum spin
Hall case. See `examples/readme_examples/chern_surface_states/main.py` and
`examples/2d/kdos_edge/main.py` for runnable versions ending in the plot.

## Topological markers

The real-space Berry curvature and Chern marker of the two sections above are examples of a
topological marker: a local, position-resolved quantity, computable from the ground-state
projector alone, that reveals a bulk topological invariant without relying on translational
symmetry or a clean Brillouin zone. This is what makes topological markers the tool for
disordered systems, finite flakes and islands, amorphous solids, or systems with spatially
varying parameters, a Haldane mass that changes sign across a boundary, say, or a topological
insulator with dilute vacancies, where the marker density directly shows where the invariant
is carried and where it changes. See `topology.real_space_chern` above for the code that
computes it, and the `05_real_space_chern_amorphous.ipynb` notebook for the amorphous Chern
insulator of the section on the Chern number in real-space, a system with no lattice at all.
