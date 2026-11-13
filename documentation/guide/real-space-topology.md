# Topology in real space

A topological invariant is defined in the Brillouin zone, but its
consequences are local: the bulk carries a quantized Berry flux and the edges
carry the states that cross the gap. In this page we compute the Chern number
from the local marker of a finite sample, also without any lattice, the Bott
index of a disordered sample, and the spectral function at the edge of a
semi-infinite system.

## Chern number in real space

The local marker $C(\mathbf r)$ of
[Berry curvature density in real space](quantum-geometry.md#berry-curvature-density-in-real-space)
is a number without units: the marker of a site divided by the area of the
site. So in principle

$$
C = \frac{1}{A_{\rm cell}} \int_{\rm cell} C (\mathbf r)\, d^2 \mathbf r
  = \frac{\sum_{i} C_i A_i}{\sum_i A_i} ,
$$

the average of the marker over one unit cell of the bulk, with $C_i$ the
marker and $A_i$ the area of the sites $i$ of the cell. In a crystal every
site of the cell has the same area, so this is the plain average over the
sites of the cell, and every site of the bulk gives about $C$. On a finite sample this has
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
number without units, whose value in the bulk is the Chern number. The area of a site is
the area of its Voronoi cell, the region of the plane closer to this site
than to any other. In a crystal this is the area of the unit cell divided by
the number of sites in it. The sites at the edge have open Voronoi cells, and
they take the mean area of the others.

For this island the innermost quarter of the sites gives about $1.95$, close
to the $C=2$ of the spinful Haldane model. The difference of a few percent
comes from the small gap of $t_2=0.05$. A small gap means a long decay length
of the edge into the bulk, and the island with `n=8` is not much larger than
it, so its innermost quarter still feels the edge. A larger gap or a larger
island removes the difference: with $t_2=0.2$ the innermost sites give
$2.00$, and with $t_2=0.05$ the islands with `n=12` and `n=16` give $1.985$
and $1.995$. The outermost sites are strongly negative, and the marker times
the area adds up to zero over the whole island.

![Local Chern marker of each site against its distance to the center of the island, and its sum inside a radius R.](img/realspace_marker.png#only-light)
![Local Chern marker of each site against its distance to the center of the island, and its sum inside a radius R.](img/realspace_marker-dark.png#only-dark)

The figure uses the larger island, `n=16`, with 2301 sites, which takes about
two minutes. On the left is the marker of each site against its distance to
the center of the island. The innermost quarter of the sites, in orange, sits
on the plateau, with mean $1.995$. Further out the marker spreads, between
about $0.7$ and $2.3$, and the sites at the edge go down to $-38$. On the right
we add up the marker times the area of each site, for the sites closer than
$R$ to the center. In the bulk every site gives about $2$ times its area, so
the sum follows $2$ times the area inside $R$ (dashed line). It grows up to
$R\approx23.5$, the distance from the center to the middle of an edge of the
triangle, where it is $3371$, against $3440$ for $2$ times the area. Then the
edge sites come in, with their large negative marker, and the sum goes down
to zero, within $10^{-11}$, at the last site. This zero is exact for any
finite island: the marker of a site is a diagonal element of the commutator
of $PxP$ and $PyP$, with $P$ the projector on the occupied states, and the
sum of the diagonal elements, the trace, of a commutator of two finite
matrices is zero. So the edge always gives back what the bulk carries.

??? example "Code of this figure (documentation/figures/realspace_marker.py)"

    ```python
    --8<-- "realspace_marker.py"
    ```

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

![Local Chern marker at each site of the amorphous Chern insulator, with the central square drawn.](img/realspace_amorphous.png#only-light)
![Local Chern marker at each site of the amorphous Chern insulator, with the central square drawn.](img/realspace_amorphous-dark.png#only-dark)

The figure shows the marker at each site of the snippet, with the central
square of the average drawn in gray. Inside the sample the marker is
negative at almost every site, around $-1$, but it changes a lot from site to
site. The area-weighted average over the central square is $-1.03$. The
sites at the edge of the sample are strongly positive, and they compensate
the bulk, as on the island above.

??? example "Code of this figure (documentation/figures/realspace_amorphous.py)"

    ```python
    --8<-- "realspace_amorphous.py"
    ```

See
`jupyter-notebooks/functionalities/topological_characterization/05_real_space_chern_amorphous.ipynb`
for the maps of the Chern density and the two transitions.

### The marker of a large flake

The marker above needs the projector on the occupied states as a dense
matrix. The cost grows as the cube of the number of sites, so a flake of
$10^4$ sites is already very slow. We will now see how to compute the
marker at a cost linear in the number of sites.

The idea is that the marker is local. In an insulator the projector
$P_{ij}$ decays exponentially with the distance between the sites $i$ and
$j$. So the marker of a site only needs the projector between the sites
close to it. We take the sites within $R$ hops of the Hamiltonian around
the site $a$, the ball of $a$, and compute

$$
C_a = \frac{2\pi}{A_a}\sum_{\alpha\in a}{\rm Im}\,\left[P X_a P,\, P Y_a P\right]_{\alpha\alpha} ,
\qquad X_a = X - x_a ,\quad Y_a = Y - y_a ,
$$

with every matrix restricted to the ball of $a$, and $A_a$ the area of the
site. This is the marker of `topology.real_space_chern()`, with the
positions measured from the site $a$. For a true projector the origin of
the positions does not matter. Here the entries of $P$ come from a
Chebyshev expansion of the step function $\theta(-H)$, as in
[Kernel polynomial method](kpm.md), and the expanded $P$ is not exactly a
projector. With a fixed origin, its error would be multiplied by the
distance to that origin. With every site as its own origin, the error stays
local. Only the entries of $P$ between sites at most $2R$ hops apart are
computed, so the cost is the number of sites times a constant set by $R$.
This is what `h.get_kpm_chern_marker()` computes.

```python
from pyqula import islands
from pyqula import topology
import numpy as np
g = islands.get_geometry(name="honeycomb",n=12,nedges=4,rot=0.0,clean=False) # 984 sites
h = g.get_hamiltonian(has_spin=False) # spinless honeycomb flake
h.add_haldane(0.1) # Haldane coupling, gap of about 1
(r,c) = h.get_kpm_chern_marker(radius=6,npol=200) # local marker, per site
(r,c0) = topology.real_space_chern(h) # the dense marker, for comparison
r = np.array(r)
bulk = np.sum(r**2,axis=1) < np.max(np.sum(r**2,axis=1))/16 # inner sites
```

In the bulk the two markers agree. For the 84 sites within a quarter of
the radius of this flake, the dense marker gives $0.9997$ and the KPM one
$0.992$. The difference comes from the ball, and it falls by about a factor
$2.5$ for every hop: the largest difference in the bulk is $0.30$, $0.049$,
$0.008$ and $0.001$ for $R=2$, $4$, $6$ and $8$. A smaller gap means a
projector that decays more slowly, and so a larger radius for the same
accuracy. With $t_2\to-t_2$ every value changes sign, and with a large
sublattice mass, $M=0.9$, the bulk marker goes to zero. The same marker is
`topology.real_space_chern(h,mode="kpm",radius=6)`, which takes the same
keywords.

The price is the sum over the flake. The dense marker adds up to zero, the
edge giving back what the bulk carries. The edge states are gapless, so the
projector decays slowly along the edge, and a ball of finite radius does
not see this compensation. For this flake the sum of the local marker is
about $0.85$ times the number of sites, against zero for the dense one. So
this marker is for the bulk, and for the regions where the gap is open, not
for the edge. In `examples/2d/kpm_chern_marker/main.py` a flake of 6180
sites has a Haldane insulator in its left half and a large sublattice mass
in its right half. Away from the edge and from the boundary between the
halves, the marker is $0.992$ on the left and $0.004$ on the right, so the
map shows where the invariant is carried.

The keywords are:
- radius=6: the sites around each site that enter its marker, in hops of
  the Hamiltonian. The bulk error falls exponentially with it.
- npol=200: the number of Chebyshev polynomials of the projector. The step
  of the expanded $\theta(-H)$ has a width of roughly $\pi s/2n_{\rm pol}$,
  with $s$ the bound of the spectrum, and it has to be much smaller than the
  gap.
- kpm_radius=None: the hops around each orbital kept in the Chebyshev
  recursion, equal to `radius` by default.
- operator=None: weight the marker with an operator, as for
  `topology.real_space_chern()`, for example `h.get_operator("sz")` for a
  spinful flake. The two agree when the operator commutes with the
  Hamiltonian. When it does not, as $s_z$ with Rashba coupling, the dense
  marker depends on the origin of the positions, and this one takes each
  site as its own origin.
- scale=None: the bound $s$ of the spectrum, estimated if None. A given $s$
  has to exceed the spectral radius by more than 1%, because the occupied
  window stops at $-0.99s$.

It raises `ValueError` if the Hamiltonian is not zero dimensional, or if a
given `scale` is within 1% of the spectral radius.

For a large flake, build the Hamiltonian sparse, with
`g.get_hamiltonian(is_sparse=True)`. The default is a dense matrix, which
for $4\times10^4$ sites already takes 24 GB before any marker is computed.

### Where local markers are useful

The real-space Berry curvature and the Chern marker are examples of a
topological marker: a local quantity, computed from the ground-state
projector only, that gives a bulk invariant without translational symmetry.
So they are the tool for disordered systems, finite flakes, amorphous solids,
or systems where the parameters change in space, for example a Haldane mass
that changes sign across a boundary, or a topological insulator with dilute
vacancies. The map of the marker shows where the invariant is carried, and
where it changes.

## Bott index of a disordered sample

The local marker gives the invariant only as an average over the bulk. The
Bott index gives it as one integer for the whole sample. It is equal to the
Chern number, and it needs no translational symmetry. The sample is a
periodic supercell of $N$ sites, closed into a torus, so it has no edge. We
take $P$, the projector on the states below zero energy, and the
fractional coordinates $X$ and $Y$ of the orbitals along the two lattice
vectors of the supercell, between $0$ and $1$. On a torus $X$ is not a
well-defined operator, but $e^{2\pi i X}$ is. So we define

$$
U = P e^{2\pi i X} P, \qquad V = P e^{2\pi i Y} P,
\qquad
B = \frac{1}{2\pi} \,{\rm Im}\, {\rm Tr} \log \left( V U V^\dagger U^\dagger \right) .
$$

In a trivial insulator the projected positions almost commute, $VUV^\dagger
U^\dagger$ is close to the identity, and $B=0$. In a Chern insulator they
cannot commute, and the phases of the eigenvalues of $VUV^\dagger U^\dagger$
add up to $2\pi C$. Before the trace $U$ and $V$ are replaced by their
unitary part, which keeps $B$ an integer. The difference with the Chern
number of the k-space routines is of order $1/L$, with $L$ the size of the
supercell, so in practice they agree as soon as the supercell is larger
than the localization length.

```python
from pyqula import geometry
import numpy as np
g = geometry.honeycomb_lattice() # honeycomb lattice
h = g.get_hamiltonian(has_spin=False) # spinless Hamiltonian
h.add_haldane(0.15) # Haldane coupling, C=1
h = h.get_supercell(12) # periodic supercell of 288 sites
rng = np.random.default_rng(0) # random numbers
W = 3.0 # strength of the disorder
h.add_onsite(W*(rng.random(len(h.geometry.r))-0.5)) # Anderson disorder
B = h.get_bott_index() # Bott index
```

This gives $B=1$, and the same value for any other random sample at this
disorder: the disorder is about twice the clean gap of $1.35$, but the
states at the Fermi energy are localized, so the invariant survives. The
sign follows the order of the two lattice vectors, as `h.get_chern()` does,
so for a clean crystal $B$ is exactly the value of `h.get_chern()`.

With time reversal $B=0$, and the invariant is the spin Bott index. The
occupied states are split by the sign of their projected spin, $P s_z P$,
into $P_+$ and $P_-$, and

$$
B_s = \frac{1}{2} \left( B_+ - B_- \right),
$$

with $B_\pm$ the Bott index of $P_\pm$. It is the spin Chern number of a
disordered sample. Its parity is the $Z_2$ invariant, so $B_s=1$ in a
quantum spin Hall insulator. The spin $s_z$ does not have to be conserved.
What matters is that the spectrum of $P s_z P$ keeps a gap around zero.
`h.get_spin_bott_index()` raises only when an eigenvalue of $P s_z P$ is at
zero, so a $B_s$ that is not an integer means that the two halves are not
separated.

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # honeycomb lattice
h = g.get_hamiltonian(has_spin=True) # spinful Hamiltonian
h.add_kane_mele(0.1) # Kane-Mele spin-orbit coupling
h.add_rashba(0.05) # Rashba coupling, s_z is not conserved
Bs = h.get_supercell(8).get_spin_bott_index() # spin Bott index
```

This gives $B_s=1$, with $B=0$, and adding a sublattice imbalance of $0.8$
gives $B_s=0$, the trivial insulator.

![Bott index of the Haldane model and spin Bott index of the Kane-Mele model with Rashba coupling, against the strength of the disorder.](img/realspace_bott.png#only-light)
![Bott index of the Haldane model and spin Bott index of the Kane-Mele model with Rashba coupling, against the strength of the disorder.](img/realspace_bott-dark.png#only-dark)

The figure shows the two indices of twenty disordered samples of 288 sites
for each disorder strength $W$, in gray, and their average, in color. Every
single sample gives an integer. For weak disorder all the samples give the
clean value, $B=1$ on the left and $B_s=1$ on the right. Between $W=4.5$
and $W=6.5$ on the left, and between $W=4$ and $W=6$ on the right, more and
more samples give $0$, and above that all of them do: the disorder localizes
every state and the sample becomes an Anderson insulator. The transition is not sharp because the samples are
finite. With $8\times8$ supercells and six samples the average moves by up
to $0.2$ inside the window, and not at all outside it, so the plateaus are
converged and the exact position of the transition is not.

??? example "Code of this figure (documentation/figures/realspace_bott.py)"

    ```python
    --8<-- "realspace_bott.py"
    ```

The keywords of both methods:
- tol=1e-8: the smallest distance to zero of an energy level, and for the
  spin Bott index of an eigenvalue of $P s_z P$, that is accepted
- operator="sz": for `h.get_spin_bott_index()`, the spin component that
  splits the occupied states

The Hamiltonian has to be two dimensional, and the Fermi energy at zero; a
level exactly at zero, as at the Dirac points of the clean honeycomb
lattice, raises an error.

## Multipole chiral number of a higher-order insulator

We will now see an integer that counts the zero-energy states at the corners
of a two-dimensional insulator with chiral symmetry. Chiral symmetry means
that the states of every unit cell split into two sublattices A and B, and
that the Hamiltonian only connects A with B. In the basis (A,B) the
Hamiltonian is

$$
H = \begin{pmatrix} 0 & h \\ h^\dagger & 0 \end{pmatrix} .
$$

The singular value decomposition $h = U_A \Sigma U_B^\dagger$ pairs every
state of A with one state of B, the two halves of the levels $\pm\Sigma$.
The sample is a supercell of $L_x\times L_y$ unit cells, and $x$ and $y$
count its unit cells from one corner, from $0$ to $L-1$. The sublattice
quadrupole operator $Q_s$, with $s={\rm A,B}$, is diagonal, and its entry
for a state of the sublattice $s$ in the unit cell $(x,y)$ is the phase
$e^{-2\pi i xy/L_xL_y}$. With the projected operators
$\bar Q_s = U_s^\dagger Q_s U_s$, the multipole chiral number is

$$
N_{xy} = \frac{1}{2\pi i}\, {\rm Tr}\, \log \left( \bar Q_A \bar Q_B^\dagger \right) .
$$

Every unit cell has as many states on A as on B, so the determinant of
$\bar Q_A \bar Q_B^\dagger$ is exactly one, and the phases of its eigenvalues
add up to $2\pi N_{xy}$, meaning that $N_{xy}$ is an integer for any sample.
In a trivial insulator the states of A and of B see the quadrupole phase in
the same way, and $N_{xy}=0$. In a higher-order insulator they do not, and
$N_{xy}$ is the number of zero modes at each corner of the open flake. They
all sit on one sublattice: on A at the corner $(0,0)$ and at the opposite
one when $N_{xy}>0$, on B when $N_{xy}<0$, and on the other sublattice at
the two other corners. In STM they give a peak at zero bias at the corners
of a flake. The quadrupole moment $q_{xy}$ of the nested Wilson loop adds
up modulo one, so two copies of a quadrupole insulator give $q_{xy}=0$,
while they give $N_{xy}=2$ and two zero modes per corner.

We take the model of Benalcazar, Bernevig and Hughes, with the hopping
$1-\delta$ inside the unit cell and $1+\delta$ between unit cells. It has
$N_{xy}=1$ for $\delta>0$ and $N_{xy}=0$ for $\delta<0$. The sample is
periodic, closed into a torus as for the Bott index, and the chiral
operator is the sublattice $(-1)^{n_x+n_y}$, with $(n_x,n_y)$ the position
of the site on the square lattice.

```python
from pyqula import specialhamiltonian
import numpy as np
h = specialhamiltonian.square_2OTI(0.5) # BBH model, delta=0.5
h = h.get_supercell((10,10)) # periodic sample of 10x10 unit cells
r = h.geometry.r - h.geometry.r[0] # positions from the first site
S = np.diag((-1.)**np.round(r[:,0]+r[:,1])) # sublattice, first site on A
N = h.get_multipole_chiral_number(cells=(10,10),chiral=S) # N_xy
```

This gives $N_{xy}=1$, and $\delta=-0.5$ gives $N_{xy}=0$. The open flake
of the same $10\times10$ unit cells has four states at zero energy, within
$3\times10^{-5}$, one at each corner. On the A sublattice minus the B
sublattice they count $+1$ at the corners $(0,0)$ and $(L,L)$ and $-1$ at
the other two, so the corner count is $N_{xy}$. Random complex hoppings on
the bonds, which keep the chiral symmetry, leave $N_{xy}=1$ as long as the
gap stays open. Two copies of the model coupled by a Rashba term give
$N_{xy}=2$.

The number is returned as a real number, not rounded, but it is an integer
by construction. So a sample smaller than the localization length of the
corner modes gives a wrong integer, not a fraction, and you should compare
two sizes. For $\delta=0.2$ the torus gives $1$ already for $6\times6$ unit
cells, while the open flake gives $0$ for $6\times6$ and $8\times8$ and $1$
from $10\times10$.

The keywords are:
- cells: the number of unit cells $(L_x,L_y)$ of the sample along its two
  lattice vectors, as in `h.get_supercell(cells)`, at least 4 each
- chiral=None: the chiral operator, a matrix, a name or an operator that
  acts inside each unit cell; the default is the sublattice of the
  geometry, with the first orbital on A. Exchanging A and B changes the
  sign of $N_{xy}$
- boundary="periodic": the sample closed into a torus; "open" takes the
  open flake instead
- tol=1e-8: on the torus, the smallest distance to zero of an energy level
  that is accepted

It raises `ValueError` if the Hamiltonian is not two dimensional, if the
sample has fewer than 4 unit cells along a direction, if the chiral operator
does not anticommute with it, if a unit cell has more states on one
sublattice than on the other, and on the torus if a level is at zero
energy, where the gap is closed. A gap that closes at a momentum the torus
of $L_x\times L_y$ cells does not sample is not detected, which is one more
reason to compare two sizes.

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

![Surface and bulk spectral functions of the Haldane model as a function of the momentum along the edge and of the energy.](img/realspace_surface.png#only-light)
![Surface and bulk spectral functions of the Haldane model as a function of the momentum along the edge and of the energy.](img/realspace_surface-dark.png#only-dark)

The figure shows the four arrays of the snippet, reshaped to $100$ momenta
times $100$ energies: the surface weight on the left and the bulk weight on
the right, with the same color scale, cut at the 99th percentile of the
surface weight. The bulk weight shows the gap, from $-0.225$ to $0.225$, at
the momenta $1/3$ and $2/3$, the projections of K and K'. In the surface
weight one chiral branch crosses the gap, from $-0.225$ at momentum $1/3$ to
$0.225$ at momentum $2/3$, through zero energy at momentum $1/2$. The two
spins give the same branch, on top of each other, so this one line carries
the two units of $C=2$.

??? example "Code of this figure (documentation/figures/realspace_surface.py)"

    ```python
    --8<-- "realspace_surface.py"
    ```

The same calculation is available as `h.get_surface_kdos()`, with the
arguments `energies`, `delta`, `kpath` and `operator`, described in
[Spectral functions](spectral-functions.md), together with the quantum spin
Hall case. See `examples/readme_examples/chern_surface_states/main.py` and
`examples/2d/kdos_edge/main.py` for scripts that make the plot.
