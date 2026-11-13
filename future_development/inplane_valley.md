# The in-plane valley operators: what was checked and what is open

Record of the check of `valley_x`, `valley_y` and `add_valley_exchange` of
12 November 2026, of the replacement of their construction, and of what was
left unbuilt. The code is `src/pyqula/operatortk/inplane_valley.py` and the
tests are `tests/topology/test_inplane_valley.py`.

## What exists today

The operators are a closed form. With $\mathbf Q=\mathbf K-\mathbf K'$ and
$\mathbf r_{ij}$ the midpoint of a first-neighbor bond,

$$
\tau_x + i\tau_y = \frac{2}{3}\sum_{\langle ij\rangle}
e^{i\mathbf Q\cdot\mathbf r_{ij}}\left(c^\dagger_i c_j + c^\dagger_j c_i\right) ,
$$

so $\tau_x$ and $\tau_y$ are the real hoppings $\frac23\cos$ and
$\frac23\sin$ of $\mathbf Q\cdot\mathbf r_{ij}$. Two conventions fix what the
formula leaves free:

- $\mathbf K$ is the bond from a B site to its A neighbor (sublattice $+1$)
  rotated by 90 degrees, of length $4\pi/3\sqrt3$. Of the three such bonds
  the one with the smallest polar angle is used. The sublattice labels fix
  the sign of $\tau_z$ in the same way, so $\tau_x\tau_y=i\tau_z$ whatever
  the order of the atoms. Each bond finds $\mathbf K$ from its own in-plane
  direction and the labels of its two sites, so a buckled honeycomb works
  (the in-plane bond length enters $|\mathbf K|$), and so does a stack of
  layers. In `multilayer_graphene` the AA stack labels its second layer
  the other way around, which exchanges the valleys there for $\tau_z$ and
  $\tau_y$ alike; the construction before used one $\mathbf K$ for all the
  layers and so was left-handed in that layer.
- The phase is measured from the site of the honeycomb lattice closest to
  the origin of coordinates, whether an atom sits there or not. A shift by
  a primitive lattice vector rotates $(\tau_x,\tau_y)$ by 120 degrees, and
  a small shift does nothing. A flake and the same flake with a vacancy
  share their axes, also after `gv.center()`.

With the electron-hole degree of freedom the operators and the exchange
are the Nambu form of the normal-state ones, built with
`setup_nambu_spinor()`; before, both raised a shape error there. A sparse
Hamiltonian, with or without the electron-hole degree of freedom, gets the
same matrices as the dense one, as sparse matrices; before, the operator
of a sparse Hamiltonian came back dense. One test runs every route: a
flake, a zigzag and an armchair ribbon, a cell and a three-dimensional
stack of cells, each spinless, spinful and Nambu, each dense and sparse.

The second convention was the maintainer's choice of 13 November 2026,
over the plain origin of coordinates, where a shift by $\mathbf d$ rotates
the axes continuously by $\mathbf Q\cdot\mathbf d$. The cost of a lattice
site as origin is a rule for equally close sites, since the usual origins
of coordinates (a hexagon center, a bond midpoint) are exactly such ties.
Taking the closest site to the last digit was measured to fail there: on
an island with its origin on a bond midpoint, `gv.center()` after removing
one atom moved the coordinates by $0.004$ and rotated the axes by 120
degrees. So every site within `_ORIGIN_WINDOW = 0.3` of the closest
distance competes, and the one furthest along a fixed direction
`_ORIGIN_DIRECTION = 0.1` rad above the $x$ axis wins. With this the
operators of islands with zigzag and armchair edges do not change under
random shifts of up to $0.1$. The jumps did not disappear: they sit where
the origin of coordinates is about $0.15$ away from a tie, and for a lattice
rotated so that two competing sites are level along the fixed direction.

## What was wrong before

The previous construction summed a chiral Kekule coupling over the three
Kekule registries, with two amplitudes $(t_1,t_2)$ calibrated numerically.
In the bulk it was the formula above to $10^{-3}$. Three defects were
measured:

- The calibration asked for "$\tau_x\sigma_x$" in a basis whose phases came
  from `eigh`. With $10^{-12}$ of noise on the Hamiltonian, or another
  LAPACK driver, $(t_1,t_2)$ went from $(0.692,0.055)$ to values such as
  $(0.04+0.71i,\,0.005-0.09i)$: an operator that breaks time reversal,
  with imaginary hoppings of $0.14$ in `add_valley_exchange([0.3,0,0])` and
  a gap of $0.21$ instead of $0.3$. The three tests of that time passed
  with it, since at the Dirac point any pair of valley-flipping operators
  obeys the algebra. The shipped values had an imaginary part of $10^{-3}$
  and an axis $4.07$ degrees away from a plain `add_kekule` pattern.
- A bond at the edge of a finite system whose second hexagon lies outside
  got $|T|=0.692$ or $0.055$ instead of $2/3$. The two equivalent edges of
  an armchair ribbon gave textures of $0.212$ and $0.237$ ($0.218$ on both
  with the truncated bulk operator), and the second row of a zigzag edge
  $0.199$ instead of $0.032$.
- With the first atom on sublattice B, $\tau_x\tau_y=-i\tau_z$, and the
  construction needed the first two atoms to be first neighbors.

## The normalization of tau_z on a buckled lattice

The Haldane hopping is the coupling times the $z$ component of the cross
product of the two bonds it goes through, so it scales with the square of
the in-plane bond length. With the factor $1/4.5$ of the flat lattice,
$\tau_z$ of `geometry.buckled_honeycomb_lattice()` had eigenvalues
$\pm8/9$ on the Dirac points, and `add_valley_exchange([0,0,vz])` opened a
gap $8v_z/9$. `operatortk.valley.valley_coupling` now divides by the
squared in-plane bond length. It returns exactly $1/4.5$ for a flat
lattice, and also when the geometry has no first neighbors or in-plane
bond lengths that differ, so nothing else changes. The Haldane term itself
(`kanemele.haldane`) was not touched.

## Twisted bilayers

Each layer finds its own $\mathbf K$. The moire cell of
`specialgeometry.twisted_bilayer(n)` does not hold a period of the plane
wave of the layers, so it raises the commensurability `ValueError`, and
three moire cells do: with first-neighbor hopping only (decoupled layers)
the eight Dirac states of `twisted_bilayer(n).get_supercell(3)` obey the
full algebra to $10^{-15}$, for $n=1$ (252 sites) and $n=2$ (684 sites).
With an interlayer coupling and for a finite twisted flake nothing was
computed.

## What the textures are

Checked with the old and the new operator, which agree away from the edges:

- Pristine bulk: zero. A valley exchange gives a uniform texture
  antiparallel to $\vec v$.
- Vacancy, on a disk centred on it: $f=\tau_x+i\tau_y$ obeys
  $f(C_3\mathbf r)=e^{2\pi i m/3}f(\mathbf r)$ to $10^{-12}$, with $m=-1$
  for a vacancy on sublattice $+1$ and $m=+1$ on $-1$. Next to the vacancy
  only $m$ modulo 3 is defined (the shells show $m$ and $m\pm3$); for
  $r>6$ the harmonic $e^{im\theta}$ dominates and decays roughly as
  $1/r^2$, the same on disks of radius 10 and 16. $|f|=0.20$ on the three
  neighbors.
- Zigzag edge: $|f|\approx0.2$ on the edge row, rotating by 120 degrees
  from cell to cell, zero net over the Kekule period. Armchair edge:
  uniform along the edge, $|f|=0.22$.
- These numbers are for the half-filled first-neighbor model, summed over
  the occupied states. The texture of a flake did not move under a shift of
  the Fermi level of $10^{-3}$.

## Open

- **A sublattice-diagonal operator.** $2\cos(\mathbf Q\cdot\mathbf r_i)$
  onsite is $\tau_x\sigma_0$ on the Dirac points, the density wave at
  $\mathbf Q$, the part of the intervalley coherence that STM sees on the
  atoms rather than on the bonds. It is not built. The bond operator here
  is the Kekule mass, which anticommutes with the Dirac Hamiltonian; around
  a vacancy the onsite one is expected to wind twice on the sublattice
  opposite to the vacancy, which was not computed.
- **A coarse-grained texture.** The local expectation value keeps the
  120 degree spiral that a bond anisotropy at zero momentum produces,
  which is not intervalley coherence. This was measured at a zigzag edge;
  a uniaxial strain should do the same and was not computed. Averaging
  over the Kekule cell removes it; no routine does this.
- **`real_space_vev` is at the Gamma point only**, so a periodic texture
  needs a supercell in place of a k-mesh.
