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

## Repairs of the seventh bug audit

Done on 14 November 2026, on the findings nov13#0, nov13#1, nov13#3 and
nov13#4 of `future_development/bug_audit_7.md`.

- **Vertical bonds gave NaN.** A first-neighbor bond with no in-plane
  part, between two layers on top of each other at the bond length, gave
  $\mathbf K=0/0$, and both guards of the geometry let the NaN through, so
  `valley_x`, `valley_y` and `add_valley_exchange` came back with NaN
  entries and no error. `specialgeometry.multilayer(g,l=[0,0])` at its
  default `dz=[0,0,1]` gave 36 NaN entries of 1296 in the 3x3 cell, and the
  (111) diamond film `geometry.buckled_honeycomb_lattice(2)` 14. Such a bond
  has no direction to give $\mathbf K$, and the plane wave lives on the
  bonds of each layer, so `_first_neighbor_pairs` now leaves out the bonds
  whose in-plane length is below `_VERTICAL = 1e-3`, as a longer interlayer
  bond already was. The two guards are written so that a NaN fails them.
  `test_vertical_bonds_carry_no_inplane_valley` checks that both geometries
  give the operators of the same layers pulled apart by two bond lengths,
  to $10^{-12}$. Raising for these geometries instead was the other option.
  With the vertical bonds left out no NaN reaches the guards, so a copy of
  the code with the old guards, `np.max(...)>1e-3`, passed every test.
  `test_bond_without_inplane_direction_cannot_pass_the_guards` lets the
  vertical bonds through again (`_VERTICAL=-1`) and asks for the error on
  the AA stack; the old guards fail it.
  `valley_coupling` of $\tau_z$ (`operatortk/valley.py`) had the same blind
  spot: on the film of two buckled layers it took the vertical bonds as
  in-plane bonds of length 0, so the lengths differed, it returned the flat
  $1/4.5$, and $\tau_z$ had the eigenvalues $\pm8/9$ on the Dirac states
  ($\pm1$ with the layers pulled apart). It now leaves the vertical bonds
  out too, and the eigenvalues are $\pm1$. `test_vertical_bonds_do_not_change_tauz`
  compares the film with the same layers pulled apart: $2\times10^{-2}$ on
  the code before, $5\times10^{-17}$ after.
- **`delta=` was dropped on a sparse Hamiltonian.** Since the operators of
  a sparse Hamiltonian are sparse, the sharpening $e\to e/(|e|+\delta)$
  returned them untouched, and a dense and a sparse Hamiltonian of the same
  geometry gave operators that differ by up to 0.32 at $\delta=0.1$ (0.12 for
  `valley_x` on the 3x3 cell at $k=(0.1,0.2)$). `sharpen.py` now
  diagonalizes the matrix dense (`algebra.eigh` refuses one above
  `algebra.maxsize`) and returns the result in the storage of the input.
  This also repairs `valley` ($\tau_z$), which had the same skip.
  `test_sharpened_valley_is_the_same_dense_and_sparse` compares the two
  storages and the eigenvalue map, on a flake and on the 3x3 cell.
- **The ev2d tests sampled one k-point.** With `nk=2` the four points of
  `linspace(-1,1,nk)+k0` differ by integers, so the test that a
  k-dependent operator is applied at the k-point of each state could not
  see an operator applied at a fixed mesh point. The tests use `nk=4` now
  (`nk=3` has the same defect), and a copy of the code that applies the
  operator at the first mesh point fails the new test and passes the old one.
- **One Bloch sum per state.** An expectation value, `ev2d` and
  `get_bands` apply the operator to every state of a k-point in turn, and
  the operator rebuilt its Bloch sum, and with `delta` its diagonalization,
  for each state. `get_inplane_valley` now keeps the matrix of the last
  k-point and hands out a copy when the matrix itself is asked for. The
  results are the same to the last digit. On a 9x9 honeycomb cell (162
  sites, spinless, two threads, the best of three runs, the old and the new
  code back to back on a shared machine, so the ratios are what matters):

  | call | old | new |
  | --- | --- | --- |
  | `h.get_vev("valley_x",nk=10)` | 4.91 s | 1.27 s |
  | the same with `delta=0.1`, `nk=3` | 21.0 s | 0.36 s |
  | `spectrum.ev2d(h,operator="valley_x",nk=8)` | 3.31 s | 0.78 s |
  | `h.get_bands(operator="valley_x")` at four k-points | 0.49 s | 0.18 s |

  The peak memory is 0.44 GB in both. The same rebuild was in `valley`
  ($\tau_z$, `operatortk/valley.py`) and in the two- and three-dimensional
  `velocity` (`operators.get_velocity`), which now keep the matrix and the
  velocities of the last k-point in the same way. On the same cell, the
  best of two runs, the committed tree against the working tree, with the
  same results byte for byte and 0.38 GB in both:

  | call | old | new |
  | --- | --- | --- |
  | `h.get_vev("valley",nk=10)` | 6.05 s | 1.14 s |
  | the same with `delta=0.1`, `nk=3` | 17.6 s | 0.40 s |
  | `h.get_vev("velocity",nk=4)` | 5.46 s | 0.50 s |
  | `h.get_bands(operator="valley")` at four k-points | 0.59 s | 0.20 s |

  `test_tauz_kept_between_calls_is_the_one_of_each_k_point` and
  `test_velocity_kept_between_calls_is_the_one_of_each_k_point` go back and
  forth between k-points and compare with a new operator.

  The review timed the change alone, the committed tree against the
  committed tree with only `inplane_valley.py` and `sharpen.py` replaced,
  since the working tree had other changes: on the same cell
  `h.get_vev("valley_x",nk=10)` went from 4.38 s to 1.08 s, and with
  `delta=0.1` at `nk=2` from 7.14 s to 0.17 s (the best of three pairs, two
  threads, shared machine, 0.38 GB in both). The two results are the same
  byte for byte. The k-point and its matrix are kept as one object, so that
  two threads that share an operator cannot take the matrix of one k-point
  for another, and a flake, which has no k-point, does not go through it.
  `test_operator_kept_between_calls_is_the_one_of_each_k_point` fails on a
  copy that hands out the kept matrix itself, and on a copy that ignores k.
