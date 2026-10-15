# Other classes

## Heterostructure functions and methods

### HT.get_dc_current()
Compute the time-averaged (DC) current through a two-terminal junction at a
bias voltage $V$, with the Floquet-Keldysh formalism. The leads can be
normal or superconducting, built with `heterostructures.build(h1,h2)`, with
or without a central region. With superconducting leads this includes
multiple Andreev reflection. See [Quantum transport](transport.md).

Arguments:

- voltage: bias voltage $eV$

Optional arguments:

- nmax=6, nmax_max=40: initial and maximal number of Floquet sidebands. The
  number grows until the current converges to `tol`; a warning is printed
  if `nmax_max` is reached first
- tol=1e-3: relative tolerance on the current
- temperature=0.: temperature of the leads
- delta=None: broadening, by default `HT.delta`. It must be much smaller
  than the smallest superconducting gap

Returns the DC current

### HT.get_iv_curve()
`HT.get_dc_current()` for an array of voltages.

Arguments:

- voltages: array of bias voltages

Returns an array of DC currents

### HT.didv_curve() / lp.didv_curve()
The differential conductance $dI/dV$ for an array of energies, the same as
`[ht.didv(energy=e) for e in es]`. It is also available as
`didv(energies=...)`.

Arguments:

- energies: array of bias energies
- any keyword argument of `didv` (`method`, `delta`, `temp`, ...)

Returns an array of $dI/dV$ values

Note that `T` means different things on the two classes. On a
`Heterostructure`, `T` is the temperature. On a `LocalProbe`, `T` is the
transparency of the probe, the coupling between probe and sample, and the
temperature is `temp`. A `T=` or a `delta=` given to one call applies only to
that call. `HT.with_delta(delta)`, `lp.with_delta(delta)` and
`lp.with_coupling(T)` return a copy with a new broadening or coupling, to
use in several calls.

### HT.get_kappa()
Compute the exponent ratio $\kappa$, which tells single-particle tunneling
from Andreev reflection (also `LocalProbe.get_kappa()`, see
[Quantum transport](transport.md)).

Optional arguments:

- energy=0.0: energy at which $\kappa$ is computed (returns a number)
- energies: array of energies instead (returns an array); do not give both
- temp=0.: temperature. A finite value averages the conductances over the
  Fermi window
- T=1e-2: coupling around which the exponent is computed

Returns $\kappa$, a number or an array like `energies`

$\kappa$ is the exponent $d\log G/d\log T$ of the conductance with the
coupling $T$, divided by the same exponent with the pairing removed. It is
1 for single-particle tunneling, since $G\propto T$, and 2 for Andreev
reflection, since $G\propto T^2$. Inside a superconducting gap the
broadening `delta` of a `LocalProbe` acts as a finite quasiparticle
lifetime, and it gives a normal conductance $G_N\delta/\Delta$. The Andreev
conductance is $G_N^2/4$ in the tunneling limit, so the plateau at 2 needs
`delta` well below $\Delta G_N/4$ at the smallest coupling, for example
`delta=1e-10` for a gap of 0.1 down to `T=1e-3`. See
`examples/transport/localprobe_kappa_1D` and
`jupyter-notebooks/functionalities/quantum_transport/07_differential_decay_rate.ipynb`.

## SpinModel functions and methods

A `SpinModel` is a classical spin model with energy

$$E = \sum_i \vec b_i \cdot \vec S_i + \sum_{ij} \vec S_i \cdot J_{ij} \cdot \vec S_j ,$$

where $\vec S_i$ is a unit vector with angles $(\theta_i,\phi_i)$,
$\vec b_i$ a magnetic field, and $J_{ij}$ a $3\times3$ exchange tensor. See
[Classical spin models](classical-spins.md).

### sm.add_heisenberg()
Add exchange couplings between neighbor shells. Several calls add up.

Optional arguments:

- Jij=None: list of couplings of each shell, for example `[J1,J2,J3]` for
  first, second and third neighbors. It accepts everything that `tij` of
  `g.get_hamiltonian()` accepts; `None` is first neighbors with $J=1$
- Jm=[1.,1.,1.]: diagonal weights $(J_x,J_y,J_z)$ multiplying `Jij`; use
  different values for XXZ or Ising anisotropy

Modifies `sm` in place, no return value

### sm.add_field()
Add a uniform magnetic field $\vec b$ to every site. Several calls add up.

Arguments:

- v: the field, a 3-vector, for example `[0.,0.,1.]`

### sm.add_tensor() / sm.add_tensor_2d()
Add a general coupling tensor $J_{ij}$ given by a function
`fun(r1,r2) -> 3x3 matrix` (see `generating_functions` below).
`add_tensor` only couples sites inside the unit cell. `add_tensor_2d` also
sums the periodic images $i_a\mathbf a_1 + j_a\mathbf a_2$ with
$i_a,j_a$ in `[-ncells,ncells]`, and can rotate the tensor of each image
around $z$ by the angle $\pi(v_0 i_a + v_1 j_a)$, with $(v_0,v_1)$ given by
`vspiral`. This puts a spin-spiral wavevector into the exchange.

Arguments:

- fun: coupling function, for example one from `generating_functions`

Optional arguments (`add_tensor_2d` only):

- ncells=1: number of periodic images on each side, along each lattice
  vector
- vspiral=[0.,0.]: rotation angles per image, in units of $\pi$

Modifies `sm` in place, no return value

### classicalspin.generating_functions()
Return a coupling function `fun(r1,r2) -> 3x3 matrix` of a common type, to
use with `add_tensor` and `add_tensor_2d`.

Optional arguments:

- name="Heisenberg": the type of coupling. `"Heisenberg"` is isotropic
  exchange, cut by `fc(distance)`. `"Linear"` is the tensor
  $J\,\hat r\hat r/r^3$, with $\hat r$ the bond direction. `"RKKYTI"` is the
  RKKY coupling on the surface of a topological insulator,
  $J(\tfrac32\hat r\hat r - 1)/r^3$. `"ZZ"` is Ising, only $S^z_i S^z_j$.
  `"XYZ"` is diagonal with weights `v`. `"DM"` is a first-neighbor
  Dzyaloshinskii-Moriya coupling $\vec D_{ij}\cdot(\vec S_i\times\vec S_j)$,
  with $\vec D_{ij}$ along $\mathbf r_{ij}\times\vec v$, so `v` plays the
  role of the direction of the intermediate ion (or a function of the bond
  vector)
- J=1.0: strength of the coupling
- v=[0.,0.,1.]: vector used by the type, or a function of the bond vector
- fc=None: function of the distance that cuts or modulates the coupling;
  by default first neighbors (`0.9 < d < 1.1`)
- fdiff=lambda x,y: x-y: how the bond vector is computed from `(r1,r2)`
- fr=None: extra rotation matrix `fr(r1,r2)` applied to the coupling (by
  default the identity)

Returns the coupling function

### sm.minimize_energy()
Find the classical ground state, by minimizing $E$ over the angles
$(\theta_i,\phi_i)$ of all spins from several random starting points. Only
states with the periodicity of the unit cell are found, so a spiral needs a
supercell that fits it.

Optional arguments:

- theta0=None, phi0=None: starting angles. With `None` each try starts from
  random angles, which is what makes `tries` useful. With explicit arrays
  every try starts at the same point and gives the same result
- tries=10: number of minimizations; the one with the lowest energy is
  kept
- calle=None: a function `calle(sm) -> float` added to the energy, for
  example a penalty that favors a texture
- silent=False: if `True`, the minimum energy is not printed

Sets `sm.theta`, `sm.phi` and `sm.magnetization` to the best result, and
returns `(theta,phi)`

### sm.get_energy() / sm.energy()
The energy $E$ above for the current angles. Returns a number.

### sm.get_local_energy()
The energy of each site: its field term plus half of each exchange term
it belongs to. The values add up to `sm.get_energy()`.

Returns an array over sites

### sm.get_magnetization()
The Cartesian components of the spins,
$\vec S_i = (\sin\theta_i\cos\phi_i,\sin\theta_i\sin\phi_i,\cos\theta_i)$.

Returns `(mx,my,mz)`, arrays over sites

### align.most_perpendicular_vector() / align.most_perp_basis()
(`classicalspintk.align`) Find the direction most perpendicular to a set of
vectors, for example a magnetization, and rotate the vectors so that this
direction is the new $z$ axis. The vectors are then mostly in the new $xy$
plane, which helps to plot a skyrmion or a spiral as arrows in the plane.
See `examples/classicalspin/perpendicular/main.py`.

Returns the perpendicular unit vector, or (for `most_perp_basis`) the
vectors in the rotated basis

## LatticeGas functions and methods

A `LatticeGas` is a classical lattice gas with occupations $n_i=0,1$ and
energy

$$E = \sum_i \mu_i n_i + \sum_{i\neq j} J_{ij} n_i n_j ,$$

where the sum runs over ordered pairs, so each bond counts twice. See
[Lattice gas models](lattice-gas.md).

### lg.add_interaction()
Add a density-density coupling $J_{ij}$ between neighbor shells. Several
calls add up.

Optional arguments:

- Jij: list of couplings of each shell, for example `[J1,J2,J3]`. A positive
  value is repulsive and a negative one attractive. It accepts everything
  that `tij` of `g.get_hamiltonian()` accepts

Modifies `lg` in place, no return value

### lg.set_filling()
Replace the occupations `lg.den` by a random configuration with a given
filling, for example to start a new anneal.

Arguments:

- filling: fraction of occupied sites (rounded to an integer number of
  sites)

### lg.get_energy()
The energy $E$ above for the current occupations `lg.den`. Returns a
number.

### lg.optimize_energy()
Lower the energy with Metropolis moves at fixed filling. In each step one
to three random pairs of an occupied and an empty site are swapped. The
move is accepted if the energy does not increase, and with probability
$e^{-\Delta E/T}$ otherwise.

Optional arguments:

- temp=0.1: Metropolis temperature $T$. `temp=0` only accepts moves that
  do not increase the energy
- ntries=1e5: number of attempted swaps
- resync_every=1000: every how many attempts the energy is recomputed from
  scratch, to avoid accumulated rounding errors
- patience=None: if given, stop after this many attempts without a new
  lowest energy
- checkpoint_at=None: an integer or a list of integers; a copy of `lg.den`
  after that many attempts is stored in `lg.checkpoints`, a dictionary
  `step -> den`

Sets `lg.den` to the final configuration, and returns the energy after each
attempt

### lg.anneal()
Simulated annealing: `lg.optimize_energy()` at each temperature of a
decreasing list. The lowest-energy configuration of the whole schedule is
kept.

Optional arguments:

- temps=None: temperatures, from high to low; by default 10 temperatures
  on a geometric scale from 2.0 to 0.05
- ntries=1e4: attempts per temperature
- checkpoint_at=None: as in `lg.optimize_energy()`, but counted over the
  whole schedule
- any other keyword of `lg.optimize_energy()`, used at every temperature

Sets `lg.den` to the best configuration, and returns the energies of the
whole schedule

### lg.optimize_energy_multistart()
Run `nstart` independent optimizations from random configurations at the
current filling, and keep the lowest energy. This lowers the chance of
ending in a metastable state. The runs are parallel after
`parallel.set_cores()` (see [Parallelism](parallelism.md)).

Optional arguments:

- nstart=10: number of runs
- any other keyword of `lg.optimize_energy()` (`temp`, `ntries`,
  `patience`...), the same for every run

Sets `lg.den` to the best configuration, and returns its energy

### lg.optimize_grand_canonical()
Metropolis sampling in the grand-canonical ensemble. Single sites are
switched between occupied and empty, so the filling is not fixed and is set
by the chemical potential `lg.mu`. Use it to scan a phase diagram as a
function of the chemical potential, or to sample at a fixed temperature for
`latticegas.get_specific_heat()` and `get_susceptibility()`.

Optional arguments: the same as `lg.optimize_energy()` (`temp`, `ntries`,
`resync_every`, without `patience`), but the default is `temp=1.0`

Sets `lg.den` to the final configuration, and returns `(es, ns)`: the energy
and the number of occupied sites after each attempt

### latticegas.get_specific_heat() / latticegas.get_susceptibility()
Thermodynamic quantities from a trajectory sampled at a fixed temperature
$T$, by `lg.optimize_energy()` or `lg.optimize_grand_canonical()`.
`get_specific_heat(es, temp, burn=0.2)` gives the specific heat
$C=\mathrm{Var}(E)/T^2$ from the energies. `get_susceptibility(ns, temp,
burn=0.2)` gives the compressibility $dN/d\mu=\mathrm{Var}(N)/T$ from the
number of particles; it only makes sense in the grand-canonical case, where
$N$ fluctuates. `burn` is the fraction at the start of the trajectory that
is thrown away as thermalization.

### lg.add_tensor()
Add a coupling $J_{ij}=\mathrm{fun}(\mathbf r_i,\mathbf r_j)$ between all
pairs of sites, for interactions beyond neighbor shells, for example a
screened or dipolar $1/r^n$ interaction. Pairs with zero coupling are not
stored.

Modifies `lg` in place, no return value

### lg.regroup()
Merge repeated pairs from several `add_interaction()` or `add_tensor()`
calls, summing their couplings. The energy does not change, only the
calculation gets faster.

Modifies `lg` in place, no return value

### lg.write() / lg.read()
Write the occupations `lg.den` to a text file, and read them back. `read()`
raises `ValueError` if the number of sites in the file is not `lg.nsites`.

Optional arguments:

- name="DENSITY.OUT": file name

### lg.get_local_energy() / lg.get_local_mu()
`get_local_energy()` gives the energy of each site,
$\mu_i n_i + \sum_j J_{ij} n_i n_j$; the values add up to `lg.get_energy()`.
`get_local_mu()` gives the same expression with site $i$ occupied, meaning
the energy cost of putting a particle on site $i$ with its neighbors as they
are.

Optional arguments:

- normalize=False: divide the value of each site by the sum of its couplings

Returns an array over sites

### lg.get_correlator()
The density-density correlator of the current occupations, as a function of
the neighbor shell. It shows whether the ground state is ordered. `n` is
the number of shells, `normalized` divides out the mean occupation.

Returns `(distances, correlators)`

### lg.get_structure_factor()
The structure factor of the current occupations,

$$S(\mathbf q)=\frac{1}{N}\left|\sum_i (n_i-\bar n)\, e^{-i\mathbf q\cdot \mathbf r_i}\right|^2 ,$$

computed on the real positions of the sites. The correlator gives the
length scale of the order, and $S(\mathbf q)$ gives its wavevector. Since
the mean occupation $\bar n$ is subtracted, $S(\mathbf q=0)=0$, so a peak at
another $\mathbf q$ signals order. This is the quantity measured in a
diffraction experiment.

Optional arguments:

- qpath=None: array of $\mathbf q$ vectors. By default a square grid of
  `nq`$\times$`nq` points between $-q_{max}$ and $q_{max}$
- nq=60: points per direction of the default grid
- qmax=None: size of the default grid; by default $2\pi/d$, with $d$ the
  first-neighbor distance

Returns `(qpath, sq)`, the $\mathbf q$ vectors and $S(\mathbf q)$


## LatticeIsing functions and methods

A `LatticeIsing` is a classical Ising model with spins $s_i=\pm1$ and
energy

$$E = -\sum_i b_i s_i - \sum_{i\neq j} J_{ij} s_i s_j ,$$

where the sum runs over ordered pairs, so each bond counts twice, which is
two times the usual convention. See [Ising models](ising.md).

### li.add_interaction()
Add an exchange $J_{ij}$ between neighbor shells. Several calls add up.

Optional arguments:

- Jij: list of couplings of each shell, for example `[J1,J2,J3]`. A
  positive value is ferromagnetic (note the opposite sign to
  `LatticeGas.add_interaction()`). It accepts everything that `tij` of
  `g.get_hamiltonian()` accepts

Modifies `li` in place, no return value

### li.add_field()
Add a magnetic field $b_i$.

Arguments:

- h: a number (the same on every site) or an array with one value per site

Modifies `li` in place, no return value

### li.set_magnetization()
Replace the spins `li.s` by a random configuration with a given
magnetization.

Arguments:

- m=0.0: average magnetization, between -1 and 1 (rounded to an integer
  number of up spins)

### li.get_energy()
The energy $E$ above for the current spins `li.s`. Returns a number.

### li.get_magnetization()
The magnetization per site, $m=\frac{1}{N}\sum_i s_i$, between -1 and 1.

### li.optimize_energy()
Metropolis dynamics with single spin flips. In each step one random spin is
flipped. The flip is accepted if the energy does not increase, and with
probability $e^{-\Delta E/T}$ otherwise. The magnetization is not fixed.

Optional arguments:

- temp=1.0: Metropolis temperature $T$; `temp=0` only accepts flips that do
  not increase the energy
- ntries=1e5: number of attempted flips
- resync_every=1000: every how many attempts the energy is recomputed from
  scratch
- checkpoint_at=None: an integer or a list of integers; a copy of `li.s`
  after that many attempts is stored in `li.checkpoints`

There is no `patience` option, because the trajectory is meant for
`latticegas.get_specific_heat()` and `get_susceptibility()`, and stopping
early would bias them.

Sets `li.s` to the final configuration, and returns `(es, ms)`: the energy
and the total magnetization $\sum_i s_i$ after each attempt

### li.optimize_conserved()
Kawasaki dynamics: in each step a random up spin and a random down spin are
exchanged, so the magnetization is fixed. Raises `ValueError` if `li.s` does
not contain both up and down spins.

Optional arguments: the same as `li.optimize_energy()`, but with default
`temp=0.1`, and also:

- patience=None: if given, stop after this many attempts without a new
  lowest energy

Sets `li.s` to the final configuration, and returns the energy after each
attempt

### li.anneal()
Simulated annealing: `li.optimize_energy()` at each temperature of a
decreasing list, keeping the lowest-energy configuration of the whole
schedule.

Optional arguments:

- temps=None: temperatures from high to low; by default 10 temperatures on
  a geometric scale from 2.0 to 0.05
- ntries=1e4: attempts per temperature
- checkpoint_at=None: as in `li.optimize_energy()`, counted over the whole
  schedule
- any other keyword of `li.optimize_energy()`, used at every temperature

Sets `li.s` to the best configuration, and returns `(es, ms)` for the whole
schedule

### li.optimize_energy_multistart()
Run `nstart` independent optimizations from random spins with the current
magnetization, and keep the lowest energy. Parallel as in
`lg.optimize_energy_multistart()`.

Optional arguments:

- nstart=10: number of runs
- any other keyword of `li.optimize_energy()` (`temp`, `ntries`), the same
  for every run

Sets `li.s` to the best configuration, and returns its energy

### li.get_local_energy() / li.get_local_field()
`get_local_energy()` gives the energy of each site; the values add up to
`li.get_energy()`. `get_local_field()` gives the effective field on each
site, $h^{\mathrm{eff}}_i=b_i+2\sum_k J_{ik}s_k$, so that flipping $s_i$
costs exactly $2s_ih^{\mathrm{eff}}_i$.

Returns an array over sites

### li.get_correlator() / li.get_structure_factor()
The spin-spin correlator and the structure factor of the current spins
`li.s`. The arguments are the same as in `lg.get_correlator()` and
`lg.get_structure_factor()`.

### li.add_tensor() / li.regroup()
The same as `lg.add_tensor()` and `lg.regroup()`.

Modifies `li` in place, no return value

### li.write() / li.read()
Write the spins `li.s` to a text file, and read them back. `read()` takes
the sign of each value, so the spins are exactly $\pm1$, and raises
`ValueError` if the number of sites in the file is not `li.nsites`.

Optional arguments:

- name="SPIN.OUT": file name
