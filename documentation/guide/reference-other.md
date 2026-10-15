# Other classes

## Heterostructure functions and methods

### HT.get_dc_current()
Compute the time-averaged (DC) current through a two-terminal junction at a
given bias, using the Floquet-Keldysh formalism (see "Multiple Andreev
reflection and AC-Josephson current"). Works for any combination of
normal/superconducting leads built with `heterostructures.build(h1,h2)`,
with or without an explicit `central=` Hamiltonian (the latter is solved
by a general dense Floquet inversion, and assumes the bias drops across
the junction's rightmost bond).

Arguments:

- voltage: bias voltage

Optional arguments:

- nmax=6, nmax_max=40: initial/maximum number of Floquet sidebands (increased adaptively until convergence; a warning is issued if `nmax_max` is reached before `tol` is satisfied)
- tol=1e-3: relative convergence tolerance used to decide when to stop increasing the sideband count
- temperature=0.: lead temperature
- delta=None: broadening (defaults to `HT.delta`); should be much smaller than the smallest relevant gap for small-gap superconductors

Returns the DC current

### HT.get_iv_curve()
Convenience wrapper: `get_dc_current` evaluated over an array of voltages.

Arguments:

- voltages: array of bias voltages

Returns an array of DC currents

### HT.didv_curve() / lp.didv_curve()
Convenience wrapper: `didv` evaluated over an array of energies, in parallel, the array-native
equivalent of `[ht.didv(energy=e) for e in es]`. Also reachable as `didv(energies=...)` (mutually
exclusive with `didv`'s scalar `energy=...`, same convention as `get_kappa`). If `use_aaa=True`
and the sweep resolves to the Floquet-Keldysh method, one AAA self-energy interpolant is built and
shared across the whole sweep.

Arguments:

- energies: array of bias energies
- any keyword argument accepted by `didv` (`method`, `delta`, `use_aaa`, `nmax_max`, `temp`, ...)

Returns an array of dI/dV values

`T` means different things on the two classes. On a `Heterostructure`,
`T` is the temperature. On a `LocalProbe` it is the probe *transparency*,
the same knob as `LocalProbe(...,T=...)`, `set_coupling` and
`get_kappa(T=...)`, and the temperature there is `temp` (or its alias
`temperature`). Both `lp.didv(T=...)` and `lp.didv_curve(...,T=...)` honour
it for that call alone, leaving the probe itself untouched.

`delta=` passed to any of `didv`, `didv_curve` or `get_smatrix` likewise
applies to that call, self-energies included, taking precedence over the
junction's own attribute. For a `LocalProbe` it sets both the probe
broadening and the sample's bulk broadening, exactly as the constructor
argument does. `HT.with_delta(delta)`, `lp.with_delta(delta)` and
`lp.with_coupling(T)` return a shallow copy with that one knob rebound,
for when the same setting is wanted across several calls.

### HT.get_kappa()
Compute the superconducting/normal conductance power-law-ratio "kappa"
diagnostic (also available as `LocalProbe.get_kappa()`, see "Multiple
Andreev reflection and AC-Josephson current").

Optional arguments:

- energy=0.0: energy at which to evaluate kappa (returns a scalar)
- energies: array of energies to evaluate at once instead (returns an array); mutually exclusive with `energy`
- temp=0.: temperature. The default is the zero-temperature result; a nonzero value thermally averages each conductance entering the power-law fit
- T=1e-2: reference coupling the power-law exponent is extracted around

Returns kappa (a scalar, or an array matching `energies`)

At `temp=0.` (the default), kappa is the exponent `d(log G)/d(log T)` of the conductance divided by the same exponent with the pairing removed: how steeply the conductance scales with the probe-sample coupling, 1 for single-particle tunneling and 2 for Andreev reflection. For a `LocalProbe` whose probe lead is not itself superconducting each exponent is an exact derivative, obtained with `jax.grad`; otherwise it is estimated from the conductance at two nearby couplings (0.9T and 1.1T). Which of the two applies is decided automatically, and the two agree to within the finite-difference bias of the secant. Inside a superconducting gap, the broadening `delta` of a `LocalProbe` acts as a finite quasiparticle lifetime of the sample, which leaves a normal conductance $G_N\delta/\Delta$; since the Andreev one is $G_N^2/4$ in the tunneling limit, the plateau at 2 needs `delta` well below $\Delta G_N/4$ at the smallest coupling, `delta=1e-10` for a gap of 0.1 down to `T=1e-3`. See `examples/transport/localprobe_kappa_1D` for a runnable version, and `jupyter-notebooks/functionalities/quantum_transport/07_differential_decay_rate.ipynb` for the executed notebook.

## SpinModel functions and methods

### sm.add_heisenberg()
Add shell-based exchange couplings to the model, on top of any interactions already added
(repeated calls accumulate).

Optional arguments:

- Jij=None: list of shell couplings, e.g. `[J1,J2,J3]` for first/second/third neighbor
  exchange; passed through to `Geometry.get_hamiltonian(tij=Jij)`, so anything that
  constructor accepts for `tij` works here too (`None` is that constructor's own default,
  first-neighbor hopping)
- Jm=[1.,1.,1.]: diagonal $(J_x,J_y,J_z)$ weights, applied on top of `Jij`; use unequal
  values for XXZ/Ising-like anisotropy

Modifies `sm` in place, no return value

### sm.add_field()
Add a uniform Zeeman field to every site's `sm.b`. Repeated calls accumulate (not overwrite).

Arguments:

- v: 3-vector field, e.g. `[0.,0.,1.]`

### sm.add_tensor() / sm.add_tensor_2d()
Add a general pairwise tensor coupling $J_{ij}$ from a function `fun(r1,r2) -> 3x3 matrix`
(see `generating_functions` below), rather than the fixed-form couplings `add_heisenberg`
builds. `add_tensor` only considers pairs within the home cell; `add_tensor_2d` additionally
sums periodic images `ia*a1 + ja*a2` for `ia,ja` in `[-ncells,ncells]`, and can rotate the
coupling tensor of each image about $z$ by `vspiral[0]*ia + vspiral[1]*ja` (in units of
$\pi$), a way to embed a spin-spiral wavevector directly into the exchange tensor.

Arguments:

- fun: coupling function, e.g. one returned by `generating_functions`

Optional arguments (`add_tensor_2d` only):

- ncells=1: number of periodic images to sum on each side, along each lattice vector
- vspiral=[0.,0.]: per-image in-plane rotation angle coefficients, see above

Modifies `sm` in place, no return value

### classicalspin.generating_functions()
Factory returning a `fun(r1,r2) -> 3x3 matrix` two-point coupling function for a standard
exchange form, for use with `add_tensor`/`add_tensor_2d`.

Optional arguments:

- name="Heisenberg": one of `"Heisenberg"` (isotropic, cutoff `fc(distance)`), `"Linear"`
  (dipolar $1/r^3$ tensor), `"RKKYTI"` (RKKY on a topological-insulator surface, PRB 81
  233405), `"ZZ"` (Ising, $S^z_i S^z_j$ only), `"XYZ"` (diagonal, anisotropic weights `v`),
  `"DM"` (Dzyaloshinskii-Moriya, with `v` the intermediate-ion/mirror direction, or a function
  of the bond vector)
- J=1.0: overall coupling strength
- v=[0.,0.,1.]: form-dependent vector (or vector-valued function of the bond vector), see above
- fc=None: distance-dependent cutoff/envelope, e.g. restricting `"Heisenberg"`/`"ZZ"` to first
  neighbors (`0.9 < d < 1.1`, the default)
- fdiff=lambda x,y: x-y: how the bond vector is computed from `(r1,r2)`; override for e.g. a
  minimum-image convention
- fr=None: extra rotation matrix `fr(r1,r2)` applied on top of the base coupling (defaults to
  the identity)

Returns the coupling function

### sm.minimize_energy()
Multistart local minimization of the classical energy over every spin's $(\theta,\phi)$
angles, by local gradient-based minimization of the
energy (or, if `calle` is given, gradient-free Powell). Only the ground state at $\Gamma$ is
found. There are no twisted boundary conditions, so an incommensurate (e.g. spiral) texture
needs an explicit supercell that fits it.

Optional arguments:

- theta0=None, phi0=None: initial angles; each try re-randomizes them in $[0,\pi]$/$[0,2\pi]$
  when left as `None` (the default), which is what makes `tries` explore different basins;
  passing explicit arrays starts every try from the same point (the optimizer itself is
  deterministic), so `tries>1` is only useful with the default
- tries=10: number of independent minimizations; the lowest-energy one is kept
- calle=None: optional extra function `calle(sm) -> float` added to the energy during
  minimization, e.g. a penalty favoring a particular texture
- silent=False: `True` does not print the minimum energy found

Updates `sm.theta`, `sm.phi` and `sm.magnetization` in place to the best try found, and
returns `(theta,phi)`

### sm.get_energy() / sm.energy()
Evaluate the total energy $\sum_i \vec b_i \cdot \vec S_i + \sum_{ij} \vec S_i \cdot J_{ij}
\cdot \vec S_j$ of the current `sm.theta`/`sm.phi` snapshot. Returns a scalar.

### sm.get_local_energy()
Per-site breakdown of the current snapshot's energy: each site's own field term plus half of
every exchange term it takes part in (as with `LatticeGas.get_local_energy()`, each bond's
energy is split evenly between its two endpoints), so the values sum exactly to
`sm.get_energy()`.

Returns an array over sites

### sm.get_magnetization()
Convert the current `sm.theta`/`sm.phi` angles to Cartesian components.

Returns `(mx,my,mz)`, arrays over sites

### align.most_perpendicular_vector() / align.most_perp_basis()
(`classicalspintk.align`) Given a set of vectors (typically a `SpinModel`'s magnetization),
find the direction most nearly perpendicular to all of them, and use it to build a rotated
basis in which that direction is the new $z$ axis, i.e. the vectors end up mostly in the new
$xy$ plane. Useful for plotting a magnetization texture (e.g. a skyrmion or spiral) as an
in-plane quiver plot when it is not already aligned with a coordinate axis; see
`examples/classicalspin/perpendicular/main.py`.

Returns a perpendicular unit vector, or (for `most_perp_basis`) the input vectors expressed in
the rotated basis

## LatticeGas functions and methods

### lg.add_interaction()
Add a coupling shell to the model, on top of any interactions already added (repeated calls accumulate).

Optional arguments:

- Jij: list of shell couplings, e.g. `[J1,J2,J3]` for first/second/third neighbor $J_{ij} n_i n_j$ repulsion (or attraction, for negative values); passed through to `Geometry.get_hamiltonian(tij=Jij)`, so anything that constructor accepts for `tij` works here too

Modifies `lg` in place, no return value

### lg.set_filling()
Reset the occupation array `lg.den` to a new random configuration with a given filling fraction, discarding the current snapshot (e.g. to re-seed a fresh anneal, see `examples/latticegas/optimize/main.py`).

Arguments:

- filling: fraction of sites occupied (rounded to the nearest integer site count)

### lg.get_energy()
Evaluate the total energy $\sum_i \mu_i n_i + \sum_{ij} J_{ij} n_i n_j$ of the current occupation snapshot `lg.den`. Returns a scalar.

### lg.optimize_energy()
Anneal `lg.den` towards a low-energy configuration with a Metropolis discrete-swap optimizer: at each step, 1-3 random occupied/empty site pairs are swapped (preserving the total filling) and accepted unconditionally if the energy does not increase, or with probability $e^{-\Delta E/T}$ otherwise. Only the energy change of each swap is computed, not the full energy, so the cost of a step is set by how many neighbors a site has rather than by the size of the system.

Optional arguments:

- temp=0.1: Metropolis temperature (higher accepts more uphill moves; anneal by calling this repeatedly with decreasing `temp`, or see `lg.anneal()` below). `temp=0` runs zero-temperature (greedy) dynamics, where an uphill move is never accepted
- ntries=1e5: number of swap attempts
- resync_every=1000: how often (in swap attempts) to recompute the energy from scratch, bounding floating-point drift in the incremental tracking
- patience=None: if set, stop early once this many attempts have passed without a new best energy being found (the returned array is truncated to what actually ran)
- checkpoint_at=None: an int or iterable of ints; captures a copy of `lg.den` after that many attempts (1-indexed) into `lg.checkpoints` (a dict `step -> den` snapshot), independent of the final configuration, e.g. to inspect or animate how the configuration evolves partway through a run

Overwrites `lg.den` with the final configuration and returns the array of energies recorded at each attempt (whether or not it was accepted)

### lg.anneal()
Simulated annealing over a decreasing temperature schedule, running `lg.optimize_energy()` once per temperature in `temps`. The best configuration seen anywhere in the schedule is kept, since a high-temperature step can wander back up in energy before it ends.

Optional arguments:

- temps=None: sequence of temperatures, high to low; defaults to a 10-step geometric schedule from 2.0 down to 0.05 (any entry, including 0, is passed straight through to `lg.optimize_energy()`)
- ntries=1e4: number of swap attempts per temperature
- checkpoint_at=None: an int or iterable of ints; like `lg.optimize_energy()`'s `checkpoint_at`, but numbered continuously across the whole schedule (e.g. step 120 is the 20th attempt of the 3rd temperature stage if `ntries=100`), so a snapshot can be recovered after any number of annealing steps, not just the final/best configuration
- any other keyword accepted by `lg.optimize_energy()` (e.g. `patience`, `resync_every`), applied at every temperature

Overwrites `lg.den` with the best configuration found and returns the concatenated energy trajectory across all temperatures

### lg.optimize_energy_multistart()
Run `nstart` independent anneals from independent random seeds at the current filling, and keep the lowest-energy result, which reduces the risk of a single anneal settling into a metastable configuration. Each restart is a full `lg.optimize_energy()` run; the restarts run in parallel once `parallel.set_cores()` has been called, and serially by default (see "Parallelism and reproducibility").

Optional arguments:

- nstart=10: number of independent restarts
- any other keyword accepted by `lg.optimize_energy()` (e.g. `temp`, `ntries`, `patience`), applied identically to every restart

Overwrites `lg.den` with the best configuration found and returns its energy (a scalar)

### lg.optimize_grand_canonical()
Grand-canonical Metropolis sampling/annealing: instead of swapping pairs at fixed filling, single sites are flipped (occupied $\leftrightarrow$ empty) and accepted/rejected the usual Metropolis way, so the total filling fluctuates under `lg.mu` rather than being conserved. This is the standard lattice-gas MC move set, useful for scanning a phase diagram vs. chemical potential, or for equilibrium sampling at one fixed temperature (see `latticegas.get_specific_heat()`/`get_susceptibility()` below). Unlike `lg.optimize_energy()`, it does not need both occupied and empty sites to start from: `lg.den` can start uniformly empty or full.

Optional arguments: same as `lg.optimize_energy()` (`temp`, `ntries`, `resync_every`; no `patience`), except that `temp` defaults to `1.0` here rather than `0.1`. Grand-canonical sampling is usually wanted at a temperature, not as an anneal

Overwrites `lg.den` with the final configuration and returns `(es, ns)`: the energy trajectory and the filling (occupied-site count) trajectory, both arrays of length `ntries`

### latticegas.get_specific_heat() / latticegas.get_susceptibility()
Thermodynamic quantities from a trajectory sampled at one fixed temperature, as produced by `lg.optimize_energy()` or `lg.optimize_grand_canonical()` run at constant `temp` rather than annealed. `get_specific_heat(es, temp, burn=0.2)` gives $C=\mathrm{Var}(E)/T^2$ from an energy trajectory, and `get_susceptibility(ns, temp, burn=0.2)` gives $\mathrm{d}N/\mathrm{d}\mu=\mathrm{Var}(N)/T$ from a filling trajectory. The second is only meaningful in the grand canonical case, where the filling is free to fluctuate. `burn` is the leading fraction of the trajectory discarded as equilibration.

### lg.add_tensor()
Add a custom coupling $J_{ij}=\mathrm{fun}(r_i,r_j)$ between every pair of sites, for interactions beyond `add_interaction()`'s fixed neighbor shells, e.g. a screened or dipolar $1/r^n$ form. Scalar analog of `classicalspin.SpinModel.add_tensor` (which returns a 3x3 tensor per pair); self-pairs are skipped, and pairs where `fun` evaluates to (near) zero are dropped.

Modifies `lg` in place, no return value

### lg.regroup()
Merge duplicate interaction-pair entries accumulated from repeated `add_interaction()`/`add_tensor()` calls (e.g. overlapping neighbor shells added twice), summing their couplings. A performance cleanup only; it does not change `lg.get_energy()`.

Modifies `lg` in place, no return value

### lg.write() / lg.read()
Save/load the current occupation snapshot `lg.den` to/from a text file, the same checkpoint pattern as `classicalspin.SpinModel.write()`/`load_magnetism()`. `write()` forces `nrep=1` (no periodic replication) by default so `read()` round-trips exactly regardless of `lg.geometry.dimensionality`; `read()` raises `ValueError` if the file's site count does not match `lg.nsites`.

Optional arguments:

- name="DENSITY.OUT": file path

### lg.get_local_energy() / lg.get_local_mu()
Per-site breakdown of the current snapshot's energy. `get_local_energy()` returns each site's own contribution $\mu_i n_i + \sum_j J_{ij} n_i n_j$, summed over its interaction neighbors $j$ (`lg.get_energy()` itself counts every bond twice, once from each endpoint, so the values here sum exactly to `lg.get_energy()`, not to half of it); `get_local_mu()` instead evaluates that same expression with site `i` forced occupied, i.e. the energy cost/gain of occupying site `i` given its neighbors' current state.

Optional arguments:

- normalize=False: divide each site's value by the total coupling weight of its own interaction terms

Returns an array over sites

### lg.get_correlator()
Neighbor-shell density-density correlator of the current snapshot `lg.den`, useful for detecting ordered ground states (e.g. after `optimize_energy`). `n` sets how many neighbor shells to return, `normalized` whether to divide out the mean occupation.

Returns `(distances, correlators)`, arrays of matching length

### lg.get_structure_factor()
Reciprocal-space structure factor $S(q)=|\sum_i (n_i-\bar n) e^{-iq\cdot r_i}|^2/N$ of the current snapshot `lg.den`, evaluated directly on the real-space site positions, the reciprocal-space companion to `lg.get_correlator()`: where the neighbor-shell correlator tells you the ordering length scale, $S(q)$ tells you the ordering wavevector. Subtracting the mean occupation makes $S(q=0)=0$ identically, so a peak elsewhere in $q$ is what signals order.

Optional arguments:

- qpath=None: explicit array of $q$ vectors to evaluate; if omitted, a default square grid of `nq`$\times$`nq` points spanning $\pm$`qmax` is used
- nq=60: grid resolution when `qpath` is not given
- qmax=None: half-width of the default grid; defaults to $2\pi/d$ set by the nearest-neighbor spacing $d$

Returns `(qpath, sq)`: the array of $q$ vectors evaluated and the matching array of $S(q)$ values


## LatticeIsing functions and methods

### li.add_interaction()
Add a coupling shell to the model, on top of any interactions already added (repeated calls accumulate).

Optional arguments:

- Jij: list of shell couplings, e.g. `[J1,J2,J3]` for first/second/third neighbor $-J_{ij}s_is_j$ exchange; positive is ferromagnetic (opposite sign convention from `LatticeGas.add_interaction()`). Passed through to `Geometry.get_hamiltonian(tij=Jij)`, so anything that constructor accepts for `tij` works here too

Modifies `li` in place, no return value

### li.add_field()
Add an external (Zeeman-like) field to `li.b`.

Arguments:

- h: a scalar (applied uniformly to every site) or a per-site array

Modifies `li` in place, no return value

### li.set_magnetization()
Reset the spin array `li.s` to a new random $\pm1$ configuration with a given average magnetization, discarding the current snapshot.

Arguments:

- m=0.0: target average magnetization in $[-1,1]$ (rounded to the nearest integer up-spin count)

### li.get_energy()
Evaluate the total energy $-\sum_i b_i s_i - \sum_{ij} J_{ij} s_i s_j$ of the current spin snapshot `li.s`. Returns a scalar. Since `li.pairs` lists both directions of every bond, this is twice the usual sum-over-unordered-bonds convention.

### li.get_magnetization()
Return $\mathrm{mean}(s)$, the average magnetization per site of `li.s` (a scalar in $[-1,1]$).

### li.optimize_energy()
Single-spin-flip Metropolis dynamics, the standard Ising Monte Carlo move set: at each step, one random site is flipped and accepted unconditionally if the energy does not increase, or with probability $e^{-\Delta E/T}$ otherwise. Magnetization is *not* conserved (it fluctuates under `li.b`), the spin analog of `LatticeGas.optimize_grand_canonical()`.

Optional arguments:

- temp=1.0: Metropolis temperature; `temp=0` runs zero-temperature (greedy) dynamics
- ntries=1e5: number of flip attempts
- resync_every=1000: how often (in flip attempts) to recompute the energy from scratch, bounding floating-point drift in the incremental tracking
- checkpoint_at=None: an int or iterable of ints; captures a copy of `li.s` after that many attempts (1-indexed) into `li.checkpoints` (a dict `step -> s` snapshot)

No `patience` option: unlike `optimize_conserved()`, this trajectory is meant to be fed to `latticegas.get_specific_heat()`/`get_susceptibility()`, and early truncation would silently bias those variance estimates.

Overwrites `li.s` with the final configuration and returns `(es, ms)`: the energy trajectory and the total-magnetization ($\sum_i s_i$) trajectory, both arrays of length `ntries`

### li.optimize_conserved()
Kawasaki spin-exchange dynamics: at each step, one up spin and one down spin are picked at random and swapped, which conserves the total magnetization, the spin analog of `LatticeGas.optimize_energy()` (swap-based, fixed filling). Raises `ValueError` if `li.s` does not have both $+1$ and $-1$ present (e.g. after `set_magnetization(1.0)`).

Optional arguments: same as `li.optimize_energy()`, except that `temp` defaults to `0.1` here rather than `1.0`, plus:

- patience=None: if set, stop early once this many attempts have passed without a new best energy being found (the returned array is truncated to what actually ran)

Overwrites `li.s` with the final configuration and returns the array of energies recorded at each attempt

### li.anneal()
Simulated annealing over a decreasing temperature schedule: calls `li.optimize_energy()` once per temperature in `temps`, keeping the best (lowest-energy) configuration seen across the whole schedule.

Optional arguments:

- temps=None: sequence of temperatures, high to low; defaults to a 10-step geometric schedule from 2.0 down to 0.05
- ntries=1e4: number of flip attempts per temperature
- checkpoint_at=None: like `li.optimize_energy()`'s `checkpoint_at`, but numbered continuously across the whole schedule
- any other keyword accepted by `li.optimize_energy()`, applied at every temperature

Overwrites `li.s` with the best configuration found and returns `(es, ms)`, the concatenated energy and magnetization trajectories across all temperatures

### li.optimize_energy_multistart()
Run `nstart` independent flip-based anneals from independent random seeds (same initial magnetization as the current `li.s`) and keep the lowest-energy result. Each restart is a full `li.optimize_energy()` run, in parallel or serially as for `lg.optimize_energy_multistart()`.

Optional arguments:

- nstart=10: number of independent restarts
- any other keyword accepted by `li.optimize_energy()` (e.g. `temp`, `ntries`), applied identically to every restart

Overwrites `li.s` with the best configuration found and returns its energy (a scalar)

### li.get_local_energy() / li.get_local_field()
Per-site breakdown of the current snapshot. `get_local_energy()` returns each site's own contribution to `li.get_energy()` (values sum exactly to `li.get_energy()`, mirroring `LatticeGas.get_local_energy()`'s $j/2$ correction for the double-counted bonds). `get_local_field()` returns the effective field $h^{\mathrm{eff}}_i=b_i+2\sum_kJ_{ik}s_k$ seen by each site, defined so that flipping $s_i$ costs exactly $2s_ih^{\mathrm{eff}}_i$, the spin analog of `LatticeGas.get_local_mu()`.

Returns an array over sites

### li.get_correlator() / li.get_structure_factor()
Spin-spin correlator and reciprocal-space structure factor of the current snapshot `li.s`. See `LatticeGas.get_correlator()`/`get_structure_factor()` for the argument reference.

### li.add_tensor() / li.regroup()
Same as `LatticeGas.add_tensor()`/`regroup()`.

Modifies `li` in place, no return value

### li.write() / li.read()
Save/load the current spin snapshot `li.s` to/from a text file. See `LatticeGas.write()`/`read()`. `read()` rounds to $\{-1,+1\}$ (sign, treating exact 0 as $+1$) to absorb the text round-trip's floating point noise, and raises `ValueError` if the file's site count does not match `li.nsites`.

Optional arguments:

- name="SPIN.OUT": file path
