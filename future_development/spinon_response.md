# Spinon spin response: what was built and what was left

This file records the work of 12 November 2026 on the dynamical spin
response of the spinon mean field (`spinon.SpinonHamiltonian`). The question
was how to go beyond the response of free spinons, with the condition that
whatever is built still works for unit cells of about 200 sites. The first
part says what exists now and what was measured. The second part lists the
suggestions that were not built, with what is known about each one.

## What exists now

- `decoupling="bond"` of `SpinonHamiltonian.get_mean_field_hamiltonian()`
  keeps the bond channel of the exchange alone, $t_{ij}=J_{ij}\chi_{ij}/2$
  with $\chi_{ij}=\sum_s\langle f^\dagger_{is}f_{js}\rangle$, and no
  magnetic mean field. This is the saddle point of Willsher and Knolle,
  arXiv:2503.13831. It runs in the Hartree-Fock engine through a
  `callback_mf` hook of `VJinteraction`: the mean field is projected on its
  spin-singlet part and the exchange is multiplied by 4/3, since the singlet
  Fock term of $J\vec S_i\cdot\vec S_j$ is $3J\chi_{ij}/8$.
- `ansatz=` holds a hopping pattern. With `hold="phases"` every hopping
  keeps the phase of the ansatz with its own real amplitude, and a hopping
  absent from it stays zero. With `hold="pattern"` the hoppings are
  $t_{ij}=cJ_{ij}P_{ij}$ with one amplitude, the stationary point of the
  bond functional in that family. In the form of the code it is
  $c=\sum\mathrm{Re}(\bar P_{ij}s_{ij})/\sum J_{ij}|P_{ij}|^2$, with
  $s_{ij}=-(gJ_{ij}/2)\bar\chi_{ij}$ the singlet mean field, that is
  $c=-\frac{g}{2}\sum J_{ij}\mathrm{Re}(P_{ij}\chi_{ij})/\sum J_{ij}|P_{ij}|^2$,
  with $g$ the `gutzwiller` factor and both sums over the bonds. There is
  no conjugate on $P$, because $t_{ij}$ multiplies $f^\dagger_if_j$, whose
  average is $\chi_{ij}$. The hopping energy at this $c$ is
  $-4c^2\sum J_{ij}|P_{ij}|^2/g$, so
  $c^2=-g\,E_{\rm hop}/(4\sum J_{ij}|P_{ij}|^2)$. At $|P|=1$ and $g=1$ this
  is $c^2=-E_{\rm hop}/(4\sum_{\rm bonds}J_{ij})$, which was checked on a
  modulated chain of four sites from the converged state alone, to
  $10^{-10}$. The energy returned is half of the hopping energy,
  $-2c^2\sum J_{ij}|P_{ij}|^2/g$ (see the repairs of 14 November below).
- `weiss=` adds the Weiss field $\sum_jJ_{ij}\langle\vec S_j\rangle$ to
  the bond loop, by default only with a Zeeman field. The callback keeps the
  onsite spin part of the Hartree-Fock mean field with the factor 3/4 that
  undoes the 4/3 of the loop. For this the `callback_mf` of `VJinteraction`
  acts on the newly computed mean field only, not on the mixed one: a
  rescaling applied at both places is applied twice, and the loop then
  never converges. Two exact checks pass on the chain: the static RPA at
  $q=0$ is the derivative of the moment of the loop with the field
  ($-0.71704$ against $-0.71703$), and the uniform transverse mode is at
  the bare Zeeman energy for $J=1$ and $J=0.5$, where it is 50 percent
  higher without the Weiss field in the loop. Both also hold for a field
  in a generic direction (the moment is $4\chi\vec b$ to $4\times10^{-8}$)
  and with `gutzwiller=2`. The response has to take the temperature of the
  loop: with the default `T=delta` the peak of the $\pi$-flux state moved
  from 0.200 to 0.195. Three more things were seen. A field on a liquid with
  a negative stiffness drives the loop to a magnetic order: 36 sites with
  the uniform pattern ($-0.85$ at K) gave moments from $-0.88$ to $0.69$.
  With `weiss=True` and no field, a small staggered seed orders the
  two-site chain and decays on the $\pi$-flux state, as the stiffness says,
  but a seed of 0.1 takes the $\pi$-flux state at $J_2=0.125J_1$ (stiffness
  0.04 at M) to fully polarized sublattices: the stiffness is a linear
  stability, and the classical state is always another solution of the
  Weiss loop. A 36-site $\pi$-flux cell with $J_1$ and $J_2=0.11J_1$
  modulated by 20 percent along both cell vectors,
  $1+0.1[\cos(2\pi u_1)+\cos(2\pi u_2)]$ with $u$ the fractional
  coordinates of the bond measured from a point that is not a symmetry
  point of the lattice, so that the 36 sites are all different, held with
  `hold="pattern"` (`nk=3`, `T=0.02`),
  has a positive stiffness at the five momenta that were tried (smallest
  0.060, at $q=0$), and in a field $b=0.1$ has 36 different moments between
  $-0.0128$ and $-0.0074$ and the transverse peak at the Zeeman energy 0.20
  (`examples/spinon/spin_liquid_large_cell/main.py`, 16 s for the whole
  example). The first version of this example modulated the Cartesian $x$
  of the bond, which is not periodic in this cell, and the numbers it gave
  (smallest stiffness 0.059, moments from $-0.016$ to $-0.004$) are void;
  see the repairs of 14 November below.
- `gutzwiller=` multiplies the exchange in the mean field and in the vertex.
  The paper uses 2.
- The spin response of a bond state is dressed in the site basis with the
  vertex $J(\vec q)$, $\chi=\chi_0[1+J(\vec q)\chi_0]^{-1}$ in the usual sign
  convention. The Hamiltonian carries `site_spin_vertex=True`, which makes
  `chitk.spinchi._use_pair_basis` return False, so `get_spinchi_full`,
  `get_spinchi_ladder` and `get_magnon_bands(method="rpa")` all take this
  vertex. The explicit pair-ladder calls raise on a bond state.
- `get_rpa_stiffness(q)` is the smallest eigenvalue of the static kernel.
- `chitk/localspinchi.py` is a bubble for the local spin operators that
  builds the matrix elements from the amplitudes on each site. It agrees
  with `chitk.chiAB` to $10^{-15}$ and it is the one that reaches 200 sites.
  `SpinonHamiltonian.get_spinchi_full` uses it.

## What was measured

Prefactors, against arXiv:2503.13831 (TeX source read, Methods and
Supplementary Note 1), on the $\pi$-flux state of the triangular lattice held
with `ansatz=`:

| quantity | here | paper |
| --- | --- | --- |
| $t/J_1$ | 0.1976 | 0.198 |
| static bubble at K, $\Pi^0_K J_1$ | 0.389 | 0.389 |
| static bubble at M, $\Pi^0_M J_1$ | 0.425 | 0.425 |
| stiffness zero at K | $J_2/J_1=0.0716$ | 0.07158 |
| stiffness zero at M | $J_2/J_1=0.1758$ | 0.1758 |

The paper writes the kernel as $1+2J_{\vec q}\chi^0$. With the bubble
normalized per site, as in pyqula, its own critical couplings are those of
$1+J(\vec q)\chi_0$, with $J(\vec q)=\sum_\delta J_\delta
e^{i\vec q\cdot\vec\delta}$ over all the neighbors. So the vertex here is
$J(\vec q)$, not $2J(\vec q)$. The spinon exciton at K is at $0.189J_1$ for
$J_2=0.09J_1$ and at $0.321J_1$ for $J_2=0.125J_1$, without the Gutzwiller
factor.

What the default Hartree-Fock decoupling does, and why it was not the
starting point:

- On the chain the free spinon continuum has its lower edge at
  $0.479J|\sin q|$, and the exact one is at $(\pi/2)J|\sin q|$.
- The free spinon response has a total moment of 0.121 per component, half
  of the 1/4 of a spin one half. Empty and doubly occupied sites carry the
  other half.
- The pair-basis ladder (the default `RPA=True` of a Hartree-Fock state) is
  a response around an unstable state. Its static kernel is negative for
  $q>0.4$ on the chain, independently of the broadening and of the k-mesh.
  With two sites per cell the same loop goes to the classical Néel state,
  $-0.250J$ per site against $-0.153J$ for the uniform one.
- On the triangular lattice six random seeds all converged to weakly coupled
  chains (hopping 0.23 along one direction, 0.03 along the others).
- The ladder reduces the density response by a factor of 7 to 70 against the
  bubble on the chain. It does not go to zero, and the reason was not found.

What the bond decoupling does without an ansatz:

- The chain with two sites per cell dimerizes completely, $-J/4$ per site
  against $-2J/\pi^2$ for the uniform state. This is the known mean-field
  ground state of the bond functional.
- On the triangular lattice a guess with a small time-reversal breaking
  goes to a state with complex hoppings, at $-0.2404J$ per site against
  $-0.2341J$ for the $\pi$-flux state. From an exactly real guess the loop
  can stay on the $\pi$-flux state, so this depends on the numerical noise.
- At zero temperature with the Dirac points on the k-mesh the held state
  gets a staggered Lagrange multiplier of order $1/n_k$ (0.009 at
  $n_k=20$). A temperature of $0.01J$ in the loop removes it.
- `no_magnetism` alone is not enough to hold a spin liquid, for the two
  reasons above.

Cost at 196 sites (triangular supercell), on a workstation:

| step | time |
| --- | --- |
| held bond mean field, 4 k-points | 15 s |
| free bond loop from a random guess | not converged in 300 iterations, 236 s |
| old bubble (`chitk.chiAB`), $S_z$ only, 1 k-point, 4 energies | 84 s, 3.5 GB |
| new bubble, 1 k-point, 81 energies | 20 s |
| new bubble with the RPA, 1 k-point, 81 energies | 30 s |
| new bubble with the RPA, 9 k-points, 81 energies | 197 s, 2.5 GB |
| stiffness at one q, 9 k-points | 10 s |

## Not built

The first group stays inside the mean-field machinery and scales to large
cells. The second group does not scale, or is a new formalism.

### Inside the mean-field machinery

- **The longitudinal response in a field.** With a Zeeman field the state
  is polarized, and a longitudinal perturbation changes the bond field and
  the Lagrange multipliers at first order. The spin vertex alone does not
  have this coupling, so the dressed $S_z$ response in a field is
  approximate. The transverse one is not affected. The total energy of a
  bond loop with a Weiss or a Zeeman field raises `NotImplementedError`,
  because the bond functional that the energy is now (see the repairs
  below) has no term for the Weiss field, and no functional of that loop
  was written.
- **Constraint projection of the response.** A Gaussian fluctuation of the
  Lagrange multiplier removes the density from the bubble,
  $\chi_{SS}-\chi_{Sn}\chi_{nn}^{-1}\chi_{nS}$. By spin symmetry $\chi_{Sn}$
  vanishes at zero field, even with inequivalent sites, so this only
  matters in a field, together with the item above. It is our own derivation and was
  not checked against a paper.
- **Popov-Fedotov chemical potential.** The imaginary potential $i\pi T/2$
  on each site cancels the empty against the doubly occupied states in the
  partition function, which is exact for decoupled spins. It is not the free
  addition it was first taken for. The occupation becomes complex,
  $n(\epsilon)=f_{T/2}(\epsilon)+i/(2\cosh\beta\epsilon)$. For a
  particle-hole symmetric spectrum on a bipartite lattice the imaginary part
  cancels in $\chi_{ij}$, and the result is the usual loop at half the
  temperature, which doubles the mean-field transition temperatures; this
  is what the abstract of cond-mat/0609239 reports, and cond-mat/0509724 is
  the mean-field paper to read first (both seen in a search, not opened).
  On a frustrated lattice or in an inhomogeneous cell the mean field becomes
  non-Hermitian, and no reference for that loop was found. It changes
  nothing at zero temperature.
- **Structure factor at a momentum of the lattice.** For a supercell the
  response at a momentum $\vec Q$ of the underlying lattice is the plane
  wave contraction $\frac1N\sum_{ij}e^{-i\vec Q\cdot\vec r_i}
  \chi_{ij}(\vec q)e^{i\vec Q\cdot\vec r_j}$, with $\vec q$ the same
  momentum in the vectors of the supercell. The guide figure
  (`documentation/figures/pseudofermions_spinliquid.py`) does it by hand for
  two sites, and the convention was checked there (the contraction is an
  eigenvector of the bubble). A method that takes $\vec Q$ would be the
  natural way to compare a 200-site cell with neutron scattering.
- **More ways of holding an ansatz.** `hold="phases"` lets every amplitude
  relax, so a cell with inequivalent sites dimerizes. This was measured
  with the first-neighbor exchange modulated by 30 percent along $a_1$ and
  the uniform real pattern as ansatz. On 36 sites the loop converges in 220
  iterations to hoppings between 0.000 and 0.645, a dimerized state. On 196
  sites it does not converge in 600 iterations (error 0.25, 486 s).
  `hold="pattern"` was built for this: the same two cells converge in 20
  iterations (1 s) and in 23 iterations (28 s), with $t_{ij}/J_{ij}=0.162$
  and 0.164 on every bond, the occupation right to $10^{-5}$ and Lagrange
  multipliers that differ by 0.09 between the sites. What is left is the
  family itself: one amplitude times $J_{ij}$ is the simplest choice, and
  it is not self-consistent bond by bond. Symmetrizing the mean field with
  the symmetry group of the ansatz, or one amplitude per class of bonds,
  are the next options. A small library of named
  patterns (uniform, $\pi$-flux of the square and triangular lattices) would
  spare the user the sign bookkeeping of the guide snippet.
- **The fast bubble everywhere.** `get_magnon_bands` and
  `get_spinchi_ladder` still go through `chitk.chiAB`, which was 84 s per
  k-point at 196 sites. `chitk/localspinchi.py` gives the same numbers and
  could serve `chitk.spinchi.spinchi_full` and the pole search for any
  spinful normal-state Hamiltonian on the CPU. The pole search was not timed
  at 200 sites.
- **Convergence of the loop.** The held $\pi$-flux state needs about 150
  iterations at `mix=0.5`, with an error that decays by 4 percent per step.
  The slow mode was not identified; the proportional step of the Lagrange
  multipliers is the first suspect.
- **KPM and sparse paths.** The bond projection handles dense matrices and
  `integration="ed"` only.
- **Z2 ansatz.** A spinon pairing next to the hopping needs the Nambu
  spinon Hamiltonian that the module docstring lists as missing, and two
  more Lagrange multipliers for $\langle f_\uparrow f_\downarrow\rangle=0$
  (from memory of the SU(2) formulation, no paper opened). The response then
  needs the BdG bubble.
- **Sum-rule diagnostics.** The total moment against 1/4 and the first
  moment against the bond energy are the cheapest measure of what a spectrum
  misses. Nothing prints them now.
- **Gutzwiller factor.** The paper uses 2, from the local moment (3/8
  against 3/4). The Gutzwiller approximation of the $t$-$J$ model gives 4
  for the exchange (seen in a search result for arXiv:0707.1020, not
  opened). With the bond decoupling the lower edge of the chain continuum is
  $(2/\pi)gJ|\sin q|$, so $g=2$ gives 1.27 against the exact $\pi/2$, and
  $g=4$ would give 2.55. No factor is applied to the weight.

### Beyond mean field

- **Exact projection on a small cluster.** Enumerate the spin
  configurations (12870 for 16 sites), take each amplitude as a product of
  two determinants, and diagonalize the Heisenberg model in the projected
  particle-hole states of each momentum. This removes the charge
  fluctuations completely and changes the shape of the spectrum. It stops at
  about 24 sites, so it is a benchmark for the rest and not a method for
  large cells. On a ring Lanczos gives the exact answer to compare with.
- **Variational Monte Carlo.** The same construction with sampling reaches
  hundreds of sites: arXiv:1803.02359 for the chain (abstract read), and
  arXiv:1903.05691 for the triangular lattice (search result only), which is
  the benchmark the paper above compares with. A new formalism for pyqula.
- **Schwinger bosons with Gaussian fluctuations.** The choice for ordered
  states, where magnons are bound states of two spinons, arXiv:1802.06878
  (abstract read).
- **The pair-basis ladder on a spinon state.** It is the conserving
  response of the Hartree-Fock saddle, and that saddle is unstable (see
  above). It also grows to about 5600 pair operators for 200 sites with six
  neighbors.
- **Gauge sector.** Monopoles and the low-energy response of a Dirac
  liquid are outside any of the options above. The paper adds them by hand
  through a monopole susceptibility.

The guide figure was checked at a second resolution: between `nk=31` and
`nk=45` the response changes by less than 0.03 away from M. At M itself the
peak moves by one energy step (0.24 to 0.21) and by 5 percent in height,
and it does not move between `nk=61` and `nk=91`. The stiffness at
$J_2=0.125J_1$ was scanned on a 12 x 12 mesh of momenta: it is positive
everywhere, with its minimum 0.042 at M (`nk=31`).

## Repairs of 14 November 2026

These come from the seventh bug audit (`bug_audit_7.md`).

- **The exchange must be reciprocal.** $\vec S_i\cdot\vec S_j$ is symmetric,
  so no spin model has $J[\vec d]\neq J[-\vec d]^T$. The 36-site example
  had one: its modulation was a cosine of the Cartesian $x$ with period
  $L=6$, and a translation by $a_2'=(-3,3\sqrt3)$ shifts $x$ by $-L/2$, so
  the bonds that cross the $a_2'$ boundary were 0.4 apart seen from their
  two ends. The Hartree-Fock field averaged the two directions, while
  `hold="pattern"` and the RPA vertex took one each, and the converged
  Hamiltonian of `hold="pattern"` was not Hermitian, by 0.079.
  `get_mean_field_hamiltonian` now builds the exchange of every bond once
  and raises a `ValueError` when $|J[\vec d]_{ij}-J[-\vec d]_{ji}|$ is above
  $10^{-8}$ times the largest exchange, for both decouplings and every
  hold, naming the worst bond. The test is a ramp $1+0.1x$ on a chain of
  four sites, symmetric in its two arguments and not periodic (mismatch
  0.4). A second test calls the check itself, with a direction whose
  opposite is absent, a cell matrix that is not symmetric and the two
  sides of the relative tolerance; without it a check that skipped a
  missing direction, or one with an absolute tolerance, passed every test.
  The trimer exchange of the Kondo spectrum example and its test,
  and the four-site cosine of the pattern test, are reciprocal and still
  run. The example now modulates the fractional coordinates of the bond
  along both cell vectors (numbers above). With the origin of the
  modulation at the center of the cell the exchange keeps the inversion
  and the two mirrors that exchange the cell vectors, and only 12 of the
  36 sites are different (6 in the first version). The origin is now
  displaced by $a_1/6+a_2/3$ of the lattice, which leaves 36 different
  sites and moves the stiffness by $10^{-6}$. The generic `Vr`/`Jr` of
  `VJinteraction` and `Vinteraction` still averaged a function that is
  not reciprocal, with no error; it raises too since the decisions below,
  from one check in the builder, and the spinon path has no check of its
  own any more.
- **The energy of the bond loop is the bond functional.** The engine
  returned its Hartree-Fock energy, $-2A+B'$ in the notation of the audit,
  with $A=\sum_b2|t_b|^2/(gJ_b)$ and $B'$ its double counting, which is the
  value of nothing when $\chi_b$ is not proportional to $J_bP_b$. The
  energy is now $E_{\rm band}+\sum_b2|t_b|^2/(gJ_b)$, with $E_{\rm band}$
  the hopping energy of the engine at zero temperature with the Lagrange
  multipliers undone, for the free loop and for both holds. At a free
  fixed point it is the old number bond by bond (the dimer $-J/4$, the
  uniform chain $-2J/\pi^2$ and the $\pi$-flux state against the chiral
  one did not move). On a chain of two sites with $J=1.3$ and $0.7$ held
  with `hold="pattern"` (`nk=100`, `T=0.01`, gap 0.42) the new energy is
  $-0.48726=-4c^2/g$ to $10^{-10}$ and the old one was $-0.39327$; with
  `gutzwiller=2` they are $-0.97452$ and $-0.78654$. The difference,
  $B'-A=0.0940$, is the one of the closed form of the dimerized chain at
  $t_2/t_1=0.538$, where $|\chi|=0.923$ and $0.280$ on the two bonds. With
  `hold="phases"` and a twisted $\pi$-flux ansatz the old energy was
  $8\times10^{-5}$ above the functional. The new energy is half of the
  hopping energy to $4\times10^{-6}$ at `T=0.01` and to $2\times10^{-8}$ at
  `T=0.005`, since the hopping energy is summed at zero temperature.
  $E_{\rm band}$ is the band energy of the engine before its double
  counting, which `scftk/spinspin.py` now keeps as `scf.band_energy`, so no
  diagonalization is added; the 41 tests of `tests/spinon` pass with it as
  with the energy rebuilt from the converged Hamiltonian. The multiplier is
  undone as the engine does it, $+2\sum_i\lambda_i
  n_i$ with $n_i=1/2$ per spin: on the uniform state of the triangular
  lattice ($\lambda=-0.14gJ$) the old and the new energies agree to
  $10^{-12}$, and a uniform onsite energy $\epsilon$ moves the multiplier
  and the energy by $\epsilon$ per site. On a metal this zero-temperature
  sum also carries $\lambda(N-N_0)$, with $N_0$ the number of negative
  levels of the mesh at zero temperature: $3\times10^{-4}$ there at
  `nk=30`, as before the change.

The bubble of `chitk/localspinchi.py` and the moment response of
`kondotk/momentchi.py` were made cheaper at the same time, with the same
numbers. All timings are on a shared workstation with two threads, the
smallest of three runs of the old and the new code back to back, so the
ratio is what matters.

- **A running sum over the k-mesh.** The bubble kept one array per
  k-point and per q and then stacked them. It now adds each k-point into
  one array, and the dressing writes into a preallocated one. On 49 sites
  with a field that mixes the spins (9 k-points, 81 energies) the peak
  memory goes from 848 to 475 MB, with the same numbers to the last bit and
  27.4 s against 25.5 s. At 196 sites with a Zeeman or Weiss field and 9
  k-points the old code needs about 8 GB for the list alone. A smaller case
  of the same kind ran on a compute node with one thread: 98 sites in a
  field that mixes the spins, 49 k-points and 61 energies. The peak memory
  goes from 8.1 GB to 1.3 GB, the time stays at 1335 s, and the two arrays
  are the same to the last bit.
- **The collinear route.** A Hamiltonian that does not mix the spins but
  is different for them (a Zeeman field along $z$, the Zeeman and Weiss
  path of the bond loop) now diagonalizes the two spin blocks apart and
  contracts four $N\times N$ blocks, $C^{ss'}$, instead of $3N$ columns. A
  field in another direction was not rotated to $z$ and took the general
  route; it takes the collinear one in its own spin frame since the
  decisions below. At 196 sites in a field along $z$ (1 k-point, 3 energies) it goes
  from 27.1 s and 1.71 GB to 4.0 s and 0.78 GB, agreeing to $2\times
  10^{-15}$, and it agrees with `chitk.chiAB` to $10^{-12}$. Timed again
  in the review of the same day, with more load on the machine: 42.8 s
  against 7.2 s, a ratio of 5.9 where the first timing gave 6.8, with the
  same memory.
- **The dressing of a degenerate bubble.** A vertex with the same block
  for the three components and none between them, a structural check that
  the exchange of a bond state passes exactly, now dresses the
  $N\times N$ block alone, with a linear solve instead of an inverse and a
  product. The static kernel takes the eigenvalues of that block, each one
  three times. At 196 sites (1 k-point, 41 energies) the dressed response
  goes from 22.6 s to 13.2 s, and the static kernel from 1.17 s to 0.54 s,
  agreeing to $2\times10^{-14}$; the peak memory was 1.07 GB and 1.16 GB.
  A vertex that tells the components apart, an exchange weaker along $z$
  or an antisymmetric one between $x$ and $y$, must take the
  $3N\times3N$ route. The review added a test of both against
  `chi_ops_RPA`, since a structural check that had lost one of its two
  conditions passed every test before.
- **One diagonalization of the mesh for the local response.** With
  `q=None`, $k+q$ is a point of the mesh, so every point is diagonalized
  once and the states at $k+q$ are taken from it, for `spin_chi` and for
  `moment_spinchi`. On the square Kondo lattice at `nk=8` (64 q and 64 k)
  this gives 7.4 s against 7.1 s with 200 energies, where the contraction
  dominates, and 0.70 s against 0.22 s with one energy. The contraction
  itself still scales as $n_k^{2d}$ times the energies; only a binned
  route through the local spectral matrices would change that, at the
  price of the binning. The test of the moment response dresses every $q$
  with a residual interaction that depends on $q$; with a number alone, a
  loop that dressed every $q$ with the interaction of $q=0$ passed.
- **The bare moment response contracts $S^+$ alone.** Without the residual
  interaction `moment_spinchi` returns only $\chi^{+-}$, so it now contracts
  the matrix element of $S^+$ (`localspinchi.bare_raising_chi`): a third of
  the columns on the general route, $C^{ud}$ alone on the collinear one and
  twice $\chi^{zz}$ on the degenerate one. On a $3\times3$ Kondo cell (9
  moments, `nk=6`, 200 energies) it goes from 2.4 s to 0.93 s in a field
  that mixes the spins and from 2.6 s to 0.48 s in a field along $z$,
  agreeing to $10^{-14}$.
- **The Kondo spectrum of a degenerate state from one spin.**
  `local_spectral_components` diagonalizes the $N\times N$ block of one
  spin and gives each orbital state the weight $2|u_i|^2$. At 196 sites
  (`nk=4`, 401 energies) `kondo_spectrum` goes from 3.9 s to 2.7 s. The
  same half block in the k-loops of `kondotk/cotunneling.py` and of the
  composite spectrum was not done then; it is done since the decisions
  below (`kondo_tunneling.md`). A batched diagonalization of the k-points
  was not done. The batched numba diagonalization of
  `documentation/gpu_porting_plan.md` (2.77 s for 16 matrices of 512 on 8
  threads) is not clearly faster than one threaded LAPACK call per
  k-point at these sizes; this was read there, not measured again.

The new tests are in `tests/spinon/test_spinon_bond_rpa.py` (the raise,
the check itself, the energy of both holds, the multiplier, and an assert
that a converged bond state takes the fast routes of the bubble),
`tests/chi/test_local_spin_chi_routes.py` and
`tests/kondolattice/test_moment_spinchi_routes.py`. What ran after these
changes and their review: `tests/spinon` (41 passed),
`tests/chi/test_local_spin_chi.py` and the routes file (38),
`tests/kondolattice` except `test_kondolattice.py` and
`test_kondolattice_energy.py` (74 passed, 10 skipped by their
parametrization), `tests/kondotunneling/test_spinon_kondo_spectrum.py` (8),
and the collective-mode neighbours `tests/chi/test_kernel_chern.py`,
`tests/chi/test_response_winding.py` and
`tests/magnon/test_magnon_chern_routes.py` (44). The bug tests, five with
their parameters, fail on the code before the change. In the review 19
lines of the repairs and of the new routes were changed one at a time on
a copy of the tree, and each change makes a test fail. Six of them passed
every test before the review and only fail the tests it added: the two of
the reciprocity check, the three of the structural check of the vertex
and the $q$ of the residual interaction.

## Decisions of 14 November 2026

These close what the repairs above left open in this file. Two changed the
code, one is left open with its reason, and one keeps a choice of the
review. All timings are on a shared workstation with two threads, the old
and the new code back to back.

- **`Vr` and `Jr` must be reciprocal in every builder.** One check,
  `specialhopping.require_reciprocal_interaction`, runs where a function of
  two positions becomes an interaction: in
  `specialhopping.distance_cut_interaction` (the mean field, the
  Bethe-Salpeter kernel, the density response and `Jr`) and in the sparse
  builder of `scftk/sparsemeanfield.py`. It raises a `ValueError` that
  names the keyword (`Vr` or `Jr`), the pair, its two values and the
  mismatch, with the relative tolerance $10^{-8}$ of the spinon check. The
  spinon path has no check of its own any more: its `Jr` goes through the
  same builder, before the loop starts. A raise was chosen over a
  symmetrization because what reads the interaction does not agree on a
  function with two values: the Hartree-Fock field averages the two ends
  of a pair, and the Bethe-Salpeter kernel and the $V(q)$ of the RPA take
  one end each, which is not Hermitian. A symmetrized function would also
  be a model that nobody wrote, since the usual cause is a modulation that
  does not have the period of the cell. Measured before the change, on a
  chain of four sites: the ramp $1+0.1x$, symmetric in its two arguments
  and not periodic, gave no warning at all and a kernel of
  `density_interaction` with a mismatch of 0.4; a function modulated with
  the first position alone gave the warning of the hopping builder, whose
  advice (`non_hermitian=True`) is wrong for an interaction. The builder
  silences that warning now, since the raise replaces it. Every `Vr` and
  `Jr` of the tests, the examples, the guide figures and the notebooks is a
  function of the distance or a periodic modulation, and still runs.
  The review of the same day found two holes in this check, both closed.
  The comparison of the two ends takes no conjugate, and the warning that
  the builder silences compared with the conjugate, so an imaginary `Vr`
  (`0.3j` on first neighbors) went through with no message at all, and a
  Hermitian complex one raised with the message of a pair with two
  values. And an infinite value at zero distance was the scale of the
  relative tolerance, so no mismatch was ever above it, while a NaN hid
  the largest one. The builders now call
  `specialhopping.require_pair_interaction`, which asks three things in
  this order: every value finite, every value real, and the same value
  from the two ends. The coupling of $n_in_j$ or of
  $\vec S_i\cdot\vec S_j$ is a real number, so a complex one raises with
  a message that says so. An infinite or NaN value raises where the pair
  of a site with itself is used, as an onsite $U$ in the mean field and
  in the density response. The Bethe-Salpeter kernel does not use that
  pair, so `density_interaction` asks the builder to drop it before the
  check (`onsite=False`): a Coulomb tail without a soft core is taken
  there, as before, and its other pairs are checked.
  `tests/scf/test_vr_reciprocal.py` has 48 tests over nine routes, the
  two sparse mean fields through the call of the engine itself: both
  functions that are not reciprocal, an imaginary and a Hermitian complex
  one, an infinite value at zero distance with and without a mismatch, a
  NaN between two sites, a periodic modulation that every route must
  take (the loops for two iterations), the check itself on dense and
  sparse matrices (moved from the spinon tests), a finite system and a
  Coulomb tail. On the code of 8518a8e 10 of the first 16 failed; the 27
  cases of the complex, infinite and NaN values fail on the tree before
  the review.
- **A field in any direction takes the collinear route.** The route was
  for a field along $z$ only, so the cost depended on a direction that has
  no physical meaning in a spin-rotation invariant model.
  `localspinchi._spin_axis` finds the direction $\hat n$ when every hopping
  matrix depends on the spin through $\hat n\cdot\vec\sigma$ alone. The
  Bloch matrix is then rotated with one $2\times2$ matrix $U$ on the spin of
  every site, $U^\dagger\hat n\cdot\vec\sigma U=\sigma_z$, the four blocks
  are contracted in that frame, and the response is rotated back,
  $\chi_{ab}=\sum_{cd}R_{ac}R_{bd}\chi'_{cd}$ with
  $U^\dagger\sigma_aU=\sum_bR_{ab}\sigma_b$, before the dressing, since the
  vertex is given in the lab frame. With $S_z$ conserved nothing is rotated
  and the code path is the one of before. $\chi^{+-}$ in a rotated frame
  builds the four blocks and combines them after the rotation, since $S^+$
  of the lab is not one pair of spins there. At 196 sites in a field
  $(0.2,-0.1,0.25)$ (1 k-point, 3 energies) the bubble goes from 27.6 s and
  1.77 GB to 4.3 s and 0.86 GB, agreeing to $1.3\times10^{-15}$. A
  Hamiltonian with two spin directions (a Rashba coupling, a field that
  turns from site to site) takes the general route.
  The direction is looked for once per call, and only when the two
  cheaper checks fail, with the hopping matrices read one at a time. The
  first version held the spin coefficients of every hopping matrix at
  once and ran twice: 15 hopping matrices of memory on top of reading
  them on a 72-site honeycomb cell with a Rashba coupling, where the
  search now adds nothing to the peak. On 288 sites the review measured
  149 MB, and the peak is now 69 MB, which is what reading the nine
  hopping matrices costs.
  `tests/chi/test_local_spin_chi_axis.py` has 15 tests, against the general
  route and `chitk.chiAB` for a direction that is not an axis: bare,
  dressed with a vertex that is weaker along $z$ and antisymmetric between
  $x$ and $y$, the static kernel, the local response, $\chi^{+-}$ of a
  subset of sites, a complex spin-dependent hopping along one direction
  (a Kane-Mele coupling rotated by hand), and $\chi_{nn}=\chi_{zz}$ of the
  same field along $z$. Six fail on the code before, the ones that ask for
  the route; the eight that compare numbers pass on both, as a change of
  route must. A bond state converged in a field $(0.06,-0.03,0.07)$ with
  its Weiss field takes the route too, with the frame of the field (an
  assert in `tests/spinon/test_spinon_bond_rpa.py`), so the tolerance
  $10^{-10}$ of the direction holds on a converged state. The tilted
  field was the case of the general route in three older tests, of the
  local response and of $\chi^{+-}$ in `tests/chi/test_local_spin_chi_routes.py`
  and `tests/kondolattice/test_moment_spinchi_routes.py`. They have a
  fourth case now, a field that turns from site to site, which is on the
  general route (7 more tests).
- **The binned local response of `q=None` is not built.** The local
  response contracts every pair of a $k$ and a $q$ of the mesh at every
  energy, so its cost grows as $n_k^{2d}$. From the 7.4 s at `nk=8` on the
  square Kondo lattice (200 energies) this is about 5 minutes at `nk=20`
  and about 6 hours at the default `nk=60` (extrapolated, not run). A route
  through the local spectral matrices on an energy grid scales as $n_k^d$,
  but it is not the same calculation with a faster loop. It covers only
  the bare response, since the dressed one needs every $q$ on its own. It
  changes the numbers at the level of the bin, so it brings a new knob for
  the accuracy. And between different sites it holds one spectral matrix,
  2 x 2 in spin, per pair of sites and per bin, which is 5 GB for 200
  moments and 2000 bins. So it is a new option of the public calls, and it waits for a
  case that needs it: the bare local response of a 2D or 3D lattice with a
  small cell on a mesh beyond about `nk=20`. What changed now is only the
  text: the docstrings and the guide say that the cost of `q=None` grows as
  the square of the number of k-points. Whether the default `nk=60` of
  `get_moment_spinchi` should be smaller when `q=None` in 2D is not
  decided.
- **The modulation of the 36-site example keeps its origin off the
  symmetry points.** The guide points to this example for a cell of 36
  inequivalent sites, and the numbers above belong to this origin. Run
  again after the changes of this section, the example gives the same
  numbers (smallest stiffness 0.060 at $q=0$, moments between $-0.0128$
  and $-0.0074$, peak at 0.20). Counted on the converged state in the
  field: 36 different moments with this origin and 12 with the origin at
  the center of the cell. The price is one line that displaces the origin,
  and its comment says why.

What ran after these changes and their review: the two new files,
`tests/spinon` (40 passed), `tests/chi/test_local_spin_chi.py` and the
routes file (41),
`tests/kondolattice` (88 passed, 10 skipped by their parametrization),
`tests/kondotunneling` (26), the files of `tests/scf`, `tests/bse`,
`tests/chi`, `tests/magnon` and `tests/conductivity` that pass a `Vr` or a
`Jr` or use the bubble, and `tests/documentation` (14). The whole suite was
not run.

## State of the documentation and of the tests

The work is in three commits on master, `c0922af`, `cd5a60d` and `cc23757`
(12 November 2026). Of the five documentation surfaces, three moved: the
guide page `pseudofermions.md` with its figure, the reference entry of
`SpinonHamiltonian`, and two examples
(`examples/spinon/dirac_spin_liquid_rpa`, `examples/spinon/spin_liquid_large_cell`).
There is no README bullet and no notebook for the bond decoupling and its
response, and the PDF of the guide was not rebuilt; the website builds in
strict mode.

The whole suite was not run after these commits. What ran: `tests/spinon`,
`tests/chi`, `tests/kondolattice` and `tests/magnon` after the first commit
(348 passed), the exchange and constrained mean-field tests of `tests/scf`
(110 and 19 passed), and after the last one `tests/spinon`,
`tests/documentation` and five files of `tests/chi` (84 passed).
`tests/spinon/test_spinon_bond_rpa.py` takes 2 to 4 minutes, and its
slowest test 20 to 45 s depending on the load.

## Where to start next

The structure factor at a lattice momentum is the one a user with a large
cell will ask for first. The fast bubble for `get_magnon_bands` is the next
one, since the pole search still uses the old bubble.
