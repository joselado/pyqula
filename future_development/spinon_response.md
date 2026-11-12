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
- `ansatz=` holds a hopping pattern: every hopping keeps the phase of the
  ansatz with a real amplitude, and a hopping absent from it stays zero.
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

- **Zeeman field under the bond decoupling.** It raises
  `NotImplementedError` now. The saddle of the paper has a Weiss field
  $\sum_j J_{ij}\vec m_j$ next to the bond field, which is zero in the
  liquid and finite in a field. The projection callback can keep the onsite
  spin part of the mean field with the factor 3/4 that undoes the 4/3 of
  the loop. The test is ready-made: the static RPA response at $q=0$ must
  equal the derivative of the magnetization of that loop with the field.
  With the Weiss field in the loop, the liquid is lost again wherever the
  stiffness is negative, so it should be an option and not the default.
- **Constraint projection of the response.** A Gaussian fluctuation of the
  Lagrange multiplier removes the density from the bubble,
  $\chi_{SS}-\chi_{Sn}\chi_{nn}^{-1}\chi_{nS}$. By spin symmetry $\chi_{Sn}$
  vanishes at zero field, even with inequivalent sites, so this only
  matters together with the item above. It is our own derivation and was
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
- **More ways of holding an ansatz.** `ansatz=` freezes the phases and lets
  every amplitude relax, so a cell with inequivalent sites still dimerizes.
  This was measured with the first-neighbor exchange modulated by 30
  percent along $a_1$ and the uniform real pattern as ansatz. On 36 sites
  the loop converges in 220 iterations to hoppings between 0.000 and 0.645,
  a dimerized state. On 196 sites it does not converge in 600 iterations
  (error 0.25, 486 s). The same cells without the modulation converge in 14
  iterations (36 sites) and in 15 s (196 sites). So for a large cell with
  inequivalent sites only the cost was shown to be fine, not the mean field.
  Two things would help: freezing the relative amplitudes too (one global
  amplitude $c$ with $t_{ij}=cJ_{ij}p_{ij}/2$, stationary for
  $c=\sum J_{ij}\mathrm{Re}(\bar p_{ij}\chi_{ij})/\sum J_{ij}$), and
  symmetrizing the mean field with the symmetry group of the ansatz. Which
  restricted family is the right spin liquid of an inhomogeneous model is a
  physics choice and was left to the maintainer. A small library of named
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

## Where to start next

Holding the amplitudes of an ansatz is the first step for cells with
inequivalent sites, which is the case the 200-site condition is about. The
Zeeman field under the bond decoupling, with the static test, is the
smallest step and makes the feature complete for the existing spinon
examples. The structure factor at a lattice momentum is the one a user with
a large cell will ask for after that.
