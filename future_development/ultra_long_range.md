# The ultra long-range method in a tight-binding basis: a plan

This roadmap settles how the ultra long-range method, the ultracell, goes into pyqula. It
covers the formalism, what it buys in a tight-binding basis and what it does not, the
stages, the tests, and the decisions left to the maintainer. The method is that of
T. Müller, S. Sharma, E. K. U. Gross and J. K. Dewhurst, Phys. Rev. Lett. 125, 256402
(2020), arXiv:2008.12573, read from the TeX source. defuqumat has the plane-wave version
in its `ultracell` subpackage, which is where the design choices below were compared. The conventions were checked in a scratch prototype on 14 November 2026,
against the explicit supercell of pyqula. Stage 1 was built the same day (see "What stage
1 built"); stages 2 to 4 are not. The next decision point is the list at the end.

## What stage 1 built

`h.get_ultracell(hs,emax=None)` returns an `Ultracell` (`src/pyqula/ultracell.py`, with
the internals in `ultracelltk/`: `basis.py`, `matrix.py`, `observables.py`). It takes the
modulated supercell `hs` and projects it, `P^dagger H_S(k_S) P`, one column block at a time
with an FFT over the cells, so the clean part comes out as the band energies on the
diagonal without any subtraction. Its methods are `get_hk_gen`, `get_kdos_bands`,
`get_dos`, `get_ldos`, `get_vev` and `get_harmonics`, each with the normalization of its
counterpart on the unit cell or the supercell. The cut is `emax=`, every state up to an
energy, with a degenerate level kept whole. Diagonal supercells only; a supercell with a
site removed, a non-Hermitian Hamiltonian and `get_vev` with Nambu raise.

`tests/ultracell/test_ultracell.py` (17 tests) pins: the supercell spectrum with every
state, at 1e-12, for chains with odd and even $n$, spinless and spinful honeycomb with
Rashba coupling, an exchange field and modulated hoppings, a cubic lattice, and a Nambu
honeycomb with a modulated s-wave pairing; the unfolded weight against `operator="unfold"`;
the unit cell's `get_kdos_bands` and `get_dos` at zero modulation; the supercell's
`get_ldos` and `get_vev` (with and without `"sz"` and `"sx"`); the interlacing and the
nested bases of the cut; with a cut, the real-space states against the Rayleigh-Ritz
energies in the supercell, the LDOS summed over the sites against the DOS, the unfolded
spectral function against the kept levels of the unit cell, and the error of `get_vev`
when `emax` is below zero; the helical exchange against `generate_spin_spiral`; and the
harmonics of a cosine. The linear-response test against `densitychi_RPA` is not written,
and belongs with stage 2.

Two facts found while building it. The supercell builders center the positions, so a
function of position is shifted with respect to the cell index $\mathbf R$ that the
harmonics use: the phase of $f_{\mathbf Q}$ depends on that shift and its modulus does not
(stated in the docstring). And the cost is in the per-$\mathbf k_S$ work, not in the
unit-cell diagonalizations: the LDOS and the occupation of a $30\times30$ spinless
graphene ultracell, 1800 states, at four $\mathbf k_S$ each, take about a minute on two
threads, since `get_ldos` and `get_vev` each rebuild and diagonalize the matrix; a cache
of the eigenstates per $\mathbf k_S$ should halve it (an estimate, not measured), at
the price of storing them.

The guide page is `documentation/guide/ultracell.md`, with the figures
`ultracell_unfolding.py` and `ultracell_ldos.py`, the example
`examples/2d/ultracell/main.py`, and the notebook
`jupyter-notebooks/functionalities/single_particle_hamiltonians/14_ultracell_long_range.ipynb`.

## The method, and what it becomes in a tight-binding basis

The question is how to compute a crystal whose potential, exchange field or order
parameter changes slowly over many unit cells, while the atoms stay periodic. The paper
generalizes the Bloch state with a sum over a fine grid of momenta $\boldsymbol\kappa$
around each $\mathbf k$,
$\Phi(\mathbf r)=\sum_{n\boldsymbol\kappa}c_{n,\mathbf k+\boldsymbol\kappa}\,
u_{n\mathbf k}(\mathbf r)\,e^{i(\mathbf k+\boldsymbol\kappa)\cdot\mathbf r}$, so that
the densities and the potentials become Fourier series
$f(\mathbf r)=\sum_{\mathbf Q}f_{\mathbf Q}(\mathbf r)\,e^{i\mathbf Q\cdot\mathbf r}$
with lattice-periodic coefficients. The smallest cell that holds the whole modulation is
the ultracell, $N=n_1n_2n_3$ unit cells.

In a tight-binding basis this is simpler than in DFT, and the simplest way to see it is
to start from the supercell. Take the ultracell as a supercell with Bloch momentum
$\mathbf k_S$, in reduced coordinates of the ultracell. Without the modulation, its
eigenstates are the Bloch states of the unit cell at the $N$ folded momenta
$\mathbf p_{\mathbf q}=(\mathbf k_S+\mathbf q)/\mathbf n$ (componentwise,
$\mathbf q\in\mathbb Z_{n_1}\times\mathbb Z_{n_2}\times\mathbb Z_{n_3}$), so Bloch's
theorem gives them from $N$ diagonalizations of the unit cell. The basis is

$$
|\mathbf q,m\rangle=\frac{1}{\sqrt N}\sum_{\mathbf R}e^{2\pi i\,\mathbf p_{\mathbf q}\cdot\mathbf R}
\sum_\alpha u_{m\alpha}(\mathbf p_{\mathbf q})\,|\mathbf R,\alpha\rangle ,
$$

with $u_m(\mathbf p)$ the eigenvectors of `h.get_hk_gen()` at $\mathbf p$, in pyqula's
lattice gauge (phase $e^{2\pi i\,\mathbf k\cdot\mathbf d}$). For an onsite modulation
$W(\mathbf R)$, a matrix on the orbitals of one cell that may include spin, the matrix to
diagonalize at $\mathbf k_S$ is

$$
H_{\mathbf q m,\mathbf q'm'}=\delta_{\mathbf q\mathbf q'}\delta_{mm'}\,\varepsilon_m(\mathbf p_{\mathbf q})
+u_m^\dagger(\mathbf p_{\mathbf q})\,W_{\mathbf q-\mathbf q'}\,u_{m'}(\mathbf p_{\mathbf q'}),
\qquad
W_{\mathbf s}=\frac1N\sum_{\mathbf R}e^{-2\pi i\,(\mathbf s/\mathbf n)\cdot\mathbf R}\,W(\mathbf R),
$$

meaning that the diagonal is the band structure of the unit cell and the modulation only
couples folded momenta that differ by one of its harmonics. For a general modulation
$\delta H$, hoppings included, the element is $P_{\mathbf q}^\dagger\,\delta H(\mathbf
k_S)\,P_{\mathbf q'}$, with $P_{\mathbf q}$ the folded Bloch states written on the orbitals
of the supercell. An eigenstate goes back to real space with one FFT over $\mathbf q$,
$\Phi(\mathbf R,\alpha)=N^{-1/2}\sum_{\mathbf q m}c_{\mathbf q m}\,e^{2\pi i\,\mathbf
p_{\mathbf q}\cdot\mathbf R}u_{m\alpha}(\mathbf p_{\mathbf q})$.

With every band kept this is the modulated supercell, exactly, rewritten in the eigenbasis
of the clean supercell. So the method has one approximation, the truncation of that basis
to the states of the unit cell in an energy window, and one convergence parameter, the
window. You can think of it as the modulated supercell solved by Rayleigh-Ritz in the
low-energy states of the clean supercell, which Bloch's theorem gives from the unit cell
alone.

The paper and Elk do three things that a tight-binding basis does not need. They keep one
set $u_{n\mathbf k}$ at the central $\mathbf k$ for every $\boldsymbol\kappa$, which the
paper chooses because it makes the densities and the matrix elements cheaper (defuqumat's
notes add that in LAPW a state at $\mathbf k+\boldsymbol\kappa$ cannot be handed to
$H(\mathbf k)$); they then build the diagonal block as $O^\dagger\varepsilon O$ with the
overlap $O$ made unitary by an SVD; and they assume $|\boldsymbol\kappa\cdot\mathbf
r|\ll1$ inside a cell. In a tight-binding basis $H(\mathbf p)$ acts on any vector, and the
band basis at every folded momentum costs one diagonalization of the unit cell. This is
the route defuqumat calls the direct route, the exact Rayleigh-Ritz matrix in the stated
basis. Its blocks depend on $\mathbf q$ and $\mathbf q'$ separately, so the build is $N^2$
products of size $N_b\times N_\mathrm{orb}\times N_b$, against $(N N_b)^3$ for the
diagonalization. The ratio of the two is $N_\mathrm{orb}/(N N_b)$, so the central-k basis,
whose blocks depend on $\mathbf q-\mathbf q'$ only, saves nothing as long as $N
N_b>N_\mathrm{orb}$. For a large unit cell, a moiré cell with $10^4$ orbitals, the
balance changes: the build dominates, and the direct route needs the $N_b$ states of the
window at each of the $N$ folded momenta (ARPACK, $N$ runs), where the central-k basis
needs them at one momentum, at the price of a slower convergence in the window. That
measurement decides the route of stage 3. The paper also uses a $\boldsymbol\kappa$
window of half the $\mathbf Q$ grid, which for an even ultracell never forms one
harmonic; we use the full folded set, and only with it is the method equal to the
supercell.

Three conventions were checked in the prototype and are part of the design. The momenta
are reduced, with the $2\pi$ in the phase, as in `geometrytk/bloch.py`. The modulation is
sampled at the positions of the supercell's own geometry and mapped to $(\mathbf R,\alpha)$
by `unfolding.get_supercell_map`, which is what `h.add_onsite(fun)` sees on the same
supercell, so a function written for a supercell gives the same physics here. The
unfolded weight at $\mathbf p=\mathbf p_0$ (take $\mathbf k_S=\mathbf n\,\mathbf p$) is
$\sum_m|c_{0m}|^2$, and `operator="unfold"` on the supercell gives $N$ times that.

## What was measured

The prototype is three short scripts outside the repository. The error is the largest
difference between eigenvalues of the ultracell matrix and of the dense supercell at the
same generic $\mathbf k_S$. The spinless rows use the harmonic form of the matrix above,
and the spinful row uses the general element $P_{\mathbf q}^\dagger\,\delta H\,P_{\mathbf
q'}$, with $\delta H$ the difference of the modulated and the clean supercell.

| model | orbitals per cell | ultracell | basis | dimension | error |
|---|---|---|---|---|---|
| spinful honeycomb, Rashba, modulated exchange field and modulated hoppings | 4 | 5x3 | every band | 60 | 1e-14 |
| same exchange field, general element against harmonic form, entry by entry | 4 | 5x3 | every band | 60 | 3e-16 |
| spinful chain, helical exchange field, supercell against `generate_spin_spiral` | 2 | 7 | every band | 14 | 1e-15 |
| honeycomb, three-harmonic potential | 2 | 7x1, 6x1, 5x4, 6x6 | every band | 14 to 72 | 1e-14 |
| same, unfolded weight against `operator="unfold"` | 2 | same | every band | 14 to 72 | 1e-13 |
| honeycomb, $0.3\cos$ potential | 2 | 31x1 | $\lvert\varepsilon\rvert<1.5$ | 40 of 62 | 9e-5 for states in $[-1,1]$ |
| triangular 4x4 moiré cell, slow $0.1$ potential | 16 | 9x9 | 1 band | 81 of 1296 | 0.23 |
| same | 16 | 9x9 | 2 bands | 162 of 1296 | 2e-4 for the lowest 81 |
| same | 16 | 9x9 | 6 bands | 486 of 1296 | 5e-6 |
| same | 16 | 9x9 | $\varepsilon<-1.7$ | 150 of 1296 | 3e-4 for states below $-1.9$ |

Two lessons come out of the moiré rows. A cut by band count fails without any warning when
the kept band overlaps the next one in energy: here band 1 spans $[-2.14,-1.72]$ and band 2
$[-1.95,-1.48]$, so one band cannot hold the lowest 81 states. A cut in energy with a margin
of about twice the amplitude of the modulation works, which is what perturbation theory
suggests, since states further away than that mix in only at second order.

## Where it pays off, and where it does not

The gain of the paper is the ratio of plane waves to bands, $10^2$ to $10^3$. In a
tight-binding model with two orbitals per cell there is no such ratio: the honeycomb
window above keeps 40 of 62 states for an error of $10^{-4}$, and a sparse supercell with
the KPM already reaches $10^5$ to $10^6$ sites. So for a few-orbital model the method
gives no speed. What it still gives is the unfolded spectral function at no extra cost
(each basis state has a definite primitive momentum), the harmonics $\rho_{\mathbf Q}$ and
$\mathbf m_{\mathbf Q}$ as direct outputs, and every eigenvector in a dense matrix, which is
what a mean field and an RPA need.

The method pays off when the unit cell has many orbitals and the physics lives in a few
bands: a moiré cell as the unit cell with a super-moiré modulation on top, a Wannier
Hamiltonian from DFT (defuqumat's states brought into pyqula) with a long-wavelength
magnetic texture, a multi-orbital d-electron model. There the dimension is divided by
$N_\mathrm{orb}/N_b$, a factor 8 in the moiré rows above, so about 500 in the cost of a
dense diagonalization. As in defuqumat, this is a prefactor and not a change of scaling:
the dense solve is $(N N_b)^3$ per $\mathbf k_S$.

## What it computes

- The unfolded spectral function $A(\mathbf p,\omega)$ along a path of the primitive zone,
  from the weight of each state on the block of momentum $\mathbf p$. This is what ARPES
  measures, and at zero modulation it must equal `h.get_kdos_bands()` of the unit cell.
- The density of states on a mesh of $\mathbf k_S$, per unit cell.
- The local density of states over the whole ultracell at an energy, one FFT per state,
  which is what STM sees above a domain wall or a gate-defined potential.
- The density and the magnetization per site, and their harmonics $\rho_{\mathbf Q}$ and
  $\mathbf m_{\mathbf Q}$, which are the order parameters of a charge or spin density wave.
- In stage 2, the self-consistent mean field and its total energy, which decides whether a
  modulated state is below the uniform one.

## Two ways of giving the modulation

This is the first decision, and it sets the architecture.

In the first way, you build the modulated supercell with the existing methods
(`hs = h.get_supercell(n)`, `hs.add_onsite(fun)`, `hs.add_exchange(fun)`, strain on the
hoppings, pairing), and the ultracell solves it in the window of folded Bloch states of
the clean `h`, one block of $\mathbf q'$ at a time with an FFT over $\mathbf R$. Every term
pyqula can put on a supercell works at once, and the test against the supercell is a
change of solver on the same object. The mean field can reuse `scftk` as a new
density-matrix backend, since its mean-field Hamiltonian is again a supercell Hamiltonian
(stage 2 lists what that takes). The price is that the sparse supercell is stored, so the
size is bounded by its memory, as for a KPM calculation on the same supercell.

In the second way, you give the onsite potential and the exchange field as functions of
position, and we keep only their harmonics $W_{\mathbf Q}$. The supercell is never built,
so this reaches the largest $N N_\mathrm{orb}$, but every new kind of modulation (a
hopping, a pairing, a bond mean field) needs its own harmonic form, and the mean field
needs its own loop.

The two give the same matrix for an onsite term, to 3e-16 in the prototype. The
recommendation is the projection first, and the harmonics later as the fast path for
onsite terms, checked against the projection.

## Stages

### Stage 1: the non-interacting ultracell

The basis (folded momenta, the states of the unit cell in the window, a degenerate level
at the edge of the window taken whole or not at all, as defuqumat does), the matrix at
$\mathbf k_S$, its eigenstates and their FFT to real space, and the observables of the
list above. Diagonal ultracells in one, two and three dimensions, spinless, spinful and
Nambu.

### Stage 2: the mean field

A local Hubbard $U$ first, collinear and non-collinear, then the intersite $V_1$ and $J$,
whose Fock terms are modulated bonds and which the projection handles with no extra work.
The mean field can reuse `scftk` through a new `integration="ultracell"` of
`scftk.densitydensity.get_dm`, which computes only the entries the mean field reads, as
the `qtci` backend does. This takes three changes, read in the code but not made: an
entry in the registry of `get_dm`; a way for the clean `h` and the window to travel on the
supercell Hamiltonian through `copy()`, since the backend receives only the mean-field
Hamiltonian; and the spinful route, `VJinteraction`, which calls `get_dm` with
`integration="ed"` written into the call, passing its own `integration` through. With
them, the interaction builders, the mixing, the energies and the double counting come from
the existing code. Two physics points belong to this stage.
First, the basis is the band structure of the clean `h`, so a mean field whose
lattice-periodic part changes (a moment that appears on every site) needs a wide window;
in a tight-binding basis we can instead move the $\mathbf Q=0$ part of the mean field into
the unit cell and rebuild the basis at each iteration, at the cost of $N$ small
diagonalizations. Second, in its runs on Cr and on $\gamma$-Fe the paper holds
$\int\mathbf m_{\mathbf Q=0}=0$, so that the loop is not drawn to a lattice-periodic
solution; `scftk/mfconstrains.py` is where such a constraint goes. The paper also warns of charge
sloshing from the $1/Q^2$ of a long-range Coulomb interaction, which is why the long-range
Hartree term is left to stage 4.

### Stage 3: size

For a modulation with a few harmonics the matrix is sparse in blocks.
The unfolded spectral function then follows from the KPM with the $N_b$ start vectors of the
block of momentum $\mathbf p$, deterministic, the same factored moments as the KPM
unfolding of `unfolding.md`, and the states near the Fermi level from ARPACK through
`algebra.arpack_eigh`. The harmonic route for onsite terms goes in here, and so do the
non-diagonal ultracells ($\sqrt3\times\sqrt3$, commensurate moiré cells), whose folded set
is $\mathbb Z^d/M^T\mathbb Z^d$ with the $M$ that `get_supercell_map` already returns.

### Stage 4: response

The static RPA needs almost nothing new: the self-consistent response
to a weak modulation is the RPA susceptibility at $\mathbf Q$ in the all-band limit. The
dynamical susceptibility of a modulated state, a matrix $\chi_{\mathbf Q\mathbf Q'}(\mathbf
q,\omega)$ built from the states in the window, and the long-range Coulomb Hartree term in
$\mathbf Q$ space, with the screened-to-bare ramp the paper proposes against charge
sloshing, each want their own design once stage 2 has been used on a real problem.

Batched dense diagonalizations over $\mathbf k_S$ are the GPU-favourable shape, but no GPU
work is planned here, since each tier of `documentation/gpu_porting_plan.md` needs the
maintainer's sign-off.

## Where the code goes

- `src/pyqula/ultracelltk/`, with `basis.py` (folded momenta, window, multiplets, the FFT of
  a state), `matrix.py` (the projection, later the harmonics), `observables.py` (spectral
  function, DOS, LDOS, densities and harmonics) and, in stage 2, `densitymatrix.py` (the
  backend of `get_dm`).
- `src/pyqula/ultracell.py`, the public entry that composes them, and a one-line delegator
  on `Hamiltonian`.
- `tests/ultracell/`, the invariants below.
- `examples/2d/ultracell_*/main.py`, the guide page `documentation/guide/ultracell.md` with
  its line in the `nav` of `mkdocs.yml`, an entry in `reference-*.md`, and a bullet in the
  FUNCTIONALITIES list of `README.md`.

## Tests that can fail

- Every band kept gives the eigenvalues of the supercell at a generic $\mathbf k_S$, with a
  generic modulation, for odd and even ultracells in one, two and three dimensions,
  spinless, spinful with an exchange field and with Rashba coupling, and Nambu with a
  modulated pairing. Measured at 1e-14 in the prototype, spinless in the harmonic form
  and spinful with Rashba coupling, an exchange field and modulated hoppings in the
  general one.
- The unfolded weight equals that of `operator="unfold"` on the supercell, up to the
  factor $N$. Measured at 1e-13.
- At zero modulation the unfolded spectral function equals `h.get_kdos_bands()` of the
  unit cell at every $\mathbf p$, for any window that holds the energies.
- With a window of the form $\varepsilon<E_\mathrm{max}$ the $i$-th lowest eigenvalue is
  never below the $i$-th lowest of the supercell (Cauchy interlacing), and it decreases as
  $E_\mathrm{max}$ grows. A wrong phase or a wrong block breaks this.
- A helical exchange field of period $n$ against `h.generate_spin_spiral(qspiral=1/n)`:
  the union of the spiral bands at the folded momenta shifted by $\mathbf Q/2$ equals the
  ultracell spectrum. Measured on the supercell of a spinful chain ($n=7$) at 1e-15, and
  0.4 off without the half step, so the shift is part of the test. This pins the algebra
  of $\mathbf B_{\mathbf Q}\cdot\boldsymbol\sigma$.
- A weak potential $V_{\mathbf Q}$ induces $\delta n_{\mathbf Q}=\chi_0(\mathbf Q)V_{\mathbf
  Q}$, against `chitk.densitychi.densitychi_RPA` at $q=Q$ and zero frequency, with `U=0`
  for the bare response, and in stage 2 with the same $U$ as the mean field.
- In stage 2, the all-band mean field equals the supercell mean field from the same
  initial guess, and a stable state does not depend on the random initial guess.

## Decisions for the maintainer

Each comes with a recommendation; they are handed over one at a time.

1. How the modulation is given: the projection of a supercell Hamiltonian, the harmonics of
   functions, or both. Decided by the maintainer on 14 November 2026: both, the projection
   in stage 1 and the harmonics in stage 3 as the fast path for onsite terms.
2. How the basis is cut: `nbands=` counted from the bottom, as Elk and defuqumat do;
   `emax=`, every state below an energy, which keeps the interlacing bound and needs no
   frozen density; or `window=(emin,emax)`, which pays off for a wide multi-orbital model
   but makes the states below `emin` a frozen, lattice-periodic density that does not
   respond to the modulation. Stage 1 took the recommended default, `emax=`, with the
   maintainer's go-ahead to use the recommendations and report them; `window=` is still
   open for stage 2.
3. The normalization of the unfolded spectral function: per unit cell, so that zero
   modulation gives the unit cell's own `get_kdos_bands()`, or the convention of
   `operator="unfold"`, $N$ times larger. Stage 1 took per unit cell, the recommended
   default; it is a factor in `observables.kdos_bands` if the maintainer picks the other.
4. The public name: `h.get_ultracell(hs,emax=None)`, as in defuqumat, taken in stage 1 as
   the recommended default.
5. Whether the mean field rebuilds the basis with the $\mathbf Q=0$ part at every
   iteration, or keeps the basis of the clean `h` frozen as Elk does. To be measured in
   stage 2 before it is decided.
