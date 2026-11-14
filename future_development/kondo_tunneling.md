# Tunneling spectra with a Kondo peak: what was built and what was left

This file records the work of 12 November 2026 on the tunneling spectrum
(STM $dI/dV$) of the two pseudofermion mean fields. The request was a way
to simulate the Kondo peak seen in tunneling, for spinon Hamiltonians, with
the hybridization picture for the heavy-fermion Hamiltonian. The condition
of `spinon_response.md` holds here too: everything has to work for cells of
about 200 sites.

## What exists now

- `SpinonHamiltonian.get_kondo_spectrum()` (`kondotk/spinonspectrum.py`,
  `kondotk/thermal.py`) is the perturbation theory of Ternes,
  arXiv:1505.04430, to second and third order in the exchange between the
  tunneling electron and the spin of the site, with the spin correlators of
  a quadratic Hamiltonian. The formulas are the ones the dmrgpy package
  sums over many-body eigenstates, with its two departures from the printed
  paper (the thermal logarithm read as a convolution, and the exchange
  diagram at $F(eV-(E_f-E_m))$). Here Wick's theorem reduces them to the
  occupied and empty local spectral matrices of the site, $O(\epsilon)$ and
  $E(\epsilon)$, 2 x 2 in spin. This reduction is our own derivation, not
  taken from a paper:
  - the second order is $\frac14\sum_a\int\mathrm{Tr}[\sigma_aE(\epsilon+
    \omega)\sigma_aO(\epsilon)]$, the convolution of the two;
  - the third order has two connected terms. In the first the virtual pair
    shares its hole with the final pair,
    $\frac18\epsilon_{jkl}\mathrm{Tr}[\sigma_lE\sigma_kE\sigma_jO]$. In the
    second it shares its particle, with a minus sign,
    $-\frac18\epsilon_{jkl}\mathrm{Tr}[\sigma_lE\sigma_jO\sigma_kO]$;
  - a site with a moment adds the three terms where one spin operator is
    replaced by its average.
  Each term is a product of two functions of one energy summed over a third
  one, so a site costs a few convolutions on an energy grid.
- `KondoLatticeHamiltonian.get_cotunneling_didv()` (`kondotk/cotunneling.py`)
  is the cotunneling spectrum of the large-$N$ mean field, the spectral
  function of $\psi_j=t_cc_j+t_ff_j$ (Maltseva, Dzero and Coleman,
  arXiv:0910.1138, and Figgins and Morr, arXiv:1001.4530, TeX of both read).
  By default `tf` is the amplitude of the f fermion, as in the second
  paper. With `J=` it is the bare spin-flip amplitude of the first one, and
  the f amplitude is $2t_fV_j^*/J$, with $J$ the Coqblin-Schrieffer coupling
  of the loop (see the repairs below).

## What was measured

Spinon spectrum, against three references:

| reference | case | largest difference |
| --- | --- | --- |
| sums over the Fock space, written for the test | 1 to 3 sites, with and without a field in a generic direction, $T=0.05$ | $5\times10^{-4}$ (second order), $10^{-4}$ (third) |
| dmrgpy `get_kondo_spectrum(mode="ED")` on a free-fermion chain, whole Fock space | 2 and 3 sites, $T=0.05$ and $T=0$, with and without a field | $3\times10^{-4}$ at $T=0.05$; $10^{-14}$ (second) and $6\times10^{-4}$ (third) at $T=0$ |
| Fig. 7b of arXiv:1505.04430 (spin 1/2, 1 K, 20 meV, $J\rho_s=-0.05$) | one decoupled site, spectrum times 2 | 1.137 and 0.884 against 1.13 and 0.886 |

The factor 2 of the last line is exact at both orders: the free fermion has
$\langle S^2\rangle=3/8$. The method does not apply it.

What the method gives on spinon states (bond decoupling, $T=0.005J$,
$J\rho_s=-0.05$, $\omega_0=2J$):

- The uniform chain has no zero-bias peak: 0.0005 at zero bias, 0.015 at
  $0.1J$, a $V^2$ onset. The third order only rescales it.
- A chain of trimers ($J'=0.1J$) has a peak on the two end sites (0.805 at
  zero bias, 0.727 at $0.1J$, second order flat at 0.589) and nothing on
  the middle site. This is the mean-field form of what exact
  diagonalization finds, a peak only with a degenerate ground state.
- The free bond loop decouples the trimers completely for $J'\le0.1J$ (the
  hopping between them goes to $2\times10^{-4}$), and keeps 0.054 at
  $J'=0.3J$. So the width of the level at the Fermi energy is not
  proportional to $J'$ in this mean field. At $J'=0.3J$ the level is a
  band wider than $T=0.005J$ and the peak is gone: the end sites give 0.018
  at zero bias against 0.64 at $0.1J$. The condition for a peak is a level
  width below the temperature, which is what a trimerization sweep has to
  watch.
- The Fermi energy is taken at zero. On a frustrated cell with
  inequivalent sites (36 sites of the triangular lattice, $\pi$-flux
  pattern held, loop at $T=0.02$) the occupation that the spectrum sees is
  1 to $2\times10^{-4}$ on every site at $T_s=0.02$ and at $T_s=0.005$, on
  the k-mesh of the loop and on a denser one, and to $3\times10^{-3}$ at
  $T_s=0$ on the denser mesh.
- A Zeeman field of $b=0.1$ on a free site moves the maximum to $\pm0.23$,
  just outside the Zeeman steps at $\pm0.2$.

Cost at 196 inequivalent sites (triangular supercell), 401 bias points:
3 s with 4 k-points and 5 s with 9, 0.4 GB, for a spin-degenerate state;
10 s and 0.5 GB with a non-collinear field, where the four components of
the spectral matrices are kept.

Cotunneling spectrum, on the chain at $J=1.5$, filling 0.15 ($V=0.374$):

- It agrees with the closed form of the uniform lattice to $10^{-14}$. The
  closed form has to include the bare f level,
  $G_\psi=(t_c+t_fV/(\omega-\lambda))^2G_{cc}+t_f^2/(\omega-\lambda)$. The
  printed Eq. GpsiKL of arXiv:0910.1138 has the first term only, which
  leaves a pole inside the gap. Summed over a flat band, our f-level term
  is $2D\,t_f^2/t_c^2/(\omega-\lambda)$ in the units of its Eq.
  GpsiKLsummed, whose last term is printed as $2D/t_c^2/(\omega-\lambda)$:
  the same term without the factor $t_f^2$ (done by hand, not computed).
  So the comparison with this paper is with a closed form that we
  completed, not with a number or a figure of it.
- The distance between the two gap-edge peaks is 0.1822, and the exact
  indirect gap of the two bands is 0.18195. The leading-order estimate
  $V^2(1/d_1+1/d_2)$ gives 0.204, since $V/d_1$ is 0.43 here.
- $t_f/t_c=\pm0.3$ puts the peak at the upper or at the lower edge (2.1 and
  2.5 against 0.72 for the conduction band alone).
- 200 Kondo pairs (800 x 800): 1 s per k-point, 0.5 GB, with $V$ set by
  hand. A converged loop at that size was not run.

## Not built

- **README bullet and notebooks.** The guide page, the reference entries
  and one example for each method exist. A FUNCTIONALITIES bullet needs its
  own executed notebook, and neither was made.
- **Paper figures of the cotunneling.** The single-impurity Fano form and
  Fig. 2 of arXiv:0910.1138 need one impurity or a flat conduction band,
  which the class does not have. The figures of arXiv:1001.4530 use their
  own saddle point with a Heisenberg term.
- **The spinon Kondo-Heisenberg mean field.** Spinon hoppings and a
  hybridization with a substrate band in one loop would give a peak on a
  uniform spin liquid, which the third order cannot. It joins the two
  classes and is an architectural change. arXiv:cond-mat/0305193 is the
  paper to start from (seen in a search, not opened).
- **Leading logarithms.** dmrgpy sums them with a running coupling that
  depends on the many-body states. The local-spectral form of that flow
  was not derived.
- **Potential scattering.** The term $U$ of the paper, which makes the
  spectrum asymmetric in bias, needs the charge correlators of the site.
  With the constraint only on average they are not zero, so its meaning for
  spinons is not clear.
- **Dressed spin spectrum.** The second order uses the free spinons. The
  local spectrum of the spin-channel RPA needs a sum over $q$, 20 to 30 s
  per $q$ at 196 sites.
- **Gutzwiller factor.** Nothing multiplies the spectrum by 2. The factor
  is exact for a free moment only.
- **Small things.** The thermal broadening of the cotunneling is a
  convolution on the energy grid, resolved only when $T$ is larger than the
  step. A complex $V$ is covered by one test. The spinon spectrum takes the
  Fermi energy at zero and has no keyword for another one.

## State of the tests

`tests/kondotunneling` has 20 tests (8 spinon, 12 cotunneling), about 30 s.
They pass, together with `tests/documentation` (14), `tests/spinon` and
`tests/kondolattice`. The suite collects 3410 tests now; the whole suite was
not run. The website builds in strict mode and the PDF of the guide builds.

## Repairs of the seventh bug audit

- **The amplitude with `J=` was half of the right one** (finding kondo#0,
  14 November 2026). The loop takes the exchange
  $H_I=\frac J2\vec S\cdot(c^\dagger\vec\sigma c)$, so $J/2$ is the
  Heisenberg coupling $J'$ of Maltseva, Dzero and Coleman, and
  $[c,H_I]=\frac J2(\vec\sigma\cdot\vec S)c$. In the mean field
  $[c,H_I]$ is $V^*f$, so the spin-flip operator $(\vec\sigma\cdot\vec S)c$
  of the paper is $2V^*f/J$, and the f amplitude is $2t_fV_j^*/J$. The
  code had $t_fV_j^*/J$, so with `J=` the f part was 4 times too small and
  the interference part 2 times. The default `J=None` was not affected,
  and no example or guide snippet passes `J`. The composite fermion of
  `compositespectrum` already used the right factor, $F=\frac23
  (\vec\sigma\cdot\vec S)c=\frac{4V^*}{3J}f$.
  `test_J_gives_the_amplitude_of_the_spin_flip_operator` compares the
  amplitude, on the two inequivalent sites of the staggered chain ($J=1.5$,
  filling 0.15, $nk=50$), with $2H_{cf}/J$ read from the converged
  Hamiltonian and with $\frac32$ times the composite amplitude, and the
  whole spectrum with the one of the explicit amplitude. It fails on the
  old code by $0.38$ and passes to $10^{-12}$. The old test passed the
  formula of the code by hand, so it could not see the factor.
  Both references of that test carry the same $2/J$ as the code, so the
  review added two tests that do not.
  `test_J_amplitude_is_the_order_parameter_of_the_mean_field` uses the
  relation of the paper itself, $(\vec\sigma\cdot\vec S)c\to
  -\sum_s\langle f_s^\dagger c_s\rangle f$, an expectation value of the
  converged state where no coupling enters. Taken from the eigenstates on
  the mesh and at the temperature of the loop ($nk=50$, $T=0.02$), it
  agrees with the amplitude to $9\times10^{-9}$, and the old code is off by
  $0.38$.
  `test_J_spectrum_does_not_depend_on_the_phase_of_the_f_fermion` gives
  each f fermion a phase by hand and asks for the same spectrum, which
  holds because the amplitude follows $V_j^*$ and not $V_j$. A copy of the
  code without the conjugate passed every test before, since the loop
  always returns a real $V$, and it fails this one.
- **The k-loop of the spectrum** (finding kondo#2 of the spinon group) was
  left as it is. The batched diagonalization of that finding was not
  supported by its second reader, and the half block of a spin-degenerate
  mean field was claimed only for `spinonspectrum.py`. The same half block
  would work here when the mean field is spin degenerate, one
  diagonalization of half the size per k-point. It was not built then; it
  is built since the decisions below.
- **The loop does not keep a complex $V$** (seen in the review, not
  repaired then: `scftk/kondolattice.py` was outside that group; repaired
  in the decisions below). A seed
  $0.3e^{0.7i}$ on the staggered chain converges to the same real $V$ as a
  real seed. The reason is the update $V\leftarrow-\frac J2\sum_s$
  `dm[f,c]`: on a converged state whose f fermions were given a phase by
  hand, this target is the complex conjugate of the $V$ in the Hamiltonian,
  to $7\times10^{-9}$, so the density matrix is used transposed. Each
  $V_j$ can be made real with the phase of its own f fermion, so the
  solution that the loop finds is right. Whether a model exists in the
  package where this choice of phase is not possible was not checked.

## Decisions of 14 November 2026

These close the two items that the repairs above left open here.

- **The loop takes the density matrix the right way.** The mean-field term
  is $V_jf^\dagger_jc_j+{\rm h.c.}$, so the self-consistency equation, the
  stationary point of the energy in $\bar V_j$, is
  $V_j=-\frac J2\sum_s\langle c^\dagger_{js}f_{js}\rangle$.
  `get_density_matrix` returns the transpose of the density matrix,
  `dm[i,j]` $=\langle c^\dagger_ic_j\rangle$, so this is the element
  `dm[c,f]`, and the loop read `dm[f,c]`, whose target is $\bar V_j$. For a
  real $V$ the two are the same number, which is why every result so far
  stands. Two cases tell them apart. A seed $0.3e^{0.7i}$ on the chain
  ($J=1.5$, filling 0.15, `nk=40`) relaxed to the real $V=0.3737$; it now
  converges to $0.3737e^{0.7i}$ with the same energy. And the question
  left open above has an answer: the phase cannot be removed when the
  moments are coupled by a complex hopping, which the guide allows
  ("a hopping between the moments"). On a chain of two Kondo pairs per
  cell with a hopping $0.2e^{0.9i}$ between the two moments of the cell
  and $0.2$ between cells ($J=2$, filling 0.15, `nk=30`, $T=0.02$), the
  old loop reported convergence at $V=(0.432-0.022i,\,-0.432-0.022i)$,
  which is the conjugate of the order parameter of that state, off by
  $4.5\times10^{-2}$, with the energy $-2.7311$. The loop now gives
  $|V|=0.451$ on both sites with $V_1\bar V_0=0.203e^{2.69i}$, equal to
  the order parameter taken from the eigenstates to $2\times10^{-6}$, with
  the energy $-2.7676$, lower by $0.036$, and two seeds with different
  phases agree on the energy and on $V_1\bar V_0$. So with a complex
  hopping between the moments the old loop returned a state that was not
  a solution; with a real one, or none, nothing moves. The module
  docstring and the guide had the formula with $f^\dagger c$; both are
  corrected. `tests/kondolattice/test_kondolattice_complex_hybridization.py`
  has the two tests, and both fail on the code before.
- **The half block in the two k-loops.** A mean field without a Zeeman
  field is the same for the two spins and does not mix them, so the
  cotunneling spectrum and the composite spectrum now diagonalize the
  block of one spin, half the size, and count every state twice, as
  `spinonspectrum.py` does. The two loops built the same weights with the
  same lines; they are now one function, `kondotk.pairs.pair_weights`,
  which also serves the f block of $g_F$. With 196 Kondo pairs on the
  square lattice ($784\times784$, $V$ set by hand, 4 k-points, 401
  energies, a shared workstation with two threads, the smallest of three
  runs): the cotunneling spectrum goes from 3.12 s to 0.57 s and the
  composite spectrum from 3.17 s to 0.65 s, agreeing to $2\times10^{-13}$
  on values up to 12. A Hamiltonian with a field or a spin-orbit coupling
  takes the full matrix as before. `tests/kondotunneling/test_kondo_half_block.py`
  has four tests: the cotunneling spectrum against the Green's function
  inverted at every energy, with and without a field, every output of the
  two routines with the half block against the full matrix forced on the
  same Hamiltonian, and the size of the matrices that are diagonalized.
  Three fail on the code before; the one with a field passes on both.
  A block per spin for a field along $z$, two diagonalizations of half the
  size, is not built here: the bubble of `chitk/localspinchi.py` has it,
  and here it would save a factor 4 where the half block saves 8.
