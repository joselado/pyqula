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
  the f amplitude is $t_fV_j^*/J$.

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
