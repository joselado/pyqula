# Composite fermion and moment susceptibility of the Kondo lattice: what was built and what was left

This file records the work of 12 November 2026 on the two comparisons that
Chen, Stoudenmire, Komijani and Coleman, arXiv:2302.09701 (Phys. Rev.
Research 6, 023227), make between matrix product states and the auxiliary
fermion mean field of the Kondo chain. The request was the method of that
paper for the spectral function of the heavy-fermion Kondo lattice. The paper
has two such comparisons, and both were built (the maintainer's choice):

1. The matrix spectral function of $(c,F)$, where
   $F_\alpha=\tfrac23(\vec\sigma\cdot\vec S)_{\alpha\beta}c_\beta$ is the
   composite fermion, an electron with a spin flip. Matrix product states
   compute the Green function of this operator. The mean field replaces $F$
   by $f$ and rescales $V$ by 3/2 by hand.
2. The spin susceptibility of the moments as the product of two f
   propagators, with a residual interaction $U'(q)$ in the RPA.

The TeX of the paper was read, not the HTML. The condition of
`spinon_response.md` holds here too: cells of about 200 sites.

## What exists now

- `KondoLatticeHamiltonian.get_composite_spectral_function()`
  (`kondotk/compositespectrum.py`): $A_{cc}$, $A_{FF}$, $A_{cF}$ and
  $g_F=1/[G^{-1}]_{FF}$ along a k-path, from one diagonalization per
  k-point, per spin, diagonal in the Kondo pair. `J=None` is $F=f$; with
  `J`, $F=(4V^*/3J)f$. `hybridization_scale` multiplies $V$.
  `green=True` returns the complex Green functions.
- `KondoLatticeHamiltonian.get_moment_spinchi()` (`kondotk/momentchi.py`):
  $\chi^{-+}_{ij}(q,\omega)$ between the moments of the cell, bare or
  dressed with `U`, a number, a matrix or a function of `q`. It is the
  bubble of `chitk/localspinchi.py` restricted to the f sites, so it takes
  the spin-degenerate and the spin-split routes of that module.
- `chitk.localspinchi.bare_spin_chi(..., sites=)` and `spin_chi(...,
  sites=)`: the response of a list of sites only. The bubble still sums
  over all the states. This is what makes the moment response 4 times
  cheaper than the full one.
- `kondotk/pairs.py`: the Kondo pairs, the hybridization read from the
  matrix elements of the Hamiltonian (so a $V$ set by hand works), and the
  rescaled copy. `kondotk/cotunneling.py` uses it now.
- Three guide sections with one figure each on the pseudofermion page
  (the chain, the chain susceptibility, and the square lattice with a
  finite flake), two reference entries, and four examples under
  `examples/kondolattice/`: `composite_spectral_function`,
  `moment_spinchi`, `composite_fermion_2d` and `composite_fermion_0d`.

## Dimensionality

The maintainer asked that the method works in every dimensionality, 0D, 1D,
2D and 3D. Both routines already did, since they only use the Bloch
Hamiltonian and the k-mesh of the geometry. What changed: in 0D the
spectral function returns one k-point of shape (1, 3) (the k-path helper
gives a bare `[0.]` there), and the docstrings and the guide say what the
general case is. With several pairs per cell $G^{-1}=z-H(k)$ is a matrix in
the pairs, $g_F$ is the Schur complement
$G_{FF}-G_{Fc}G_{cc}^{-1}G_{cF}$, and the RPA is
$\chi(1-U'\chi)^{-1}$ with $U'$ a matrix between the moments.

`tests/kondolattice/test_kondo_response_dimensionality.py` checks a
honeycomb flake of 24 sites (0D), the doubled chain, the square, honeycomb
and triangular lattices and the cubic lattice: the Green functions against
the inverse of $z-H(k)$ at generic momenta, the susceptibility against the
sum over pairs of eigenstates at a generic $q$, the three forms of $U'$,
and the local sum rule $n_\downarrow(1-n_\uparrow)$ of each moment. The
triangular lattice is there because it has no particle-hole symmetry
($\lambda\neq0$, filling 0.2): at filling 0.5 its loop does not converge,
which is the known caveat of the loop and not of these routines.

Square lattice at $J'=2t$ (40 x 40 k-points in the loop): $V=1.786$, the
$V$ of the gap equation on that mesh to $10^{-4}$; indirect gap 1.36; with
$U'=-2t$ the magnon is at 3.32, 1.98 and 1.21 at $\Gamma$, X and M,
converged at 30 x 30 k-points, and $1-U'\chi(q,0)$ is 0.79, 0.66 and 0.48.
Cubic lattice: $V=1.704$.

## Conventions that had to be settled

- The paper writes $J'(c^\dagger\vec\sigma c)\cdot\vec S$. With
  $\vec S=\tfrac12f^\dagger\vec\sigma f$ its quartic term is
  $-J'X^\dagger X$, $X=\sum_ac^\dagger_af_a$, and our loop decouples
  $-(J/2)X^\dagger X$. So $J=2J'$. Checked: the loop at half filling gives
  the $V$ of $1/J'=\frac1N\sum_k1/\sqrt{(\epsilon_k/2)^2+V^2}$ to $10^{-7}$
  ($V=0.6490$ at $J'=0.9t$, $1.8771$ at $J'=2t$, $9.975$ at $J'=10t$).
- The equation of motion $i\partial_tc=\epsilon c+\tfrac32J'F$ against the
  mean-field one, $i\partial_tc=\epsilon c+V^*f$, gives
  $F=(2V^*/3J')f$. With this amplitude the mean field obeys the three
  exact relations of the paper (Eq. relations, Eq. matrixG) to $10^{-13}$,
  for any `hybridization_scale`, since the scale changes the weight of $F$
  and not the hybridization $3J'/2$ of $c$ with $F$.
- At strong coupling the loop gives $V=J'$ and one site diagonalized
  exactly costs $3J'/2$ to add an electron. This is the 3/2 of the paper.
  The weight of $F$ is then $|z|^2=4/9$ without the rescale and 1 with it.
- The paper's printed mean field does not agree with this in two places,
  and we did not follow it there (our reading, not checked with the
  authors). Its Eq. eq2band has $\sqrt{\epsilon^2+V^2}$ where its own
  matrix gives $\sqrt{\epsilon^2+4V^2}$. Its mean-field appendix prints
  $1/J=\frac1L\sum_k1/\sqrt{\epsilon^2+4V^2}$ and $V\to J/2$, a factor 2
  below ours, as if the sum over spin was left out; with $V=J/2$ the
  rescale of 3/2 would give $3J/4$, not the exact $3J/2$.
- Sign of the susceptibility: the one of the spin responses of the
  library. The pole is at $\omega-i\delta$, the static response is
  negative and the spectral weight at $\omega>0$ is $+\mathrm{Im}\chi/\pi$.
  In the pair sum the first operator takes the occupied state to the empty
  one, so $\chi^{-+}$ of the paper ($S^+$ excites) is
  $\chi_{xx}+\chi_{yy}-i\chi_{xy}+i\chi_{yx}$ of the library layout. A free
  moment in a Zeeman field fixes this (pole at the splitting, weight 1).
- $\chi^{+-}=2\chi^{zz}$, so $1/(1/\chi-U')$ is the dressing of the
  exchange $2U'$ between the spin components. It is applied in that form
  to the three components, so it is defined in a field too.

## What was measured

Exact diagonalization of a periodic ring of 4 sites (4096 states, script
written for this, not kept in the repository):

| quantity | $J'=2t$ | $J'=0.9t$ |
| --- | --- | --- |
| relations $G_{cc}=g_c+g_cvG_{FF}vg_c$, $G_{cF}=g_cvG_{FF}$, $v=3J'/2$ | $4\times10^{-15}$ | $10^{-14}$ |
| $\langle\{F,F^\dagger\}\rangle$ | 0.962 | 0.835 |
| main pole of $A_{FF}$ at $\epsilon_k=-2t,0,2t$ | 2.14, 2.88, 3.96 | 0.79, 1.16, 2.53 |
| mean field, $V$ of the infinite chain | 1.13, 1.88, 3.13 | 0.19, 0.65, 2.19 |
| the same with $V\times1.5$ | 1.99, 2.82, 3.99 | 0.40, 0.97, 2.40 |
| lowest pole of $\chi_S$ at $q=0,\pi/2,\pi$ | 4.27, 3.49, 2.84 | 1.87, 1.23, 0.59 |

So at $J'=2t$ the rescaled bands are within 7% of the exact poles and the
bare ones are 1 off. At $J'=0.9t$ a ring of 4 sites is too small to judge:
the level spacing is of the order of the gap.

Mean field of the infinite chain ($\delta=0.05$, 200 k-points):

- $J'=2t$: direct gap 3.75, indirect gap 2.25 (5.63 and 3.98 with the
  rescale). With $U'=-2t$ the magnon is at 3.46 ($q=0$) and 1.96
  ($q=\pi$), and at 5.28 and 3.78 with the rescale. The strong-coupling
  spin gap is about $2.3t$, so the paper's $U'=-2t$ fits the bare $V$, not
  the rescaled one. The paper does not say which $V$ its susceptibility
  panels use.
- $J'=0.9t$: $\chi(\pi,0)=-1.35$ with the bare $V$, so the paper's
  $U'(\pi)=-0.75t$ gives $1-U'\chi=-0.01$, the magnetic instability. With
  the rescale $\chi(\pi,0)=-0.62$, the denominator is 0.54 and the magnon
  is at 0.64, below the continuum at 0.79 (the 4-site ring has 0.59).
- The local sum rule gives 0.2476 against the 1/4 of free f fermions
  (tails of the broadening), for any $V$. A spin 1/2 has 1/2.

Agreement of the routines with the closed forms of the chain: $10^{-13}$
for the spectral function against the inverse of the $2\times2$ matrix,
$10^{-15}$ for the susceptibility against the sum of the two propagators,
at $q=0$ and at generic $q$. The susceptibility is converged to
$2\times10^{-3}$ at 200 k-points with $\delta=0.05$.

Cost at 200 Kondo pairs (800 x 800, $V$ set by hand): 0.9 s per k-point
for the spectral function with 400 energies; 49 s per k-point for the
susceptibility with 100 energies and the RPA, 2.2 GB. A converged loop at
that size was not run.

## Not built

- **The matrix product side.** The comparison of the paper needs the
  Green function of $F$ in the interacting chain. dmrgpy is installed and
  was not used; only the 4-site ring above was diagonalized.
- **$F$ with Wick's theorem.** $F=\tfrac13\sigma\cdot(f^\dagger\sigma f)c$
  is a three-fermion operator. In the mean-field state its contraction
  with $\langle f^\dagger c\rangle$ is the $(4V^*/3J)f$ that is used here.
  The normal-ordered rest is a three-particle continuum that carries the
  weight missing from $|z|^2$ (0.39 at $J'=2t$, against the exact 0.96 of
  the ring). It is not in the paper. It is a double convolution, the same
  kind of sum as the third order of `kondo_tunneling.md`.
- **$9\chi_F$ and the wave packet.** The paper compares $\chi_S$ with the
  susceptibility of $\vec S_F=\tfrac12F^\dagger\vec\sigma F$ and evolves a
  wave packet of $F$, both with matrix product states. In the mean field
  the first is $|z|^4$ times the moment response, and the second is the
  propagation in two bands. Neither was written.
- **The strong-coupling expansion** of the paper, one and two particles
  (the doublon-holon problem with its bound state).
- **$U'(q)$ from a theory.** It is a fitted parameter in the paper and a
  keyword here. Nothing derives it, and nothing warns when
  $1-U'\chi(q,0)$ is negative.
- **Unfolding.** In a supercell the spectral function is at the momentum
  of the supercell, diagonal in the pair. The elements between different
  pairs, which an unfolding needs, are not returned.
- **README bullet and notebook.** As for the cotunneling, a
  FUNCTIONALITIES bullet needs its own executed notebook, and neither was
  made.
- **Small things.** The temperature of the susceptibility is `delta` when
  not given, the convention of the spin responses, so a gap smaller than
  `delta` wants `T=` set. The rescale keeps the f levels, so away from
  half filling the f occupation is not one any more.

## State of the tests

`tests/kondolattice/test_composite_spectral_function.py` (13 tests),
`tests/kondolattice/test_moment_spinchi.py` (11 tests) and
`tests/kondolattice/test_kondo_response_dimensionality.py` (42 collected,
10 of them skips of the tests that are for one geometry only), about one
minute together. They pass, together with `tests/kondolattice`,
`tests/kondotunneling`, `tests/spinon`, `tests/chi` and
`tests/documentation`. The suite collects 3476 tests now; the whole suite
was not run.
