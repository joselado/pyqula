# Physics review of the notebooks, and the fixes it led to (2026-09-26)

Every executed notebook in `jupyter-notebooks/` was read against its own
figures and printed numbers: the 82 functionality notebooks linked from the
README's FUNCTIONALITIES list, then the 11 tutorials. For each one the question
was whether the case and the result make physical sense: printed numbers
against the analytic value where one exists, every claim in the prose against
the figure, and the convention in the source (spin degeneracy, the
normalization of a coupling, the sign of an invariant) before calling anything
a mismatch. A second pass then fixed what the review found, in the library and
in the notebooks, and re-executed every notebook whose case or output changed.

## Literature cells

56 notebooks end with a "Comparison with the literature" cell that says which
figure, equation or result of which paper the notebook reproduces ("... compare
with Fig. 2 in Ref. [1]"), with the journal reference from the arXiv abstract
page (or crossref when arXiv lists none) and an arXiv link. Each figure or
equation number was read in the paper itself (`pdftotext` of the arXiv PDF);
papers older than arXiv (Haldane 1988, BTK 1982, Onsager 1944, Jackiw-Rebbi
1976, ...) are cited by DOI and never with a figure number. All 85 titles and
journal references were checked against arXiv and crossref by script. Where the
notebook differs from the paper (a normalization, spinful against spinless, a
different lattice) the cell says so. The conventions that came up again and
again:

- the Haldane amplitude is $(\sqrt3/2)t_2$, so the phase boundary at
  $\phi=\pi/2$ sits at $|m|=4.5\,t_2$ rather than $3\sqrt3\,t_2$
- `add_soc(λ)` is Kane-Mele's $\lambda_{SO}=(\sqrt3/2)\lambda$ on the flat
  honeycomb lattice, a Dirac mass of $4.0\lambda$ on
  `buckled_honeycomb_lattice`, whose bonds are shorter in plane, and
  Fu-Kane-Mele's $\lambda_{SO}=2\lambda/3$ on the diamond lattice
- `add_rashba(c)` is Kane-Mele's $\lambda_R=c$
- the default Hamiltonian is spinful
- `get_magnetization()` returns $\langle\sigma\rangle$, and
  `extract("mz")` returns the exchange field $Um/2$, not the moment
- the RKKY map uses $J>0$ ferromagnetic, the opposite of the classical-spin
  notebooks

## What the fix pass changed in the library

Each item has a regression test that fails on the old code, and each test file
that moved records the old and new value and why.

- **Kane-Mele and Haldane couplings on multicell Hamiltonians**
  (`kanemele.py`) only reached the cells that already had a first-neighbour
  hopping: on the diamond lattice 6 of the 12 second-neighbour bonds, so the
  spectrum was not cubic symmetric and $E(W)$ was $\pm0.33$ instead of
  $\pm0.53$ for `add_soc(0.1)`; a 3D stack of Haldane layers missed 2 of 6.
  2D results are unchanged, and so are the $\mathbb Z_2$ indices of topology
  10.
- **Non-Hermitian mean field**: the density matrix went through `eigh`, which
  reads the lower triangle, so gain, loss and non-reciprocity were silently
  dropped. The SCF now uses the biorthogonal density matrix
  $\sum f(\mathrm{Re}E)|R\rangle\langle L|$ (Brody arXiv:1308.2609), with the
  Hatano-Nelson imaginary gauge transformation as its exact test; the
  Hermitian path is bit for bit unchanged. `set_filling` takes the real parts
  too.
- **`Vr` and `Jr` in the SCF are per pair**, as V1 is and as the BSE and the
  charge RPA take them; they were effectively $2V_r$. The graphene Coulomb
  reference moved from 140.48297 to 136.51566 (exactly the old code at
  $V_r/2$).
- **Charge RPA** (`chitk/densitychi.py`) put every neighbour shell in at half
  strength; `V(q)` on a chain is now $2V_1\cos q$.
- **Exactly degenerate pairs in the Lindhard function** (`chiAB.py`,
  `chijax.py`) were dropped; they now give $f'$ at $\omega=0$ (the static,
  compressibility limit) and vanish for $|\omega|\gg\delta$. At nesting q
  with a mesh through the nesting points the static value was 4% off
  ($-1.1635$ against $-1.2135$).
- **p-wave magnet**: the claim that it gives a nonlinear spin current at
  $l=0$ was $k$-mesh noise (the $l=0$ entry is a zone integral of a total
  derivative, and the p-wave bands are one rigidly shifted cosine per spin);
  removed from docstrings, guide, example and the test that asserted it.
- **Wannier Hamiltonian** is Hermitian at every $k$: hoppings on the
  Wigner-Seitz cells of the $n_k$ supercell with Wannier90's degeneracy
  weights. The default initial guess is deterministic (SCDM column selection,
  arXiv:1507.03354); the random draw ended in a wrong minimum in 3 of 12 runs.
- **KPM resolution**: `delta` is now the half width of a level's peak in every
  KPM routine (`kpmtk.kernels.jackson_npol`), as in the ED and Green's
  function modes; before, the same `delta` gave widths from $0.37\delta$ to
  $1.86\delta$. Also fixed: `get_dos(use_kpm=True)` ignored `energies`,
  `dm_vivj_energy` mixed two kernels, `correlator0d`'s file had its columns in
  the wrong order.
- **Local probe S-matrix** carried an $i\delta_{smatrix}$ sink on the central
  blocks, which left an in-gap floor of $\sim10^{-11}$ ($\kappa(0)=0.80$ at
  weak coupling); it is now unitary to $10^{-15}$ and $\kappa(0)=2.000$.
- **Real-space Chern marker** is normalized by each site's Voronoi area; the
  old disk estimate read 3% low on a triangular island. Crystalline islands
  now give exactly 1 (spinless) and 2 (spinful).
- **Amorphous Chern insulator** (new): `geometry.amorphous_lattice` and the
  Agarwala-Shenoy model (arXiv:1701.00374) in `geometrytk/amorphous.py`, so
  the README bullet on amorphous systems is now true.
- **Quantum metric in Cartesian coordinates**: `coordinates="cartesian"` in
  `topologytk/qgt.py`; the reduced-coordinate trace is not $g_{xx}+g_{yy}$
  and breaks the lattice symmetry.
- **Superfluid weight** can be split into conventional and geometric parts
  when a band touching sits on the mesh (the K point of kagome, $\Gamma$ with
  Rashba).
- `h.get_chi(energies=[0.])` works with a list; docstrings of
  `z2_invariant_3d` (sum modulo 2, not product) and `get_magnetization`;
  Coleman's review cited as Sec. II.C; the Lindhard occupation factor in the
  guide; the guide's kagome Wannier example (the flat band is index 0).

## User-visible changes (read before upgrading)

- `Vr` and `Jr` in the mean field are per pair, as V1: a script passing `Vr=`
  or `Jr=` gets half the interaction it got before (for example
  `examples/2d/hopping_renormalization_V1`, whose `Vr=[1.0]` now means $V_1=1$,
  and `examples/2d/graphene_coulomb_interaction`).
- `delta` in every KPM routine is the half width of a level's peak: at the
  same `delta`, `dos_kpm` and the KPM path of `set_filling` use 1.85 times the
  polynomials, `dos_site_kpm` 0.37 times, and `get_kdos_bands(mode="KPM")`
  about 0.6 times.
- The Wannier default initial guess is deterministic (SCDM), and the hoppings
  sit on the Wigner-Seitz cells, which changes the interpolation between mesh
  points and the number of hopping cells (never the bands on the mesh).
- `non_hermitian=True` Hamiltonians now get the biorthogonal density matrix,
  Fermi level and filling; Hermitian results are bit for bit unchanged.
- The charge RPA (`get_densitychi_RPA`, `get_plasmon_bands`) uses every
  neighbour shell at full strength, and every Lindhard response gets the
  static $-N(0)$ at $q=0$ from its degenerate pairs.
- `add_soc`/`add_kane_mele`/`add_haldane` on 3D multicell lattices couple
  every second-neighbour bond.
- The real-space Chern marker is normalized by each site's Voronoi area.
- Local-probe conductances inside a gap drop by about $10^{-11}$.
- The superfluid-weight decomposition no longer raises when a band touching
  sits on the mesh.

## What the fix pass changed in the notebooks

Every notebook whose case, code or output changed was re-executed single core
and its prose aligned with the new output. The recurring problem was a
self-consistent calculation on the default `nk=8`, where a metal's Fermi level
lands on a degenerate level and the state is pinned to a van Hove point, at the
wrong filling and with a gap several times too large (mean-field 03, 04, 05,
16, single-particle 02, 05, and the README superconductivity examples). Each now
passes a converged `nk`, and the superconducting ones are checked against an
independent solution of the BCS gap equation on the same band (to four digits
at the loop's chemical potential). The `nk=8` default itself was left alone.
Other notebooks got a case that is physically meaningful and comparable to a
reference, for example: a topological superconducting lead with the $2e^2/h$
Majorana peak (transport 04, Law-Lee-Ng), BTK and multiple Andreev reflection
against Cuevas et al. (transport 02, 09), the Hayata-Yamamoto non-Hermitian
Hubbard model (mean-field 08), the honeycomb Heisenberg magnon (mean-field 18,
Peres et al.), the mean-field Hubbard curves of Raczkowski et al. (tutorial 05),
the extended Hubbard chain (mean-field 09, Hirsch, Jeckelmann), cRPA-type
screening with a positive-definite interaction (mean-field 21, Wehling et al.),
the Fu-Kane-Mele bands (single-particle 07), the 2D RKKY asymptote (response 05,
Beal-Monod) and the valley Chern number tending to 1 (topology 02).

## Corrections to the first review

- The BdG Wannier error of 0.046 (Wannier 04) is slow interpolation
  convergence, not the Hermiticity defect: both codes give it at nk=24, and it
  falls to $1.8\times10^{-4}$ at nk=192.
- Topology 03's plateau falls short of 2 by about $2.3\delta$; the recipe is
  to extrapolate in $\delta$ with an energy grid of at most $\delta/3$, not
  a particular `dk`.
- Topology 08's point at 1.237 was mostly $k$-mesh error (1.121 converged).
- Mean-field 10's upper branch is a longitudinal bound state, not a spin-flip
  one.

## What is still open

- `greentk/rg.py::green_renormalization_python` loops forever when the
  decimation turns into NaN (`NaN < error` is never true). A finiteness check
  that raises would let the $\delta$ escalation of `smatrix._lead_selfenergies`
  take over. Reproducer: a chain lead with Rashba 0.6, Zeeman
  $0.15(-0.6,0,0.8)$, onsite 2, s-wave 0.1, at $E=0.23164556962025312$,
  $\delta=10^{-12}$.
- The RKKY routine (`chitk/magneticresponse.py`): its default
  `delta=1/nk` never converges because the temperature shrinks with the mesh,
  the denominator carries a second $i\delta^2$ broadening, and the cost is
  $O(n_k^4)$ per distance. $J(R)=-T\sum_n G(R,i\omega_n)^2$ for all $R$ at
  once by FFT is the fix.
- The degenerate-pair term overcounts by $O(f'/n_k)$ when `delta` resolves
  the mesh and the mesh contains the nesting points (use an odd $n_k$); a
  conserving (Mermin, PRB 1, 2362) form would remove it.
- `lorentz_kernel` has $\lambda=3$ fixed, which caps the KPM Green's function
  at 5-10% of the matrix inversion; a `lambda` argument would close it.
- Conventions still inconsistent: the SCF keeps $V_r(r,r)$ as an onsite term
  while `density_interaction` drops it; `get_berry_curvature` returns the
  lattice-gauge curvature (topology 07 explains it); the Heterostructure
  S-matrix keeps its central $i\delta$ on purpose, since bound states need it.
- In mean-field 08, states with $\mathrm{Re}E$ exactly at the Fermi level and
  purely imaginary energies get occupation 1/2, where Hayata-Yamamoto fill the
  $\mathrm{Im}E<0$ one; the two agree on the Néel branch the notebook shows.
  Without the no-in-plane constraint the SCF finds an in-plane order there that
  was not checked.
- Notebook 15's small-$\Delta$ dispersive corner needs nk~48 (30 min) to
  converge; the Wannier-Mott hydrogenic comparison of mean-field 17 was not
  done (the Coulomb tail is cut at `rcut=5`).
- The Nambu `VJinteraction` loop from `mf="random"` (bichain, $V_1=-2$,
  exchange 0.3) does not converge within `maxite` from every random start;
  `tests/scf/test_spinspin_nambu.py` failed when `tests/scf` ran as a whole,
  also on the code before this pass, and is now seeded. Which starts fail,
  and whether more mixing would fix them, was not looked into.
- KPM 05's "about 6 GB" peak memory for $10^7$ sites is from the run before
  this pass and was not measured again (the KPM call now takes 109 s, 1.85
  times the polynomials at the same `delta`).
- Cosmetic, left as they are: mean-field 08's $\kappa=0.30$ point is half
  cut at the heatmap's edge; tutorials 08 and 10 keep em-dashes and a
  contraction in their original sentences.
- Small: `mf="pwave"` seeds opposite-spin pairing, a poor start for
  equal-spin states; the SCF's default `mix=0.1` is 4 times slower than 0.5 on
  the equal-spin triplet; `keldyshtk/current.py` has a stale docstring about
  the local probe's broadening; the guide's MAR section calls `build(h1,h2)`
  an SNS junction.
