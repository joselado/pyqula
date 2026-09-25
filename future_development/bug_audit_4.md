# Bug audit 4: the superconductivity routines, 2026-09-24

The fourth sweep asked one question, whether the superconductivity routines
are correct: the Nambu construction in `superconductivity.py`, the pairing
modes of `sctk/pairing.py`, the d-vector and singlet/triplet extraction, and
their consumers where the two earlier sweeps left them unexercised. What the
second sweep's L2 lens cleared (`bug_audit_2.md`, "Chased and cleared") was not
re-derived. Every check ran against an oracle independent of the code under
test, a closed-form spectrum, an exact many-body diagonalization or a formula
from the theory of Andreev reflection, and every fix has a regression test that
fails on a `git archive` copy of the unfixed source.

The Nambu spinor throughout is pyqula's $(c_\uparrow, c_\downarrow,
c^\dagger_\downarrow, -c^\dagger_\uparrow)$ per site, in which the pairing
block of a site pair is $D_{ij} = \psi_{ij} + \mathbf{d}_{ij}\cdot\sigma$ and
Fermi antisymmetry reads $D_{ij} = \sigma_y D_{ji}^T \sigma_y$, meaning that the
singlet $\psi$ is even under $i \leftrightarrow j$ and the d-vector odd.

## Status

**Seven findings, all closed.** Four of them were decisions the maintainer took
on 24 September 2026 rather than repairs: remove the four pairing modes that
break Fermi antisymmetry (#1) and the graphene f-wave helper built the same way
(#3), make `hopping2deltaud` right for any hopping instead of refusing a
non-symmetric one (#2), make `decompose=True` raise under the finite difference
(#6), and remove the spinless Nambu Hilbert space altogether, so that a Nambu
Hamiltonian is always spinful (#7).

The full suite on the result collects 2087 tests: 2081 pass and 6 fail, in 37
minutes. The six fail on an untouched copy of HEAD as well, so they are not
from this sweep, and each of them fails alone in an empty directory with its
mean-field loop stopping at the `maxite=1000` default that `bug_audit_3.md`
#19 introduced, where it used to run until it converged. For the supercell
test the reason is measured: the same calculation with `maxite=None`
converges to `maxerror=1e-8` from its random guess in 27 s and 19 s for the
two cells, so it simply needs more than 1000 iterations. That test did pass
once in this session, run from the repository root, where a stale `MF.pkl`
(read by default, `load_mf=True`) can seed the loop close to the solution. The
six are
`test_magnon_bands_scan_a_path_and_reach_zero_at_gamma` and
`test_the_rpa_and_tdhf_methods_are_both_reachable` in
`tests/magnon/test_dispersion.py`,
`test_densitydensity_jax_newton_handles_filling`,
`test_superconducting_energy_per_atom_matches_in_a_supercell`,
`test_sc_gap_vs_temperature_is_bcs_like_and_mesh_independent` and
`test_vjinteraction_jax_handles_filling`. This is the complete run that
`bug_audit_3.md` said was still owed, and what it found is these six. They
were removed afterwards, the maintainer's call, which leaves 2081 tests;
`tests/scf/test_scf_sc_critical_temperature.py` held only one of them and is
gone as a whole.

## User-visible changes

- **Breaking:** the pairing modes `"haldane"`, `"antihaldane"`, `"swavez"` and
  `"SnnAB"` are gone, and `add_pairing` raises `ValueError` on them (#1).
- **Breaking:** `specialhamiltoniantk/graphene.py` and its `add_fwave` are
  gone (#3). The second-neighbor f-wave it was after is
  `hopping2deltaud(h,1j*t)` with `t` an anti-Haldane hopping, a pure $d_z$
  triplet now that #2 is fixed.
- **Breaking:** there is no spinless Nambu mode (#7). `add_swave`,
  `setup_nambu_spinor` and `add_pairing` on a spinless Hamiltonian make it
  spinful first, as `turn_nambu` always did, so every quantity computed from
  such a Hamiltonian now counts both spin species: its superfluid weight and
  its entanglement entropy are twice the old spinless numbers.
  `scftypes.attractive_hubbard`, the spinless attractive-Hubbard loop, is
  gone; `h.setup_nambu_spinor()` followed by
  `h.get_mean_field_hamiltonian(U=...,mf="swave")`, with `mu=` for a fixed
  chemical potential, replaces it. `check_mode("spinless_nambu")` raises, and
  `h.check()` refuses a Hamiltonian with `has_eh=True` and `has_spin=False`.
- `hopping2deltaud(H,T)` builds the physical up-down pairing for any `T`: the
  symmetric part of the hopping gives the singlet and the antisymmetric part
  a $d_z$ triplet. A real symmetric `T` gives exactly what it gave before (#2).
- `"swaveA"`, `"swaveB"` and `"swavesublattice"` raise `ValueError` on a
  geometry without a sublattice, where they used to add zero pairing or crash
  inside `get_index` (#4).
- `superfluid_weight(mode="finite_difference",decompose=True)` raises
  `ValueError`; it used to return the bare tensor (#6).
- `get_eh_sector(m,i,j)` raises `ValueError` for an index outside 0 and 1; it
  used to return the `NotImplemented` singleton (#5).

## Findings

### 1. Four pairing modes break Fermi antisymmetry, and their BdG spectrum is not the spectrum of any Hamiltonian

**Cause.** `"haldane"` and `"antihaldane"` put the Haldane function, odd under
$i \leftrightarrow j$, on the spin-singlet channel; `"swavez"` is an onsite
$\sigma_z$, a triplet that has to vanish on a single site; `"SnnAB"` is a
first-neighbor $\sigma_z$ that is odd on a bipartite lattice and even on any
other, where it gives $-\sigma_z$ on every bond. The part of a BdG matrix that
breaks the antisymmetry adds only a constant to $\frac{1}{2}\Psi^\dagger H
\Psi$, so the operator carries no pairing from it, and yet the BdG spectrum
shows a gap.

**Oracle.** The matrix identity $U H_R^* U^\dagger = -H_R$ for every real-space
block, with $U = \tau_y \sigma_y$ in the spinor above, and an exact
diagonalization of $\frac{1}{2}\Psi^\dagger H \Psi$ built from Jordan-Wigner
fermion operators.

**Before.** Residuals of the identity of 0.52 (`"haldane"`, `"antihaldane"`
on the honeycomb lattice), 0.6 (`"swavez"` anywhere, `"SnnAB"` on the square
lattice) against 1e-16 for every other mode. On a single site with $\mu=0.4$
and $\Delta=0.3$, `"swavez"` reports a gap of 0.5 where the exact many-body
excitation is 0.4, which is no pairing at all, while `"swave"` gives 0.5 in
both.

**Status.** The four modes were removed, which was the maintainer's call; the
registry's error message lists the ones left. A comment in `sctk/pairing.py`
states the requirement. `test_every_pairing_mode_obeys_fermi_antisymmetry`
asks for the identity on every registered mode on two lattices with Rashba, a
Zeeman field and a generic d-vector.

### 2. hopping2deltaud used t_R for both spin blocks

**Cause.** The routine turns the hoppings $t_{ij}$ of a Hamiltonian into the
pairing $\sum t_{ij} c^\dagger_{i\uparrow} c^\dagger_{j\downarrow} + h.c.$, and
put $t_R$ in both the up-down and the down-up block of the pairing matrix. The
down-up block describes $c^\dagger_{j\uparrow} c^\dagger_{i\downarrow}$, so the
same operator needs $(t_{-R})^T$ there, which equals $t_R$ only for a symmetric
hopping.

**Before.** A Haldane hopping gave an identity residual of 0.052 on the
honeycomb lattice.

**Status.** Fixed with $(t_{-R})^T$, which the maintainer preferred to
refusing a non-symmetric hopping. In $k$-space the up-down block is now exactly
$t(k)$ and the down-up block $t(-k)^T$, the identity holds to 0, a real
symmetric hopping gives what it gave before (the existing comparison with
`add_pairing` still passes) and `1j` times an anti-Haldane hopping gives a
triplet with a singlet part of exactly zero. Two tests in
`tests/superconductivity/test_hopping2delta.py`.

### 3. graphene.add_fwave was built on #2

**Before.** Identity residual 0.35. It fed `1j` times a Haldane hopping to the
old `hopping2deltaud`, an odd singlet. Nothing in the package called it.

**Status.** Removed with the file, the maintainer's call.

### 4. The sublattice pairing modes on a geometry without a sublattice

**Before.** On the square lattice (`has_sublattice=False`, `sublattice=[0.]`)
`"swaveA"`, `"swaveB"` and `"swavesublattice"` added zero pairing and said
nothing; on its supercell they raised `IndexError`, and on the triangular
lattice `AttributeError`.

**Status.** Fixed with the shared `require_sublattice` guard. On the honeycomb
lattice `"swaveA"` still pairs the A sites alone and `"swaveB"` the B sites.

### 5. get_eh_sector returned NotImplemented for an index above 1

**Status.** Fixed by letting the `ValueError` below it fire.

### 6. superfluid_weight dropped decompose=True under the finite difference

**Status.** It raises, the maintainer's call; the docstring had documented the
argument as ignored.

### 7. Two Nambu Hilbert spaces for one spinless Hamiltonian

**Cause.** `turn_nambu()` made a spinless Hamiltonian spinful before the
doubling, four components per site, while `setup_nambu_spinor()` and
`add_swave()` built a spinless Nambu one, $(c, c^\dagger)$ per site, two
components. About twenty places in the package then had to refuse the second
one (the d-vector, the pairing operators, the KPM and qtci density matrices,
dI/dV, the mean-field constrains, the real-space LDOS, strain), and a few
treated it (the superfluid weight, the filling, the entanglement entropy).

**Status.** Decided by the maintainer: only spinful Nambu exists.
`sctk/spinless.py` and `scftk/attractive_hubbard_spinless.py` are gone, every
spinless-Nambu branch and guard is gone, `check_mode` no longer knows the
name, and `h.check()` refuses a hand-built one. This supersedes
`bug_audit_3.md` #18, the spinless Nambu `get_mean_field_hamiltonian`
returning `None`, since that Hamiltonian can no longer be built. The examples that used the
spinless loop (`examples/2d/comparison_scf_swave`, `SC_phase_diagram`) run the
spinful SCF at fixed chemical potential; a reduced phase diagram gives $\Delta$
falling with temperature at both chemical potentials it samples. The tests
that pinned a spinless-Nambu refusal now pin the promotion:
`tests/superconductivity/test_nambu_is_spinful.py`, and in the superfluid,
entanglement, operator, identification and dI/dV tests the spinless input
gives the spinful answer.

## Chased and cleared

- **The BdG spectrum against closed forms.** All to 1e-14 or better: s-wave
  with a Zeeman field in a generic direction,
  $E = \pm\sqrt{\xi^2+\Delta^2} \pm |B|$; the Anderson theorem, $E =
  \pm\sqrt{\epsilon_n^2+\Delta^2}$, for Kane-Mele plus Rashba plus an onsite
  modulation on a honeycomb supercell; a triplet on a chain for all nine
  combinations of d-vector and field axis, pair broken only when the field is
  along d; $\mathbf{d}=(1,i,0)$ pairing the up spins alone, with the down
  electrons unpaired at $\pm(\xi - B)$ and the non-unitarity along $+z$; and
  chiral p, $d_{x^2-y^2}$ and extended s on the square lattice at $\mu=-0.7$
  with their first-neighbor form factors. This pins the physical spin labels
  of the d-vector, which the L2 round trip (add d, extract d) could not see,
  since a global relabeling would pass it. `test_analytic_bdg_spectra.py`.
- **Order of operations.** Adding onsite energy, Zeeman, exchange, Rashba,
  Kane-Mele, Haldane, modified Haldane, anti-Kane-Mele, sublattice imbalance,
  antiferromagnetism or crystal field before or after `turn_nambu()`, and
  before or after `add_swave`, gives the same matrix to 0.
- **Supercells.** Pairing then supercell and supercell then pairing give the
  same spectrum for every mode on the honeycomb lattice and every
  non-sublattice mode on the square and triangular ones.
- **Andreev reflection through a barrier.** Beenakker's zero-bias
  $G_{NS} = 4T^2/(2-T)^2$ for a single spin-degenerate channel, with the
  normal transmission $T$ read from the same junction without pairing, holds
  with a deviation of order $\Delta$ over the bandwidth: 9.8e-3 at
  $\Delta=10^{-2}$ and 9.8e-4 at $10^{-3}$, for couplings 0.8, 0.5 and 0.3.
  The normal $T$ matches $4c^2/(1+c^2)^2$ exactly. The existing Andreev test
  only covered the transparent contact. `test_zero_bias_conductance.py`.
- **Exact many-body diagonalization.** The BdG eigenvalues of `"swave"` on a
  site and of `"chiral_dwave"` on an eight-site cluster match the exact
  excitation energies of $\frac{1}{2}\Psi^\dagger H \Psi$.
- **A note for the next sweep.** A spectrum-level particle-hole check,
  $E_n(k) = -E_n(-k)$, cannot see a broken Fermi antisymmetry: an odd singlet
  passes it, which is how the four modes of #1 survived L2's sweep of every
  mode. The identity $U H_R^* U^\dagger = -H_R$ is the one to run, and
  `h.check()` already runs it through `superconductivity.eh_operator`.

## Left open

- What the six removed tests pinned is untested now: the BCS-like shape of
  the self-consistent gap against temperature, independent of the k-mesh
  (`bug_audit_2.md` #63), the total energy per atom of a paired mean field
  being the same in a supercell, the magnon dispersion along a path, and the
  jax filling targets. Restoring any of them needs a guess that converges in
  fewer than 1000 iterations, or `maxite=None` in the test.
- From L2's list, the decomposition's degenerate-band branch has not been
  run on a Rashba BdG. The duplicate `C3nn` in `sctk/pairing.py` is gone.

## Closed after the sweep

The small items this list used to carry were closed on 25 September 2026,
each with a test that fails on the source before the change:

- `add_pairing(mode=callable)` checks the callable for Fermi antisymmetry,
  $D(\vec r_1,\vec r_2)=\sigma_y D(\vec r_2,\vec r_1)^T\sigma_y$, on every pair of
  positions `add_pairing` evaluates, inside the cell and between the cell and
  each neighboring replica, before the Hamiltonian is touched, and raises
  `ValueError` naming the first pair that breaks it and by how much. Every
  registered mode passes it to 1e-15 on a honeycomb, square and triangular
  lattice and a chain, and the odd singlet and onsite triplet shapes of #1
  fail it by 2, which is what settled that the rule applies to the 2x2 weight
  the callable returns. `tests/superconductivity/test_callable_pairing_antisymmetry.py`.
- `MultiHopping.get_dagger` writes the dagger of every block at $-R$ instead
  of looking up the partner there, so a lone `(1,0,0)` block gets its dagger
  at `(-1,0,0)` and `m + m.get_dagger()` is Hermitian for any `m`. The
  comment in `operatortk/inplane_valley.py` that described the old behaviour
  now says the check there is only a guard. `tests/hopping/test_multihopping_dagger.py`.
- `bkt_temperature` doubles the bracket, up to `maxexpand=20` times, when
  the stiffness is still above the Nelson-Kosterlitz line at `tmax`, and
  raises `ValueError` when it never crosses, where it used to return `tmax`.
  With the stiffness replaced by a closed form rising as $g(T)=1+T-T^2/5$ it
  finds the crossing at $\sqrt 5$, where the old code returned the bound 1,
  and a `tmax=0.1` given below a crossing at 2/3 comes back as 2/3.
  `tests/superfluid/test_bkt_bracket.py`.
- `kanemele.get_haldane_function` and `dvector_times_mij_map`, the
  byte-identical copy of `dvector_times_rij_map`, are deleted; neither had a
  caller in `src/`, `tests/`, `examples/`, the notebooks or the
  documentation.
