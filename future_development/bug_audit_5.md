# Bug audit 5: the KPM mean field, the density matrix and the BSE, 2026-09-25

The fifth sweep started where the lenses of the second one said they had
stopped, at four places that no sweep had run against an independent oracle:
the mean-field loop with the kernel polynomial method as its density-matrix
engine (`integration="kpm"`), every routine outside `magnetism.py` and
`scftk/kondolattice.py` that consumes a density matrix, the different code
paths that compute the same density matrix, and the Bethe-Salpeter package
`bsetk/`, which only the static scanners of the second sweep had seen. One
auditing agent per area ran real calculations against a reference that shares
no code with what it checks, a closed form, an exact diagonalization or a
construction from eigenvectors written in plain numpy, and a second agent per
area then tried to refute every finding, by re-running it and by writing a
check of its own on a different system. Every run was on the CPU, against
`dd3818e`. The reproduction scripts lived in a session scratchpad and are not
preserved here; each entry says what was run and what it printed.

| Area | What it covers |
| --- | --- |
| KPM mean field | `scftk/densitydensity_kpm.py`, `kpmtk/densitymatrix_kpm.py`, the KPM branch of `spinspin._run_anisotropic_scf`, `get_total_energy_kpm` |
| Density-matrix consumers | `vev.py`, the expectation-value methods of `hamiltonians.py`, `spectrum.py`, `spinon.py`, `entanglementtk/`, `embedding.py`, `embeddingtk/`, `topologytk/realspace.py`, `kpmtk/density.py` |
| Density-matrix backends | `densitymatrix.py` (accumulate, simultaneous, sparse, with-Fermi and local-Fermi routes and their fallbacks), `dmtk/fulldm.py`, `dmtk/fulldmjax.py` on jax's CPU backend |
| BSE | `bse.py` and `bsetk/`: the dense, iterative and quantics solvers, the gauges, the screening, the spin-flip magnons |

## Status

**Sixteen findings, none fixed yet.** The auditors filed fifteen and the
refuting agents confirmed all fifteen; the sixteenth (#9) is a crash one
refuting agent ran into on the way, re-run afterwards. Three were reproduced
again by hand after the sweep, on systems neither agent had used: the
projection gauge (#1) gives a lowest exciton of 1.8346 where the raw gauge
gives 1.8629 on a second Rashba chain at `nk=6`, and 1.8286 against 1.8612 at
`nk=12`; `get_single_vev("sy")` on a three-site Rashba island with an s-wave
term of zero gives -1.8015 where the normal Hamiltonian and `get_vev` give
-0.9008 (#2); and `get_vev("spair")` returns `[0,0,0]` on the same island at
$\Delta=0.4$, where `get_single_vev` gives -0.502 (#3).

The sweep also closes the note the second audit left open on this engine,
whether the doubled BdG total energy it fixed in the exact-diagonalization
engine also lives in `get_total_energy_kpm`. It does not, since that function
is unreachable for a Nambu Hamiltonian: both `get_total_energy_kpm` and the
KPM branch of `_run_anisotropic_scf` raise `NotImplementedError` on `has_eh`,
and the Nambu energy of `Vinteraction_kpm` goes through `h.get_total_energy`,
which carries the correction.

## What to read first

By the ordering the first sweep settled on, a silently wrong number before a
crash:

1. **#1, the projection gauge of the BSE.** `solver="qtt"` uses it by
   default, and on any model whose valence or conduction window holds more
   than one non-degenerate band it returns a lowest exciton that is wrong by
   a fixed amount, which lies below the exact one.
2. **#2 and #3, expectation values on a Nambu Hamiltonian.** Three of the
   expectation-value methods count every normal observable twice, and the one
   that counts them once returns zero for every pairing operator.
3. **#4, the KPM Fermi level in `VJinteraction`.** At a finite temperature
   the spinful KPM mean field converges to the wrong number of electrons.
4. **#5, the KPM mean-field loop amplifies its own roundoff.** A Hermitian
   start does not prevent the divergence.

## Findings

### 1. The projection gauge mixes non-degenerate bands but keeps their energies diagonal

**Cause.** `bsetk/gauge.py:127-139` (`_projection_gauge`) rotates the whole
valence window and the whole conduction window as one block each,
`out[ik][grp] = (u@vh).T@P`, whether or not the bands inside are degenerate.
`PairBasis.build` (`pairbasis.py:123-124`) and `PairOracle` then pair the
rotated vectors with the unrotated energies, `dE[m] = ekq[ik][ic] -
ek[ik][iv]`. The one-body part in the rotated states,
$\langle w_c|H(k+Q)|w_{c'}\rangle\delta_{vv'} - \langle w_{v'}|H(k)|w_v\rangle\delta_{cc'}$,
is diagonal only if the rotation stays inside a degenerate multiplet, so the
matrix that gets diagonalized is not a unitary transform of the BSE block.
The docstrings of `gauge.py`, `pairbasis.py` and `solve.py` state that the
gauge is a block-diagonal unitary on the pair index and leaves the spectrum
invariant, which is exactly the claim that fails. `gauge="auto"` switches the
projection gauge on for `solver="qtt"` only, so the dense and iterative
solvers are right by default and wrong with an explicit `gauge="projection"`,
which also hands the rotated states to `static_polarizability` when
screening is on.

**Oracle.** Three references. The raw gauge and the phase gauge (a diagonal
phase, which cannot mix bands) agree to 4e-15 and with `solver="iterative"`
to 1e-15. A Tamm-Dancoff CIS written from scratch on the $N$-cell ring in
real space, from `h.intra`, `h.hopping` and the real-space interaction, with
no Bloch conventions and no `bsetk` code, reproduces the union over the $N$
values of $Q$ of the raw-gauge spectrum to 2e-14 and misses the projection
one by 0.078. The mechanism check: replacing `diag(dE)` by the rotated
one-body matrix above brings the projection spectrum back onto the raw one to
1.2e-14, and the off-diagonal part of that matrix is 0.906.

**Reproduction.** A spinful two-site chain with a staggered onsite 0.9,
Rashba 0.3 and an exchange field along [0.3,0.5,0.2], which leaves no band
degenerate, with $U=1$ and $V_1=0.3$, in the Tamm-Dancoff approximation:

```
nk= 8 lowest exciton: raw 0.9480658375 phase 0.9480658375 projection 0.9344269323
nk=16 lowest exciton: raw 0.9464208867 phase 0.9464208867 projection 0.9324769537
nk=32 lowest exciton: raw 0.9464430001 phase 0.9464430001 projection 0.9324957062
nk=16 solver=qtt (default gauge) 0.9324769538   solver=iterative 0.9464208867
nk=32 solver=qtt (default gauge) 0.9324957062   solver=iterative 0.9464430001
nk=16 solver=qtt gauge=phase     0.9464208867
```

With one band per window (`nv=nc=1`) the projection gauge agrees with the raw
one to 2e-15, and on the z-collinear ferromagnetic chain it swaps the up and
down valence labels at every k and moves the spectrum by 0.09. The suite does
not see it because every gauge and qtt test uses a model whose windows are
either exactly degenerate or one band wide.

**Fix.** Restrict the projection rotation to the degenerate multiplets inside
each window, with the grouping `spinflip._degenerate_groups` already uses and
a tolerance on the energies; on a non-degenerate band it then reduces to a
phase, which is exact. The alternative is to carry the rotated one-body
matrix in place of `diag(dE)`, which removes the diagonal structure the
quantics and iterative solvers are built on. Whether the first choice costs
the quantics solver any rank is measured nowhere yet (`bse_excitons.md` has
the rank table for the models it tried). The docstrings need correcting in
either case. A regression test: the dense spectrum with
`gauge="projection"` equals the one with `gauge=None` on the chain above, and
`solver="qtt"` matches `solver="iterative"` on its lowest exciton.

**Status.** Open.

### 2. get_single_vev, get_several_vev and get_dm_vev count twice on a Nambu Hamiltonian

**Cause.** `get_single_vev` and `get_several_vev` (`hamiltonians.py:1144-1159`)
hand the operator straight to `spectrum.ev` or `vev.kresolved_orbital_vev`,
and `vev.get_dm_vev` (`vev.py:6-19`) contracts it with the full Nambu density
matrix. None of them restricts it to the electron sector, as `get_vev`
(`hamiltonians.py:1004-1006`) and `real_space_vev` do since `bug_audit.md`
1.3, so the sum over every negative-energy BdG state counts each physical
one-body observable twice. `bug_audit.md` 1.3 repaired the two methods it
found and not these siblings.

**Oracle.** The occupied eigenvectors of the normal Hamiltonian summed by
hand with `numpy.linalg.eigh`, and the normal description of the same state,
since a BdG copy at zero pairing describes exactly the same state. A second
check read the electron block of the BdG correlation matrix $C=V^*V^T$ by
hand at $\Delta=0$ and $\Delta=0.3$.

**Reproduction.** A honeycomb lattice with Rashba and an exchange field along
[0.3,0.5,0.2], `nk=12`:

```
sy normal single -0.17076542 | BdG single -0.34153084 | BdG several [-0.34153084] | BdG get_vev -0.17076542
sz normal single -0.09091630 | BdG single -0.18183260 | BdG several [-0.1818326]  | BdG get_vev -0.09091630
BdG get_single_vev(matrix-free sy) -0.3415308420049085
```

On a six-site island `get_dm_vev` gives -2.19215 for `sy` where the normal
one gives -1.09608. The factor is exactly 2 at finite pairing as well.

**Fix.** The same electron-sector branch as `get_vev`, shared rather than
copied three times, and in a form that keeps the anomalous block, since
copying `pe*op*pe` would carry #3 into the three siblings. A regression test:
for `h` with `add_swave(0.0)` and a random Hermitian matrix as well as
`sx`, `sy`, `sz`, all three agree with the normal Hamiltonian.

**Status.** Open.

### 3. get_vev of a pairing operator on a BdG Hamiltonian is identically zero

**Cause.** The same line that fixes the double counting of #2 for `get_vev`,
`op = pe*op*pe` for every operator when `has_eh` is set
(`hamiltonians.py:1004-1006`), annihilates any operator that lives only in
the electron-hole off-diagonal block, so `spair`, `deltax`, `deltay`,
`deltaz` and `singlet` all come back as a silent zero from the method the
guide recommends for site-resolved expectation values.

**Oracle.** On a single site with $\epsilon=0.3$ and $\Delta=0.4$, the exact
ground state of $\epsilon(n_\uparrow+n_\downarrow) + \Delta
c^\dagger_\uparrow c^\dagger_\downarrow + h.c.$ in its four-state Fock space
gives $|\langle c_\downarrow c_\uparrow\rangle| = \Delta/2E = 0.4$; on a
chain, the Nambu correlation matrix built by hand from the negative-energy
eigenvectors gives $\langle c_\downarrow c_\uparrow\rangle = -0.190$.

**Reproduction.**

```
single site:  get_vev spair [0.]   get_vev deltax [0.]   get_single_vev spair -0.8
s-wave chain: get_vev spair [0.]   get_single_vev spair -0.3802   oracle <c_dn c_up> -0.1901
```

`get_single_vev` gives twice the correlator, and that factor 2 is the
singlet structure of `spair` (both $P_{02}$ and $P_{13}$ pick the same
correlator in the spinor $(c_\uparrow, c_\downarrow, c^\dagger_\downarrow,
-c^\dagger_\uparrow)$), not a Nambu double count, so a fix for #2 must not
halve it.

**Fix.** Either restrict to the electron sector only the operators that
conserve particle number, or replace the projection by the general Nambu rule
(half the full Nambu trace, plus the constant from the hole block), which
counts a normal observable once and keeps the anomalous block. Which of the
two is the maintainer's call, and it decides the shape of #2's fix as well. A
regression test: `get_vev("spair")` on an s-wave chain equals the correlator
built by hand, per site, and is zero at zero pairing.

**Status.** Open.

### 4. VJinteraction with integration="kpm" finds the Fermi level at T=0

**Cause.** `spinspin.py:1234` calls `get_fermi4filling_kpm(h, filling, nk=nk,
scale=scale, npol=npol, ne=ne, cores=cores)` without `T=`, so it takes the
$T=0$ step count, while `_get_dm_kpm` builds the density matrix with the
Fermi-Dirac weight at `T`. `bug_audit_2.md` #19 gave `get_fermi4filling_kpm`
its finite-temperature inversion and made `densitydensity_kpm`'s callback
pass `T`; this second call site was never listed. It is the route of every
spinful `h.get_mean_field_hamiltonian(integration="kpm")`, and the electron
count enters the total energy too, which adds `fermi*N*filling` with the
requested filling.

**Oracle.** The electron count of the returned Hamiltonian from an
exact-diagonalization density matrix, and from a Fermi-Dirac sum over
`eigvalsh` on a k-grid built by hand, against twice the filling. The exact
engine and `Vinteraction_kpm`, whose callback does pass `T`, both land on the
right count on identical inputs, and every engine is exact at $T=10^{-7}$.

**Reproduction.** A spinful chain, `filling=0.1`, `T=0.05`, $U=0.2$,
`nk=30`:

```
VJ kpm  T=0.05  npol=300 N=0.191455   npol=900 N=0.191900   requested 0.200000
VJ ed   T=0.05  N=0.200000
V_kpm   T=0.05  npol=300 N=0.200011   npol=900 N=0.200001
VJ kpm  T=1e-7  N=0.200000
```

The error does not close with `npol` and does not depend on $U$ (0.1910 at
$U=0$). Passing `T` to that one call brings the count to 0.20000.

**Fix.** Pass `T=T` at `spinspin.py:1234`. A regression test: the trace of
the onsite density matrix of the returned Hamiltonian at `T=0.05` equals 0.2
to 1e-3.

**Status.** Open.

### 5. The KPM mean-field map amplifies the anti-Hermitian part of the mean field

**Cause.** `_dm_kpm_from_needed` (`kpmtk/densitymatrix_kpm.py:286-317`)
evaluates every requested entry $(d,i,j)$ from its own Chebyshev moments, and
`required_elements` asks for both $(d,i,j)$ and $(-d,j,i)$, so the two members
of a Hermitian pair are computed independently and nothing forces
$\rho_{-d}[j,i] = \rho_d[i,j]^*$. Once $H(k)$ carries any anti-Hermitian
part, $T_n(H(k))$ is not Hermitian, and the mean field feeds that part back
with a gain above one, meaning that roundoff at 1e-18 grows by a fixed factor
per iteration until the Hamiltonian is non-Hermitian at order one and the
Chebyshev recursion overflows. The docstring of `random_hermitian_guess`
(`scftk/densitydensity.py:143`) attributes the KPM blow-up to a
non-Hermitian initial guess, which a Hermitian one cures; roundoff reseeds it
every iteration, so that cure does not reach this.

**Oracle.** The exact Hermiticity of the mean field of a Hermitian
Hamiltonian, $\mathrm{mf}_d = \mathrm{mf}_{-d}^\dagger$, which needs no
reference code. The exact engine keeps the anti-Hermitian part at 1e-18,
since `eigh` builds a Hermitian density matrix by construction, and a control
that changes only one thing, symmetrizing the KPM density matrix through the
public `callback_dm` hook, makes the same runs converge with an
anti-Hermitian part of exactly zero.

**Reproduction.** A spinless Haldane honeycomb, $t_2=0.15$, $V_1=1.5$,
$V_2=0.5$, `filling=0.3`, `nk=8`, `npol=300`, from a random Hermitian guess at
the default `mix=0.1`:

```
exact engine          converged, E=-0.768363
KPM as it is          not converged after 600 iterations, E=1.9e302
                      anti-Hermitian part of the dm at it 0, 50, 100, 150, 200:
                      5.7e-17 6.7e-16 1.8e-13 8.0e-11 3.6e-08
KPM, dm symmetrized   converged in 450 iterations, E=-0.768087
```

The growth per iteration, measured by seeding 1e-10 at the exact fixed point,
is 1.64 and 1.83 at `npol=300` and 900 on this model, 1.73 and 1.58 at
`npol=300` and 600 on a Haldane honeycomb at $V_1=1$, and 1.23 and 1.20 on a
real spinless square lattice, so it is above one at every `npol` and appears
on a real Hamiltonian too; the symmetrized control has a growth of 0.66 to
0.74. `VJinteraction(integration="kpm")` on a spinful Haldane lattice goes
from 6e-9 after 5 iterations to 1.55 after 25. On a Rashba chain with $U$ and $V_1$ the
same seed decays, so whether a given run diverges depends on the system, and
a run that does converge, as one at `npol=900` from the exact fixed point
did, still returns a Hamiltonian with an anti-Hermitian part of 2.8e-9. The
gap of 3e-4 left between the symmetrized KPM energy and the exact one is
attributed to the Jackson kernel not yet resolving the discrete k-levels at
`nk=8`, roughly halving from `npol=300` to 900; a run at `npol=2500` did not
finish.

**Fix.** Symmetrize once per call inside `get_dm_kpm` or
`_dm_kpm_from_needed`, or compute one member of each pair and set the other by
conjugation, which also halves the work, so that both KPM engines inherit it.
The docstring of `random_hermitian_guess` should say what it does and does
not cover. A regression test: from the exact fixed point plus a 1e-10
anti-Hermitian seed, 30 iterations of `Vinteraction_kpm` on the Haldane
model above keep $\max|\mathrm{mf}_d - \mathrm{mf}_{-d}^\dagger|$ below
1e-12, and the same for `VJinteraction(integration="kpm")`.

**Status.** Open.

### 6. The embedding density matrix is the transpose of every other one

**Cause.** `embeddingtk/embedded.get_dm` (`embedded.py:59-66`), reached as
`Embedding.get_density_matrix` and `Embedded_Hamiltonian.get_density_matrix`,
integrates the resolvent, whose spectral weight is $\sum|n\rangle\langle n|$,
so it returns the one-body projector $\rho_{ij} = \sum_{occ}\psi_i\psi_j^*$.
`Hamiltonian.get_density_matrix` returns $\rho^T$, the full_dm convention,
and its docstring prescribes `sum(dm*A)` for an expectation value. Neither
convention is wrong by itself; three methods with the same name and opposite
conventions, with nothing saying so, are the defect. Nothing inside the
package consumes an embedding density matrix today (see #15).

**Oracle.** A single spin in a field along [0.2,0.7,-0.3], where the closed
form is $\rho = (1-\hat b\cdot\sigma)/2$, and a finite chain of up to 2401
sites with the same defect in the middle, diagonalized with `eigh`.

**Reproduction.**

```
single spin: closed-form rho offdiag -0.127+0.4445j   h.get_density_matrix -0.127-0.4445j
             Embedded_Hamiltonian -0.12703+0.4446j
Rashba island, sum(dm*sy) the documented way: oracle -0.97277, embedded dm +0.97280
defect in an infinite chain: Embedding -0.0835+0.1285j, finite-chain rho -0.0838+0.1284j,
                             full_dm block -0.0838-0.1284j
```

**Fix.** Transpose in `get_dm`, so that every `get_density_matrix` shares
full_dm's convention, or document the convention in both embedding
docstrings; the maintainer's call. A regression test: for a 0d complex
Hamiltonian, `Embedded_Hamiltonian(h).get_density_matrix()` agrees with
`h.get_density_matrix()` to the broadening.

**Status.** Open.

### 7. The quadrature tolerance of the embedding density matrix is fixed and swallowed

**Cause.** `get_dm` calls `integration.complex128contour` without `eps`, so
its adaptive Simpson rule runs at an absolute tolerance of 1e-2 on a matrix
whose entries are of order 0.1 to 1. An `eps=` passed by the caller goes
through `**kwargs` into `get_gf` and is dropped there without a word.

**Oracle.** The finite chain of #6 at 601, 1201 and 2401 sites, converging to
0.20898 and 0.35529 on the diagonal of the defect block, and a 400-point
Gauss-Legendre integration of the same `get_gf` on the same contour, which
gives 0.20902 and 0.35532 at `delta=1e-3`.

**Reproduction.**

```
eps 1e-2 (the fixed default), delta 1e-3  diag [0.21229 0.35802]
eps 1e-3,                     delta 1e-3  diag [0.20901 0.35531]
eps 1e-4,                     delta 1e-3  diag [0.20902 0.35532]
get_density_matrix(delta=1e-3, eps=1e-4)  diag [0.21229 0.35802]   (eps swallowed)
default call at nk=100 and nk=400         identical
```

The error is quadrature noise rather than a fixed offset: scanning the lower
end of the contour, `emin`, hardcoded at -10, moves it erratically between
7e-4 and 1e-2, and it vanishes at `emin=-20`. On a second chain it was 5e-4
or less. Neither `delta` nor `nk` converges it away.

**Fix.** Give `get_dm` an `eps` keyword with a default around 1e-4, passed to
`complex128contour` and kept apart from what goes to `get_gf`. A regression
test: the defect block agrees with a large finite chain to 1e-3 at
`delta=1e-3`.

**Status.** Open.

### 8. kpmtk.density.get_density is right only at fermi=0

**Cause.** `kpmtk/density.py:12-35` passes `fermi` to the arccos without
dividing it by `scale`, while the moments are those of `m/scale`, and the
constant term of the Chebyshev integral uses $\arccos(-x)$ where the final
$1-\rho$ needs $\arccos(x)$, so even with $x$ scaled the result is off by
exactly $2\arccos(x)/\pi - 1$. The second audit already noted that `npol`
and `kernel` are computed or accepted and never used; the two defects above
are new.

**Oracle.** The exact occupation $\sum_{E<f}|\psi_i|^2$ from `eigh`, and the
closed-form integral $N(x) = (\mu_0\arccos(-x) - 2\sum_n g_n\mu_n\sin(n\arccos
x)/n)/\pi$ with Jackson damping written by hand, which matches the exact
occupation to 1e-3 at every $f$.

**Reproduction.** A 400-site chain:

```
fermi -1.0  exact 0.2715  get_density 1.0
fermi -0.5  exact 0.3692  get_density 0.574 (scale 3: 0.4579, scale 6: 0.3334)
fermi  0.0  exact 0.4532  get_density 0.4531
fermi  0.5  exact 0.5321  get_density 0.3075 (scale 3: 0.3786, scale 6: 0.6667)
fermi  1.0  exact 0.6122  get_density 0.0
```

**Fix.** Scale the Fermi energy, fix the sign of the constant term and apply
the kernel, or delete the module, since nothing in `src`, `tests` or
`examples` calls it; the maintainer's call.

**Status.** Open.

### 9. Embedded_Hamiltonian.get_density_matrix(delta=...) raises TypeError

**Cause.** `embedded.py:12-13` passes `delta=self.delta` together with
`**kwargs`, so a caller's `delta` arrives twice.

**Reproduction.**

```
TypeError: get_dm() got multiple values for keyword argument 'delta'
```

**Fix.** Take the caller's value when there is one, falling back to
`self.delta`.

**Status.** Open.

### 10. Vinteraction_kpm swallows keywords that its exact sibling refuses

**Cause.** `generic_densitydensity_kpm` (`densitydensity_kpm.py:43-48`) ends
in `**kwargs` and its body never reads them, and `Vinteraction_kpm` has
neither the refusal of the old `selfconsistency` keywords nor the refusal of a
spinless $U$ that `3b43557` gave `Vinteraction`: it builds $U$ only inside `if
h.has_spin:`. A spinless `h.get_mean_field_hamiltonian(integration="kpm")`
goes straight there.

**Oracle.** The exact sibling on the identical call, and the bit identity of
the result with and without the keyword.

**Reproduction.** A spinless two-site chain at $V_1=1$:

```
KPM filling=0.25              : E=-0.8745119120
KPM filing=0.25 (typo)        : E=-0.9772559373   (half filling)
KPM U=2 spinless              : E=-0.9772559373
KPM kernel=lorentz            : E=-0.9772559373
KPM solver=broyden_mixing     : E=-0.9772559373
KPM integration=qtci          : E=-0.9772559373
KPM g=1 mode=V (old keywords) : E=-1.2707566137   (no interaction at all)
ED  filing=0.25               : TypeError unexpected keyword argument(s) ['filing']
ED  U=2 spinless              : ValueError a local Hubbard U requires the spin degree of freedom
```

The public route returns `None` on a spinless call with `solver="newton"` or
`filing=0.25`, where `integration="ed"` raises. There is no kernel keyword
anywhere in the KPM mean field (the Jackson kernel is hardcoded).

**Fix.** Raise `TypeError` on leftover keywords at the top of
`generic_densitydensity_kpm`, and give `Vinteraction_kpm` the refusals of
`Vinteraction`, sharing that code rather than copying it.

**Status.** Open.

### 11. A KPM scale below the extent of the expanded spectrum is not checked

**Cause.** `_dm_kpm_from_needed` and `_kpm_dos_moments`
(`densitymatrix_kpm.py:256-264` and `391-397`) refuse only `scale<=0`. The
loop expands the Hamiltonian after the Fermi shift, so a scale reasoned from
the bare bandwidth is short by the shift: on a spinful chain with the
spectrum in $[-2,2]$ at `filling=0.1`, the shift of about 1.9 moves it to
about $[-0.1, 3.9]$.

**Reproduction.**

```
scale 1.0  Vinteraction_kpm  ValueError The function value at x=-0.990004 is NaN; solver cannot continue.
scale 1.0  VJ kpm            converged=False E=nan   (every iteration with a NaN residual)
scale 2.5  both as above
scale 3.0  both as above
scale 5.0  both run
```

The guide describes `scale` only as the energy rescaling, estimated when not
given; the post-shift requirement is written in a comment at
`spinspin.py:1088-1105`. `kpm.tdos` and the other KPM DOS routines take a
user scale unchecked as well.

**Fix.** Compare a given scale against the Gershgorin bound
`_estimate_kpm_scale` already computes, on the shifted Hamiltonian, and raise
a `ValueError` naming the minimum.

**Status.** Open.

### 12. The dense BSE path swallows unknown keywords

**Cause.** `BSE.__init__` (`bsetk/solve.py:94-97`) takes `**kwargs` and
forwards them only to `solve_iterative` and `solve_qtt`; the dense branch
never reads them, so `h.get_bse(nkk=40, kernal="none")` runs `nk=10` with
`kernel="full"`, and `metal=True`, which `PairBasis` supports, is ignored.
`solver="iterative"` raises `TypeError` on the same misspelling.

**Reproduction.** The misspelt call and the plain one both give 1.6414636882,
while `kernel="none"` spelt right gives 1.8 and `nk=6` gives 1.6865307364.

**Fix.** Raise `TypeError` in the dense branch when `kwargs` is not empty.

**Status.** Open.

### 13. full_dm_simultaneous crashes on a list-valued k-mesh

**Cause.** `densitymatrix.py:391-394` normalizes by `1./nk**2` or `1./nk**3`
from the raw argument, where every other route uses `1./len(ks)`.

**Reproduction.** `h.get_density_matrix(nk=[3,5], dm_mode="simultaneous")`
raises `TypeError: unsupported operand type(s) for ** or pow(): 'list' and
'int'`, and so does any list or tuple, even `[4,4]` or `(3,3,3)`, while the
accumulate route agrees with a real-space supercell oracle to 4e-16.
`dm_mode="simultaneous"` is not the default and nothing in the tree passes it
a list.

**Fix.** Normalize by the number of k-points actually used.

**Status.** Open.

### 14. occupied_projector ignores delta

**Cause.** `densitymatrix.py:459-463` calls `full_dm_python(es,np.array(vs))`
without its `delta`, so any smearing returns the $T=0$ projector. The second
audit listed it among its unchased unused parameters. The only caller,
`topologytk/realspace.py:15`, uses the default, so no shipped number changes.

**Reproduction.** On a two-level matrix with levels at $\pm 0.01$, the
occupations are [0, 1] at `delta=1` and `delta=100` where Fermi-Dirac gives
[0.4975, 0.5025] and [0.499975, 0.500025].

**Fix.** Forward it, with the $T=0$ guard `full_dm` uses, or drop the keyword
so that passing it raises.

**Status.** Open.

### 15. The embedding mean-field examples call a removed method

**Cause.** `c1be61a` removed `get_mean_field_hamiltonian` from both embedding
classes, saying that nothing in the examples called it, but
`examples/embedding/scf/main.py:33` and
`examples/embedding/scf_chain/main.py:38` both do, and stop with
`AttributeError`. These two scripts are the only place in the repository that
would consume an embedding density matrix, which is why #6 and #7 have no
victim today.

**Fix.** Remove the two examples, or rebuild an embedding mean field, which
would need #6 and #7 first; the maintainer's call.

**Status.** Open.

### 16. The sign of the physical field of a spinon add_zeeman

**Cause.** `add_zeeman(b)` writes $+\vec b\cdot\vec\sigma = +2\vec b\cdot\vec
S$, so in the convention $H=-\vec h\cdot\vec S$ the physical field is
$\vec h = -2\vec b$ and the induced moment points against $\vec b$. The
docstring of `SpinonHamiltonian` (`spinon.py:80-83`) and
`documentation/user_guide.md:2065` both say $\vec h = 2\vec b$. The code is
consistent; the two texts have the sign wrong.

**Reproduction.** The moment of a converged spinon chain under a field along
[0.3,0.5,0.2] is exactly antiparallel to it, and a single spinful site with
`add_zeeman([0.2,-0.5,0.4])` has $\langle\vec S\rangle\cdot\hat b = -1$.

**Status.** Open.

## Chased and cleared

- **The KPM mean field against the exact engine.** On an s-wave attractive
  Hubbard Rashba chain at fixed $\mu$ and at fixed filling, a spinless charge
  density wave, a metal at fixed filling, and an anisotropic exchange
  ($J_1$, $J_{1x}$, $J_{1z}$) with $U$ on a two-site Rashba chain with a
  staggered field along a generic direction, energies, gaps, magnetizations
  and fillings agree with the exact engine and close with `npol`. The
  absolute energy of `get_total_energy_kpm` passes Hellmann-Feynman
  ($dE/dU$ at fixed $\mu$ against $\sum_i \langle n_\uparrow\rangle\langle
  n_\downarrow\rangle - |\langle c^\dagger_\uparrow c_\downarrow\rangle|^2$,
  0.147041 against 0.147039), and the KPM density matrix of a frozen
  honeycomb with Haldane, Rashba and a generic exchange converges onto the
  exact one with `npol`, while its complex conjugate stays 0.14 away. An
  offset of 3.9e-3 on a 33-site flake that did not close with `npol` turned
  out to be a second, lower fixed point near a soft mode, which the exact
  engine also reaches when run tighter.
- **The normal expectation values.** `spectrum.ev`, `get_vev`,
  `get_single_vev`, `get_several_vev`, the matrix-free route and
  `get_magnetization` on a non-Nambu Kane-Mele, Rashba and generic-exchange
  honeycomb agree with the occupied-eigenvector sum to 1e-15 for `sy` and a
  random complex Hermitian operator; `get_dm_vev` and `real_space_vev` on a
  0d island, and `real_space_vev`, `get_vev` and `get_filling` on Nambu, are
  right. The transpose class of `bug_audit.md` 1.2 has no sibling left among
  the normal consumers.
- **The entanglement entropy**, normal and BdG at finite pairing on a
  four-site Rashba chain with a generic exchange, against the reduced density
  matrix of the exact many-body ground state in the $2^8$ Fock space, to
  1e-15.
- **The local Chern marker.** On a 384-site Haldane flake it gives 1.0128 in
  the bulk against a k-space Chern number of 1, and exactly -1.0128 for the
  complex-conjugated Hamiltonian; the excess of 1.3 percent is the area per
  site estimated from the sites inside `rcut`, as `bug_audit_2.md` #65 says.
- **The spinon mean field** under a field along a generic direction: the
  constraint holds and the induced moment and the mean field are collinear
  with the field to 1e-8, so its use of the density matrix carries no flipped
  `sy`.
- **Every k-integrated density-matrix route** at fixed filling, on an
  insulator and on a metal, at $T=10^{-15}$ and $0.05$, against a real-space
  supercell oracle that shares no Bloch generator, mesh or eigensolver with
  pyqula: accumulate at every `batch_size`, simultaneous, the sparse routes at
  `dense_fraction` 0 and 1 on every pair and on a random subset, the
  with-Fermi route and its `max_memory_gb` fallback, sparse Hamiltonians on
  each, Nambu, 0d and 3D, and the jax route on jax's CPU backend. The error
  is at most 7e-15 (7e-14 on the local-Fermi variant) and the Fermi level
  agrees to brentq's tolerance. The
  local-Fermi variant holds the same electron count as the global search,
  and forcing every direction onto the dense kernel reproduces the shipped
  mean field bit for bit, so the sparse pair masks are complete.
- **The dense BSE at every Q**, full and Tamm-Dancoff, bare and with the RPA
  screening, the spin-flip magnons on a collinear ferrimagnet and the
  non-collinear branch selection, all against a ring TDHF written from
  scratch (Casida's A and B blocks on the $N$-cell ring), to 7e-14 and
  better; the static polarizability and the screened interaction on
  Haldane and on Kane-Mele with Rashba and a generic exchange against a numpy
  construction, to 4e-16; the non-interacting limit; and the iterative
  solver against the dense one for every kernel.
- **Two scanner flags from the second audit**, `qtt.solve_qtt` swallowing
  `nkW` and `channel` (inert, since `solve_qtt` refuses any screening at its
  first statement and they are read only under screening) and
  `bsetk/screening.py:484` reassigning `exclude` (the documented behaviour
  for the RPA). `bsetk/oracle.py` is the lazily evaluated pair basis of the
  quantics solver, not an independent oracle.

## Left for the next sweep

- The anti-Hermitian growth of #5 with a Hubbard interaction alone on a 2D
  lattice; a bond pairing through `Vinteraction_kpm`; `hubbard_kpm` called
  directly and `constrains=` through the KPM engines; `cores=`, which sets a
  package-wide core count that outlives the call; and the finite-temperature
  total energy, where both engines add a $T=0$ band energy to a
  finite-temperature density matrix, so neither is a free energy.
- `spectrum.ev2d` and `selected_bands2d` on a Nambu Hamiltonian, which have
  the shape of #2 and no callers; current, velocity and valley operators
  through `get_vev` against an independent construction; `real_space_chern`
  with `operator=` or on a spinful or Nambu flake; the embedding with
  `nsuper>1`, in 2D, with Nambu, or with a nonzero self-energy; the
  finite-temperature agreement of `kresolved_orbital_vev` with `full_dm`
  (only the default smearing was compared); and the entanglement spectrum of
  the 2D Li-Haldane cut and the ring entropies of periodic 1D and 2D
  Hamiltonians against an independent oracle (only 0d cuts were checked
  against exact diagonalization).
- A numpy-array `nk` crashes every density-matrix route at
  `kpointstk/kmesh.py:11` (`if nk==1`), while a list works; the sparse routes
  called directly with `delta=0` and a level exactly at the Fermi energy;
  k-meshes with `nsuper` other than 1, and the jax route in single precision,
  which `full_dm_accumulate` never requests; and whether
  `full_dm(dm_mode="simultaneous", batch_size=...)` refuses the forwarded
  keyword with a clear message, since `full_dm_simultaneous` has no such
  parameter.
- cRPA against an independent oracle, `nkW>nk` and the truncation of
  `ScreenedInteraction.get_dict`, the GPU screening path
  (`bsetk/screeningjax.py`), the exciton amplitudes, `exciton_bands` with
  `cores>1` and on 3D models, `metal=True` magnons and the transverse rung of
  `Vchannels` against the ring oracle, the quantics solver on a spinful 2D
  model, the iterative solver near a
  degenerate cluster at large `nk` (its LOBPCG warnings are suppressed), and
  whether an explicit `gauge="projection"` with screening moves the static
  polarizability measurably, which is the root cause of #1 and was reasoned,
  not run.
- Nothing here ran on a GPU.
