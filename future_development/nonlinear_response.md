# Nonlinear response: what is built, what is missing, and the plan to build it

This file is the roadmap of the nonlinear electrical response of pyqula, written on
7 November 2026 as the answer to "what is the status, is there something to implement".
It records the conventions, the formulas with their derivation, the checks each piece has
to pass, and the split of the work into packages. An implementer should be able to work
from this file without reading the papers again; the papers are cited so that a prefactor
can be traced when a check fails. The plan was reviewed the same evening by an independent
critic who ran the formulas on the fixtures; the fifteen points of that review were all
accepted and are folded in below, with the measurements that decided them. The maintainer
gave the go-ahead for the build and for the GPU work in the evening of 7 November 2026, and
the whole plan was built that night; the next section says what landed and what is open. The
sections after it are the plan as it was written, kept as the record of the derivations and
the checks.

## Status on 8 November 2026: what landed

All nine packages of the plan landed, none was cut.

- `kernel` (`3b3f280`, `conductivitytk/nonlinearkernel.py`, 22 tests): the general recursion,
  1D to 3D, any order, any tuple of complex frequencies. First order agrees with
  `kubo.optical_conductivity` to the aliasing of the mesh (6e-13 at `nk=48` on the gapped
  honeycomb lattice), second order at $(1,1)$ with `shg.py` to rounding, and the velocity gauge
  with an independent length gauge on a crystal of isolated molecules to 1e-13 at orders 1 to 4.
- `dc2` (`3b3f280`, `conductivitytk/nonlineardc.py`, 16 tests): the second-order dc terms with
  the three conventions, the channels and the block traces, checked as listed in "The checks".
  Ezawa's Neel-angle law holds at $J = 0.01$; at $J = 0.05$ a $\cos3\Phi$ harmonic, third order in
  $J$, is 11 % of the amplitude of the quantum-metric term, so the test uses the smaller $J$.
- `shg-twofreq-3d` (`3b3f280`, `shg.py`): the length gauge at two frequencies
  (`sigma2_length`, `sigma2_length_diagonal`) and the 3D grid SHG with the decoupled-layers test;
  `"triangle"` and `"qtci"` still raise in 3D.
- `photocurrent` (`212dc58`, `conductivitytk/photocurrent.py`, 15 tests): injection, shift and
  rectified total, and the sum frequencies. The injection rate uses a multiplet-block form,
  $\Delta^a_{mn}r^c_{nm} \to [r^c, Ph^aP]_{nm}$: the plain diagonal formula was 4.3 % off in
  $\eta$ and 24 % off in the shift current for the Haldane model folded into a $2\times2$
  supercell, and the block form agrees with the primitive cell to 1e-14.
- `dc-crosscheck` (`9c8f0f8`, `conductivitytk/nonlineardcfit.py`, 6 tests): the contour
  extraction of the Laurent coefficients, with a doubled tilted Dirac model whose bands are all
  degenerate as the fixture that reaches the degeneracy filter.
- `gpu` (`57f1c19`, 9 tests): the recursion under jax jit and vmap, routed on `gpu.get_gpu()`.
- `dc3` (`8248cf1`, 6 tests): `order=3` of `nonlinear_dc_conductivity`, the $\tau^3$ Drude term
  for any multiband charge channel (1D and 2D, since it reuses the two-index recursion of
  `nonlineardrude`) and the $\tau^1$ Berry connection polarizability term of Liu et al.,
  derived again from their Eq. (j3). Their field-corrected band energy has the opposite sign to
  the Stark shift of the second order, because the semiclassical band energy leaves out the
  potential energy of the shifted wave-packet center; the module docstring records it.
- `cpge` (`4056db9`, 2 tests): the trace of the circular injection on a two-node cubic Weyl
  model, averaged over a frequency window with a Hann weight at $\delta = 0.004$, gives
  $16\pi S/(q^3C) = 0.9768$, $0.9932$ and $0.9924$ at `nk` = 30, 60 and 120, about 0.7 % low from
  the Lorentzian tails, and $0.0015$ when both nodes absorb.
- `wannierberri` (`e1fed36`): the Berry curvature dipole of the tilted Dirac model agrees with
  WannierBerri to 0.02 % ($D_x = -1.44279\times10^{-2}$ against $-1.44303\times10^{-2}$), and the
  shift current of the gapped honeycomb lattice to 0.6 to 2.5 % at $\delta = 0.04$, a difference
  that halves with $\delta$, so it is broadening and not a prefactor. The sign is opposite since
  WannierBerri uses $e > 0$. The test skips where WannierBerri is not installed, which is the case
  in the standard environment.
- `integration`: the six public functions of `conductivity.py` with the delegators
  `h.get_nonlinear_conductivity()`, `h.get_nonlinear_susceptibility()`,
  `h.get_nonlinear_dc_conductivity()`, `h.get_berry_curvature_dipole()`, `h.get_shift_current()`
  and `h.get_injection_current()`, the guide page `documentation/guide/nonlinear-response.md`,
  the reference entries, the README bullet and the examples `examples/2d/nonlinear_hall`,
  `examples/2d/shift_current` and `examples/2d/third_harmonic`. The rectified total with its
  split and the sum frequency stay in `photocurrent.py` (`rectified_conductivity`,
  `sum_frequency_conductivity`) without a delegator; the second is
  `h.get_nonlinear_conductivity(frequencies=(c1,c2))`.

What $A_0$ is, measured by the contour at `nk=150` and identical at `nk=300` ($T = 0.05$, radius
$0.12$, 24 points), against the three conventions:

| model | component | $A_{-2}$ | $\sigma^{ND}/2$ | $A_{-1}$ | $\sigma^{BCD}$ | $A_0$ | shibata | gao | kaplan |
|---|---|---|---|---|---|---|---|---|---|
| tilted Dirac | xxx | -9.819936e-3 | -9.819936e-3 | 0 | 0 | 2.504e-2 | 2.683e-3 | 0 | -1.669e-2 |
| tilted Dirac | yxx | 0 | 0 | 1.442798e-2 | 1.442798e-2 | 0 | 0 | 0 | 0 |
| tilted Dirac | xyy | -2.727775e-3 | -2.727775e-3 | 0 | 0 | -2.377e-3 | -1.072e-2 | -1.027e-2 | -2.239e-2 |
| tilted Dirac | xxy | 0 | 0 | -7.213991e-3 | -7.213991e-3 | 0 | 0 | 0 | 0 |
| real two-band | xxx | -4.542947e-4 | -4.542947e-4 | 0 | 0 | 1.345e-4 | -4.397e-4 | 0 | -8.967e-5 |
| real two-band | xyy | 2.644602e-3 | 2.644602e-3 | 0 | 0 | 1.366e-3 | 2.106e-2 | 9.440e-3 | 2.112e-2 |
| real two-band | yxy | 2.644602e-3 | 2.644602e-3 | 0 | 0 | -1.279e-2 | -2.411e-3 | -4.720e-3 | -2.484e-3 |

Here 0 means below $10^{-16}$, and the imaginary parts are below $10^{-16}$ too. At third order
on the real two-band model, `xxxx`, $A_{-3} = -2.594542\times10^{-3} = \sigma^{ND,3}/(6\tau^3)$ and
$A_{-1} = 5.97\times10^{-4}$, at both meshes. So the pole coefficients are the relaxation-time
terms to rounding, and $A_0$ matches no convention, on either model, in any component: the
adiabatic switching is a fourth phenomenology at $\tau^0$, as the plan expected.

GPU, measured on the RTX A2000 (a float32 card) with the honeycomb supercells of 2 to 100
bands, `nk=40`, $\omega$ from 0.2 to 2 at $\delta = 0.1$, one component, against numpy on the
performance cores. The machine was shared with the other agents of the night, so the timings
are indicative only. The 64- and 100-band runs at 100 frequencies, and the 100-band run at third
order with one frequency, overlapped loads of 5 to 24, so their CPU times are inflated; the
last row of the table is one of them and is kept only for the device times.

| bands | order | frequencies | numpy | device single | device double | error single |
|---|---|---|---|---|---|---|
| 2 | 2 | 100 | 0.045 s | 0.013 s | 0.091 s | 2.0e-6 |
| 8 | 2 | 100 | 0.55 s | 0.022 s | 0.29 s | 2.0e-6 |
| 8 | 3 | 1 | 0.050 s | 0.016 s | 0.048 s | 6.7e-6 |
| 24 | 2 | 1 | 0.20 s | 0.053 s | 0.35 s | 5.8e-6 |
| 24 | 2 | 100 | 4.7 s | 0.092 s | 1.06 s | 7.2e-6 |
| 24 | 3 | 100 | 9.6 s | 0.15 s | 1.57 s | 1.2e-5 |
| 64 | 2 | 1 | 1.8 s | 0.40 s | 1.22 s | 3.2e-6 |
| 100 | 2 | 1 | 4.9 s | 0.93 s | 3.96 s | 1.4e-6 |
| 100 | 3 | 100 | (1649 s, loaded) | 5.8 s | 75 s | 3.5e-6 |

The device in double precision agrees with numpy to 1e-15 to 1.5e-14 relative at every size.
Single precision is 4.5e-7 to 1.4e-5 of the largest component at these frequencies, which is
the expected $\epsilon_{32}(t/|\omega + i\delta|)^n$ scale. The conclusions: with 100
frequencies the device in single precision is 3.5x faster at 2 bands, 25x at 8 and 50 to 65x
at 24 bands and above, and in double precision 4 to 6x from 24 bands on (on this card double is
1/70 of single, so the double numbers say nothing about a data-centre card). With one frequency
the crossover is between 2 and 8 bands in single precision and between 24 and 64 bands in
double. The kernel compiles once per call shape, 0.1 to 1 s, which a single call pays on top of
the times above.

Open, in the order of their value:

- The pinned timings on an idle machine of "What to optimize if time is left" are owed: the
  general kernel against the numba SHG kernel at 24 bands, the device crossover, the dc routine
  at `nk=200`, the fourth-order memory.
- The third-order $\tau^2$ (Berry curvature quadrupole) and $\tau^0$ dc terms, and the 3D
  third-order Drude term, are not built.
- The multiplet restriction of the velocity in the injection rate is not pinned by a test:
  every fixture that pins $\eta$ has two bands. Replacing the masked velocity by the full one
  moves $\eta$ by more than 100 % on a four-band bilayer and every test still passes, while the
  code matches the length-gauge diagonal on that bilayer to 2.5e-16. A four-band fixture is the
  missing test.
- GPU: no test checks that the single-precision recursion runs in complex64, and no test has a
  padded frequency chunk with two or more real rows; the device memory budget counts only the
  density matrices of the chunk, so the peak footprint is a few GB.
- The cell volume of the third-order Drude term is not pinned by a test, since every magnitude
  fixture has unit cell area (a brute force on the strained honeycomb lattice agrees to 7e-14).
- The user-facing notebook of the README bullet and a figure for the guide page are not made;
  `jupyter-notebooks/` and `documentation/figures/` were outside the integration package.
- The spin current at finite frequency, spin photocurrents, and a triangle or qtci integration of
  the general kernel stay parked, as below.

## What existed before the build, 7 November 2026

Three pieces exist.

- `conductivitytk/shg.py`: the second-harmonic conductivity $\sigma_{abc}(2\omega;\omega,\omega)$
  by the velocity-gauge density-matrix recursion of Passos et al. (arXiv:1712.04924), with
  the first, second and third Cartesian k-derivatives of the Peierls-coupled Bloch
  Hamiltonian as vertices, a numba kernel over frequencies, a length-gauge check for cold
  insulators, and three k-integrations (grid, triangle, qtci). The kernel takes one frequency
  and puts its poles at $2\omega$. It cannot give the rectified current, the dc limit, or a
  sum frequency. See `second_harmonic_generation.md`.
- `conductivitytk/nonlineardrude.py`: the $l$-th order nonlinear Drude conductivity
  (the $\tau^l$ term) in the spin and charge channel, with the band derivatives to any order
  from a Rayleigh-Schroedinger recursion in the Kato block form. It needs spin to be a good
  quantum number and it is two-dimensional: `_Channel.derivative` returns zero for any $z$
  axis and the Taylor recursion has two indices. See `nonlinear_spin_transport.md`.
- `conductivitytk/kubo.py` and `topologytk/anomaloushall.py`: the linear conductivity, the
  Cartesian velocities in the atomic gauge, and the Berry curvature of a band on a mesh.

Nothing exists for the second-order dc conductivity of a multiband model with spin-orbit
coupling, for the Berry curvature dipole, for the quantum-metric (intrinsic) term, for the
shift and injection currents, for sum and difference frequency generation, or for any
response above second order beyond the Drude term. So the answer to "is there something to
implement" is yes, and this file says what and in which order.

Two statements in the original task text were stale and are corrected here. The code has a
3D branch: `kubo._cell_factors` and `klist.kmesh` handle three periodic directions and
`tests/conductivity/test_optical_3d.py` exercises them, so 3D is a guard to lift in `shg.py`,
not a missing formalism. And the quantum-metric term does not need broken $PT$: the
$PT$-symmetric antiferromagnet is its canonical host (Wang et al., arXiv:2106.12695), where
$PT$ forces the Berry curvature to vanish at every k and so kills the Berry curvature dipole
exactly.

## Conventions

Everything here follows `shg.py` and `kubo.py`.

- $\hbar = 1$, energies in units of the hopping $t$, lengths in the units of the geometry.
  The carrier charge is $q = -e = -1$. An $n$-th order response goes as $q^{n+1}$, so the
  even orders flip sign with the charge and the odd ones do not. In a formula of the papers
  written with $e > 0$, $e^3$ becomes $1$ and $-e^3$ becomes $-1$.
- The Bloch Hamiltonian is pyqula's lattice-gauge $H(k) = \sum_R t(R) e^{2\pi i k\cdot R}$.
  The $m$-th Cartesian derivative of the Peierls-coupled Hamiltonian is
  $h^{a_1\dots a_m} = \sum_R (i d_{a_1})\cdots(i d_{a_m}) \circ t(R)\, e^{2\pi i k\cdot R}$,
  with $d_{ij}(R) = R + r_j - r_i$ the full bond vector and $\circ$ the elementwise product.
  This is exact for a tight-binding model whose position operator is diagonal in the
  orbitals, and the eigenbasis matrix elements are the same in the lattice and in the atomic
  gauge (`shg._derivative_tensors`, `shg._cartesian_derivative`). $h^a$ is the velocity of
  `kubo.py` and $h^{ab}$ its inverse-mass operator (`tests/conductivity/test_shg.py`).
- The k-integral is the density normalization $\int d^dk/(2\pi)^d = (1/(N_k V_{cell}))\sum_k$,
  so a conductivity is per unit cell volume (area in 2D, length in 1D). The $n$-th order
  conductivity is then in units of $e^{n+1} a^{n+1-d}/(\hbar\, t^{\,n-1})$.
- The field convention is that of Passos et al., Eq. (defresponsefunction):
  $J_\beta(\omega) = \int \prod_j \frac{d\omega_j}{2\pi}\,
  \sigma^{(n)}_{\beta\alpha_1\dots\alpha_n}(\omega_1,\dots,\omega_n)
  E_{\alpha_1}(\omega_1)\cdots E_{\alpha_n}(\omega_n)\, 2\pi\delta(\omega - \sum_j\omega_j)$,
  with time dependence $e^{-i\omega t}$ and $\sigma$ symmetrized over the pairs
  $(\alpha_j,\omega_j)$. For a real monochromatic field $E(t) = \mathrm{Re}\,E_0 e^{-i\omega t}$
  this means: the second-harmonic current is $\frac12\mathrm{Re}[\sigma(\omega,\omega)E_0^2
  e^{-2i\omega t}]$, the rectified current is $\frac12\sigma(\omega,-\omega)|E_0|^2$, and a
  static field gives $J = \sigma(0,\dots,0)E^n$. The first index is always the current.
- Every frequency carries the same broadening, $\omega_j \to \omega_j + i\delta$. This is the
  adiabatic switching of Passos et al., their "second phenomenology": a pole at a sum of $m$
  frequencies carries $i m\delta$. It is not a relaxation time. The mapping to the
  relaxation-time expansion is derived below, it is order dependent, and it does not reach
  the $\delta^0$ term.
- The chemical potential is zero (`h.shift_fermi`), the occupations are `kubo._fermi(es,T)`
  and the Fermi-surface weight is $w_n = -f'(\varepsilon_n) = f(1-f)/T$, which is zero on a
  mesh at $T = 0$ (`kubo._minus_dfermi`). Every Fermi-surface formula therefore needs $T > 0$
  and a mesh whose energy spacing $v_F\,2\pi/(nk\,a)$ is below $T$.
- A tensor component is a string of Cartesian labels with the current first, `"xyy"` at
  second order and `"xxyy"` at third. `nonlineardrude` names its components `"xy;x"` with
  the fields first; its `_sigma` returns the symmetric tensor component
  $\sigma_{b\,a_1\dots a_l}$, so the expansion $j_b = \sum\sigma^{x^{l_1}y^{l_2};b}
  E_x^{l_1}E_y^{l_2}$ in its module docstring undercounts the mixed terms by
  $\binom{l}{l_1}$. The docstring is to be corrected; the comparisons below are right as
  written and must not be "fixed" with a factor 2.

## The broadened recursion to any order

This is the general kernel, the piece that gives harmonic generation, sum and difference
frequencies, photocurrents and the dc limit at any order from one code path.

### The formulas

Passos et al., Eqs. (rho) and (responsefunction), read from the TeX, with $\hbar = 1$.
With $\rho^{(0)} = \mathrm{diag}(f)$ and $\Delta\varepsilon_{nm} = \varepsilon_n -
\varepsilon_m$, the density matrix at order $p$ for the ordered field list
$(\alpha_1,\omega_1),\dots,(\alpha_p,\omega_p)$, every $\omega_j$ already complex, is

```
rho^{a1..ap}(w1..wp)_nm = 1/(w1+...+wp - dE_nm)
    * sum_{m=1}^{p} (1/m!) [ h^{a1..am} , rho^{a(m+1)..ap}(w(m+1)..wp) ]_nm
```

and the unsymmetrized conductivity is

```
sigma_{b a1..an}(w1..wn) = (i^n q^(n+1)/(w1 ... wn)) (1/(Nk Vcell)) sum_k
    sum_{p=0}^{n} (1/p!) Tr[ h^{b a1..ap} rho^{a(p+1)..an}(w(p+1)..wn) ]
```

followed by the average over the $n!$ permutations of the pairs $(\alpha_j,\omega_j)$. For
$n = 2$ and $\omega_1 = \omega_2$ this is `shg._sigma2_velocity_jit` term by term, and for
$n = 1$ it is Passos's Eq. (sigmaA), the velocity-gauge Kubo formula with the diamagnetic
term, which equals `kubo.optical_conductivity` at the same complex frequency up to the
aliasing of the mesh (the sum rule of their Appendix A is the mesh sum of a k-derivative of
a periodic function). The $1/(\omega_1\cdots\omega_n)$ prefactor is finite at
$\omega_j = i\delta$ because of the broadening, and the apparent divergence cancels between
the $p$ terms in the way the sum rules say, so the dc limit is just $\omega_j = 0$ with
$\delta > 0$. The recursion needs the vertices $h^{(m)}$ for $m = 1,\dots,n+1$.

### How the recursion is organized

The internal entry point takes an `(nw, order)` array of complex frequencies and nothing
else about the broadening: the public function builds $\omega_j = c_j\omega + i\delta$ from
`energies`, the multipliers and `delta`, and the Laurent cross-check below passes
$\omega_j = i\delta_j$ with a complex $\delta_j$ on a circle, half of whose points have
$\mathrm{Im}\,\omega_j < 0$. The kernel therefore never clamps or validates the sign of the
imaginary part. The object at level $p$ depends on the suffix $(\alpha_{n-p+1},\dots,
\alpha_n)$ of the field list and on the sum of the suffix frequencies. So we compute, for a
batch of k-points,

- the vertices $h^{(m)}$ for $m = 1..n+1$, one `(nk,n,n)` array per sorted index
  combination, $\binom{m+2}{2}$ of them in 3D, with a lookup from an ordered tuple to its
  sorted combination. `shg._eigenbasis_derivatives` is this for $m \le 3$;
- the needed suffixes: the set of suffixes of every requested component and of its
  permutations. A few components at third order need a few dozen suffixes, not $3^p$;
- level by level, `rho[p][suffix]` as batched `(nk,n,n)` matmuls, with the elementwise
  division by $\Omega_p - \Delta\varepsilon$ where $\Omega_p$ is the suffix frequency sum;
- the trace $\mathrm{Tr}[A\rho] = \sum_{nm} A_{nm}\rho_{mn}$ for each component and $p$.

The recursion is written as array code over the k-batch with a backend switch, numpy on the
CPU and `jax.numpy` on the device, routed on `gpu.get_gpu()`. The frequency loop is outside.
This is the batched dense shape the GPU plan favours, and it makes the GPU path a small
package. The numba kernel of `shg.py` stays as the fast CPU path for SHG and as the identity
check of the general kernel at $n = 2$. There is no "diagonal split" of the output: the
first draft returned the within-multiplet part of the $p = 0$ term separately as the
injection current, and the review showed that this part is not gauge invariant (see the
photocurrent section), so the injection current is computed from its own closed form.

Cost per k-point and frequency tuple: one commutator per $(p, m)$ pair per needed suffix,
each two matmuls of $n_b^3$. With all $3^n$ components at order $n$ this is
$\sum_p 3^p p$ commutators (102 at third order, 426 at fourth). Memory: the vertices are
$\sum_{m\le n+1}\binom{m+2}{2}$ arrays per k (55 at fourth order), so the k-batch shrinks
with the order and the number of bands under a fixed budget of about 1 GB, as
`nonlineardrude._CHUNK_BYTES` does.

For $n$ fields at equal frequency the permutations only permute Cartesian labels, so the
kernel runs once and the labels are averaged. For a general frequency tuple the pairs
$(\alpha_j, c_j)$ are permuted together and the kernel runs once per distinct permutation,
at most $n!$ times, sharing the vertices and the `rho` cache keyed by (suffix labels,
suffix multipliers). A multiplier can be zero: $(1,1,0)$ is the electric-field-induced
second harmonic and $(1,-1,0)$ the jerk current, both with the static field carrying
$i\delta$ alone.

The susceptibility follows from $J = \dot P$: $\chi^{(n)}(\omega_1..\omega_n) =
\sigma^{(n)}/(-i\,\Omega)$ with $\Omega = \sum_j(\omega_j + i\delta)$, in the units of
`shg.to_susceptibility`, which is the $n = 2$ case and is generalized to any $n$.

### Degeneracies and metals

The recursion never divides by a band energy difference: every denominator carries
$i m\delta$. So degenerate bands, band crossings and Fermi surfaces are safe by
construction, and a metal is handled with the same caveat as the Drude part of
`kubo.optical_conductivity`: $T > 0$ and a mesh that resolves both $T$ and $\delta$. This is
why the general kernel is the fallback wherever the clean-limit formulas below fail, at a band
touching on the Fermi surface in particular.

### Single precision

The prefactor $1/\prod_j(\omega_j + i\delta)$ cancels Fermi-sea terms of order one, so the
float32 error of the second-order kernel measured against float64 on the gapped honeycomb
lattice is $10^{-5}$ at $\omega = 0$ with $\delta = 0.1$, $1.2\times10^{-3}$ at $\omega = 0$
with $\delta = 0.02$, and below $10^{-6}$ for $\omega \ge 0.2$; the third order adds one more
power. A device test in single precision therefore scales its tolerance as
$\epsilon_{32}\,(t/|\omega + i\delta|)^n$, and the kernel warns on `prec="single"` when some
$|c_j\omega| < \delta$. The precision argument itself goes through `gpu.resolve_prec`, which
gives the accepted-values error and refuses single precision on the CPU; the frequency
check is the kernel's own warning, not something to push into `resolve_prec`.

### The dc limit and its Laurent structure in $\delta$

At $\omega_j = 0$ the recursion gives $S(\delta) = A_{-n}/\delta^n + \dots + A_0 + O(\delta)$.
The leading coefficients can be related to the relaxation-time expansion. In the
semiclassical (length-gauge) picture with the same adiabatic phenomenology the
distribution obeys $-i\Omega_p f_p = -(q/\hbar)E\cdot\partial_k f_{p-1}$ with
$\Omega_p = p\, i\delta$ at dc, so
$f_p = -(q/\hbar)E\cdot\partial_k f_{p-1}/(p\delta)$,
while the Boltzmann equation with a relaxation time gives
$f_p = -\tau (q/\hbar)E\cdot\partial_k f_{p-1}$. The intraband poles therefore map as

```
tau^n (n fields, Drude)        ->  1/(n! delta^n)      (n = 2: 1/(2 delta^2))
tau   (one field pole, BCD)    ->  1/delta
```

The anomalous velocity term that gives the Berry curvature dipole carries one pole, in
$f_1$, so its coefficient maps with $\tau \to 1/\delta$ exactly. The nonlinear Drude term
carries all $n$ poles. Both mappings were confirmed to rounding on Shibata's tilted Dirac
model ($\mu = 0.6$, $T = 0.05$): $A_{-2} = -9.819936\times10^{-3} = \sigma^{ND}/(2\tau^2)$ for
`xxx` and $A_{-1} = 1.442798\times10^{-2} = \sigma^{BCD}/\tau$ for `yxx`, identical at
`nk=150` and `nk=300`, since the intraband denominators at dc are exactly $p\,i\delta$ and
$\delta$ does not have to be resolved by the mesh. At third order a one-pole term carries
$1/(p\delta)$ with $p$ the position of the pole, so $A_{-1}$ and $A_{-2}$ mix several $\tau$
structures and only $A_{-3} = \sigma^{ND,3}/(6\tau^3)$ is asserted.

How the coefficients are extracted. A polynomial fit of $\delta^n S(\delta)$ over a real
ladder of $\delta$ fails: $S$ has poles at $\delta = -i\,\Delta\varepsilon_{nm}/p$, and with
the minimum gap $0.8$ of the tilted Dirac model the radius of convergence at second order is
$0.4$, the top of the ladder $0.07..0.4$ of the first draft. The degree-4 fit gave
$A_{-2}$ off by $1.6\times10^{-3}$ relative, $A_{-1} = -5.2\times10^{-4}$ where the exact
value is zero, and $A_0$ off by $25\,\%$. A Cauchy contour in complex $\delta$ works:
$N = 24$ points on $|\delta| = r$ with $r < \min|\Delta\varepsilon_{nm}|/n$ over the pairs in
different multiplets (a degenerate pair has its denominator exactly $p\,i\delta$ and feeds the
pole at the origin, not one on the circle; $r = 0.12$ on the tilted Dirac model), $A_k = \mathrm{mean}_j[S(\delta_j)\delta_j^{-k}]$, which gives the
numbers above to rounding and imaginary parts of $10^{-16}$. Zeros forced by symmetry are
asserted against $\max_k|A_k|$, not in relative terms. The fixtures must have a gap over the
whole mesh: the tilted Dirac model (gap $0.8$) and the real two-band model (gap $1.0$); the
altermagnet with Rashba coupling has a crossing at $\Gamma$ that sends the radius to zero
and is not a Laurent fixture.

What $A_0$ is. On the tilted Dirac model, `xxx`: $A_0 = 2.50\times10^{-2}$, against
`gao` $= 0$, `kaplan` $= -1.67\times10^{-2}$ and `shibata` $= 2.68\times10^{-3}$; `xyy`:
$A_0 = -2.38\times10^{-3}$ against $-1.03\times10^{-2}$, $-2.24\times10^{-2}$ and
$-1.07\times10^{-2}$. On the real two-band model, `xyy`: $A_0 = 1.37\times10^{-3}$ against
$9.44\times10^{-3}$, $2.11\times10^{-2}$ and $2.11\times10^{-2}$. So the $\delta^0$
coefficient of the adiabatic-switching recursion matches none of the three conventions, as
Shibata (Sec. "Comparison with recent dc-limit formulas"), Michishita and Peters and
Nakazawa et al. say it need not: the intrinsic term depends on the scattering model and on
the order of the dc and clean limits. Two consequences for the code. The docstring and the
guide say that `nonlinear_conductivity(energies=[0.])` is the adiabatic-switching dc value,
not the $\tau$-expanded dc conductivity, and that it cannot be read at $\delta = 1/\tau$,
since $\tau^2$ maps to $1/(2\delta^2)$ and $\tau$ to $1/\delta$. And the cross-check package
asserts only $A_{-2}$ and $A_{-1}$ (and $A_{-3}$ at third order) and records $A_0$.

## The second-order dc conductivity in powers of $\tau$

This is the most valuable piece: the nonlinear Hall and nonreciprocal transport of a metal,
decomposed into the nonlinear Drude ($\tau^2$), Berry curvature dipole ($\tau^1$) and
quantum-metric ($\tau^0$) terms, each measurable by its scaling with the linear
conductivity (Kaplan, Holder and Yan, arXiv:2211.17213, Eq. (scaling)).

### The building blocks, from $h^a$ and $h^{ab}$ only

Everything is a Fermi-surface sum over the output of `shg._eigenbasis_derivatives` (bands
`es`, `h1`, `h2`), with no k-derivative of any geometric quantity. With
$\varepsilon_{nm} = \varepsilon_n - \varepsilon_m$ and $P(n)$ the degenerate multiplet of
band $n$ (consecutive bands closer than `degeneracy_tol` times the hopping scale, the
grouping of `anomaloushall.py`), for $m \notin P(n)$:

```
A^a_nm      = h^a_nm/(i eps_nm)                       interband Berry connection (= r^a_nm of shg.py)
Q^nm_ab     = A^a_nm A^b_mn = h^a_nm h^b_mn/eps_nm^2  pair quantum geometric tensor
G^n_ab      = sum_{m not in P} Re Q^nm_ab              quantum metric of band n
Omega^n_ab  = -2 sum_{m not in P} Im Q^nm_ab           Berry curvature (anomaloushall.py's)
Gt^n_ab     = 2 sum_{m not in P} Re Q^nm_ab/eps_nm     band-normalized metric
                                                        (Berry connection polarizability)
u^n_a       = h^a_nn                                   band velocity
u^n_ab      = h^{ab}_nn + 2 sum_{m not in P} Re(h^a_nm h^b_mn)/eps_nm
                                                        band inverse mass (Shibata Eq. (vnn-ij))
w_n         = -f'(eps_n) = f(1-f)/T                    Fermi-surface weight
```

The second-derivative identity is ordinary second-order perturbation theory for the
eigenvalue; `tests/conductivity` should pin it against `nonlineardrude._Channel.derivative`
on a multiband model, since the two routes share nothing. $G$ is pinned against
`qgt.quantum_geometric_tensor_k(occ_idxs=[n], coordinates="cartesian")`, whose default
`coordinates="reduced"` would not compare. $\tilde G$ has its own check on an insulator:
the second-order Stark shift of a band is $\Delta\varepsilon_n = \sum_m|\langle m|{-qE\cdot
r}|n\rangle|^2/\varepsilon_{nm} = \frac{q^2}{2}E_a\tilde G^n_{ab}E_b$, so the static
polarizability is $\chi_{ab} = -q^2\langle f\,\tilde G_{ab}\rangle$ (the occupied-occupied
pairs cancel in the sum), and it must equal $\lim_{\omega\to0} i\sigma_{ab}(\omega)/\omega$
from `kubo.optical_conductivity`.

### The formulas, $J_a = \sigma_{abc}E_bE_c$, symmetric in $bc$

In the units above ($q = -1$, so $e^3 = 1$), with $\langle X\rangle \equiv
(1/(N_kV_{cell}))\sum_k\sum_n X_n$:

```
sigma^ND_abc    = -tau^2 < w u_a u_bc >
sigma^BCD_abc   = (tau/2) < w (u_c Omega_ab + u_b Omega_ac) >
sigma^hall_abc  = - < w [ u_a Gt_bc - (1/2)(u_b Gt_ca + u_c Gt_ab) ] >
sigma^disp_abc  = - < w u_a Gt_bc >
sigma^intra_abc = -(1/2) < w d_a G_bc >  =  (1/2) < f' d_a G_bc >  =  -(1/2) < f'' u_a G_bc >
```

where in the last line $f'' = d^2f/d\varepsilon^2 = f(1-f)(1-2f)/T^2$ and the last form is one
more integration by parts on the mesh, $\sum_k f'(\varepsilon_n)\partial_a G_n = -\sum_k
f''(\varepsilon_n) u_a G_n$, so that no k-derivative is needed; the first form, with
$\partial_a G$ from the covariant derivative identity of `shg._sigma2_length` (`rgd`), is the
internal check of the last.

Derivations and sign chain.

- Nonlinear Drude. The Fermi-sea form is $-\tau^2\langle f\,\partial_a\partial_b\partial_c
  \varepsilon\rangle$ (Shibata Eq. (ND), Kaplan-Holder-Yan Eq. (fullcurrent2-drude), Wang
  Eq. (drudeexp), all with $e > 0$ and $-e^3\tau^2/\hbar^3$). Integrating by parts once gives
  $-\tau^2\langle w\, u_a u_{bc}\rangle$. `nonlineardrude._sigma` has the prefactor
  $(-1)^{l+1}/(i\omega + 1/\tau)^l = -\tau^2$ at $l = 2$, $\omega = 0$, times
  $\sum f\,\partial^3\varepsilon$, so the two agree with no sign to fix.
- Berry curvature dipole. Boltzmann with $\dot r = \partial_k\varepsilon - \dot k\times\Omega$,
  $\dot k = qE$, $f_1 = -\tau q E_d\partial_d f_0$: the anomalous current at second order is
  $j_a = -q\,\epsilon_{abc}\dot k_b\int\Omega_c f_1 = -(q^3\tau)\,\epsilon_{abc}E_bE_d
  \int f_0\,\partial_d\Omega_c$. Symmetrized in $(b,d)$ and with
  $\epsilon_{abc}\Omega_c = \Omega_{ab}$: $\sigma_{abd} = -(q^3\tau/2)\langle f(\partial_d
  \Omega_{ab} + \partial_b\Omega_{ad})\rangle = (\tau/2)\langle w(u_d\Omega_{ab} +
  u_b\Omega_{ad})\rangle$ for $q = -1$. This is Sodemann and Fu's $\chi_{abc}$ (arXiv:1508.00571,
  Eq. (chiabc)) times the factor 2 of their footnote for a real dc field, and Shibata's
  Eq. (BCD) with $e^3\tau/(2\hbar^2)$. Kaplan-Holder-Yan Eq. (fullcurrent2-berry) and
  Ezawa (arXiv:2409.09241) write $-e^3\tau/\hbar^2$ without the $1/2$, a factor 2 larger,
  since their "$+(a\leftrightarrow b)$" counts the symmetrization twice. The recursion
  settled it: $A_{-1} = \sigma^{BCD}/\tau$ to rounding with the $\tau/2$ above.
  In 2D only $\Omega_{xy} = \Omega_z$ exists, so $\sigma_{xyy} = \tau D_y$ and
  $\sigma_{yxx} = -\tau D_x$ with the dipole $D_a = \langle f\,\partial_a\Omega_z\rangle
  = \langle w\, u_a\Omega_{xy}\rangle$, the quantity to expose as `berry_curvature_dipole`.
- Quantum-metric terms. Three forms are in use, and they differ by gauge-invariant pieces, so
  the code exposes the pieces and names the conventions.
  - `gao` (Gao, Yang and Niu, arXiv:1402.2538, Eq. (cur); Wang et al. Eq. (inhexp);
    Shibata's inter-band term in the Fermi-surface form of his Eq. (dc-limit-ulrich-form)):
    $\sigma^{hall}$ above. Wang's $2e^3\sum\mathrm{Re}\int v^a A^b A^c/\varepsilon_{nm}\,
    f' - (a\leftrightarrow b)$ is $e^3\langle f'(u_a\tilde G_{bc} - u_b\tilde G_{ac})\rangle$,
    which symmetrized in $bc$ and with $f' = -w$, $e^3 = 1$ is $\sigma^{hall}$. It is purely
    transverse: $\sigma_{aaa}^{hall} = 0$.
  - `kaplan` (Kaplan, Holder and Yan Eq. (fullcurrent2-metric), their literal coefficients):
    $-\langle f[2\partial_a\tilde G_{bc} - \frac12(\partial_b\tilde G_{ca} +
    \partial_c\tilde G_{ab})]\rangle = \sigma^{hall} + \sigma^{disp}$, the Hall term plus the
    longitudinal dispersion term $-\langle w\,u_a\tilde G_{bc}\rangle$. Their dispersion
    current comes from the field-corrected band velocity $\partial_k\varepsilon'$ with
    their Eq. (energy_renorm) $\varepsilon' = \varepsilon + e^2 G^{ab}E_aE_b$ and $G$ equal to
    our $\tilde G$. The second-order Stark shift is $\frac{e^2}{2}\tilde G^{ab}E_aE_b$ (the
    polarizability check above), so with $f_0(\varepsilon)$ unshifted the dispersion current
    is $-\frac12\langle w\,u_a\tilde G_{bc}\rangle$, half of theirs, and with
    $f_0(\varepsilon')$ it cancels against the shifted distribution as a total derivative.
    The convention is kept as the paper writes it and documented with this factor; no fourth
    convention is added.
  - `shibata` (arXiv:2606.22359, Eqs. (intraQMD-band) and (interQMD-band), the velocity-gauge
    Keldysh clean limit, agreeing with the projector calculation of Ulrich et al. 2026):
    $\sigma^{hall} + \sigma^{intra}$, the Hall term plus the Fermi-surface dipole of the
    unnormalized metric. His Eq. (intraQMD-band) is $\frac{e^3}{2\hbar}\sum f'_n\partial_i
    \mathcal G^n_{jk}$, the middle form above with $e^3 = 1$.
  The default is `shibata`, decided on 7 November 2026: two independent derivations agree on
  it, its benchmark below is reproduced, and `kaplan` carries the factor 2 in the Stark
  shift. `gao` and `kaplan` stay available by name.

Selection rules worth having in the tests. $PT$ symmetry makes $\Omega \equiv 0$ at every
k, so $\sigma^{BCD} = 0$ to rounding while $\sigma^{hall}$ and $\sigma^{ND}$ are allowed. Time
reversal makes $\varepsilon(k) = \varepsilon(-k)$ and $\Omega(k) = -\Omega(-k)$, so
$\sigma^{ND} = 0$ and every quantum-metric term is zero, while $\sigma^{BCD}$ is allowed.
Inversion kills all of them. In 2D a rank-3 tensor symmetric in its last two indices has
two $C_3$-invariant patterns, $xxx = -xyy = -yxy = -yyx$ and $yyy = -yxx = -xyx = -xxy$, and
nothing else; the Berry curvature dipole is an in-plane pseudovector and vanishes under
$C_3$ (Sodemann-Fu), and a purely transverse tensor ($\sigma_{aaa} = 0$) has both patterns
zero, so $\sigma^{hall} \equiv 0$ under $C_3$ as well. A $C_3$ fixture therefore tests
$\sigma^{BCD} = \sigma^{hall} = 0$ identically and the two patterns for $\sigma^{ND}$,
$\sigma^{disp}$ and $\sigma^{intra}$; a fixture for a finite $\sigma^{hall}$ must break $C_3$.

### Degenerate multiplets

The clean-limit formulas assume $|\varepsilon_{nm}|\tau \gg 1$ and Shibata leaves
degeneracies open. We cannot, because the $PT$ fixtures are Kramers degenerate at every k.
The rule: every sum over $m$ excludes the multiplet $P(n)$, which makes
$\sum_{n\in P}\Omega^n$, $G^n$, $\tilde G^n$ the non-Abelian traces and gauge invariant.
A product of two band-diagonal quantities inside a multiplet is not invariant under a
rotation of the basis inside the block, so it is replaced by the block trace: with
$U_a = P h^a P$ and $X_{bc}$ the block matrix whose diagonal is $u_{bc}$, $\tilde G_{bc}$ or
$\Omega_{bc}$ (the same sums over $m \notin P$ with $\langle n|\dots|m'\rangle$ for
$n, m' \in P$), $\sum_{n\in P} w\, u^n_a X^n_{bc} \to w\,\mathrm{Tr}_P[U_a X_{bc}]$. The
weight $w$ is common to the multiplet. For a nondegenerate spectrum this is the formula
above; for an exact spin degeneracy without spin-orbit coupling it is exactly the sum of
the two spin copies. A true crossing of inequivalent bands on the Fermi surface at a mesh
point is a measure-zero event handled as in `anomaloushall.py`, and the broadened recursion
is the reference there. The test of this rule is direct: rotate the eigenvectors inside every
degenerate block by a random unitary and check that every term is unchanged. This is a
regularization, exact for spin degeneracy, not a non-Abelian derivation.

### Temperature, mesh and the Fermi-surface forms

All terms need $T > 0$ (default $T = 0.05$, as `drude_weight`) and `nk` such that the mesh
energy spacing is below $T$; the convergence test is to double `nk` at fixed $T$. The
Fermi-sea forms (with $f$ and a k-derivative) converge faster for a smooth integrand but
need the covariant derivatives; the Fermi-surface forms are the production path and the
Fermi-sea form of the Drude term (`nonlineardrude`) is the check.

### Spin and charge channels

For a spin-conserving Hamiltonian the two spin blocks are two independent problems
(`nonlineardrude._spin_channels`), so `channel="spin"`, `"up"`, `"dn"` run the formulas on
each block and combine them as `nonlineardrude._combine`. With spin-orbit coupling only the
charge channel is defined; a spin channel then raises with the message of
`nonlineardrude._check_spin_conserving`. The spin current at finite frequency (a spin-current
vertex in the recursion) is not built.

### 3D

`nonlineardc` is three-dimensional through the vertices of `shg.py`, whose
`_derivative_tensors` and `kubo._cell_factors` already handle $a_3$. `nonlineardrude` stays
two-dimensional: lifting it would need a three-index Taylor recursion and a `zzz` test on a
dispersing 3D model, and the decoupled-layers test of the first draft would have passed on
the cell-volume change alone without testing anything. The 3D Drude term comes from
`nonlineardc`.

## The two-frequency kernel: photocurrents and sum frequency

The general kernel at $n = 2$ with the multipliers $(c_1, c_2)$, $\omega_j = c_j\omega$:

- $(1,1)$ is SHG and must equal `shg.py` on the same mesh to rounding;
- $(1,-1)$ is the bulk photovoltaic effect, $\sigma_{abc}(0;\omega,-\omega)$, the rectified
  current under light, whose total-frequency denominator is $2i\delta$;
- $(1,2)$, $(2,-1)$ and so on are sum and difference frequency generation.

### The injection current and the shift current

The injection current is the part of the rectified response that grows linearly in time
until the relaxation cuts it: $dJ_a/dt = \eta_{abc}E_{0b}E_{0c}^*$ for
$E(t) = \mathrm{Re}\,E_0e^{-i\omega t}$. Its prefactor follows from the field convention
alone. With adiabatic switching the rate integrates to $J = \eta|E_0|^2/(2\delta)$, and
the convention gives $J_{dc} = \frac12\sigma(\omega,-\omega)|E_0|^2$, so the pole part of
$\sigma$ is $\eta/\delta$: $\eta = \delta\,\sigma^{pole}$, not $2\delta\,\sigma^{pole}$ as
the first draft had it. The golden rule gives the closed form directly. The $e^{-i\omega t}$
part of $-qE(t)\cdot r$ has amplitude $-qE_0\cdot r/2$, the transition rate $n \to m$ is
$2\pi\,(q^2/4)|E_0\cdot r_{mn}|^2\delta(\varepsilon_{mn} - \omega)$, and the current rate is
the rate times $q$ times the velocity difference, weighted by $f_n - f_m$:

```
eta_abc(w) = (pi q^3/2) (1/(Nk Vcell)) sum_k sum_{n != m} f_nm Delta^a_mn r^b_mn r^c_nm L_delta(w - eps_mn)
Delta^a_mn = v^a_mm - v^a_nn,   r^a_nm = h^a_nm/(i eps_nm),   L_delta(x) = (delta/pi)/(x^2 + delta^2)
```

with the sum excluding pairs inside a multiplet. This was checked on the Haldane-gapped
honeycomb lattice (mass $0.3$, $t_2 = 0.2$, $\omega = 2$, component `yxx`, `nk=96`):
$\delta$ times the length-gauge diagonal term $Z_{nn}$ equals this $\eta$ to six digits at
$\delta = 0.1$, $0.05$ and $0.025$ ($-0.009317$, $-0.010240$, $-0.010773$), so the closed
form is the length-gauge diagonal at any $\delta$. $\eta_{abc}$ is complex with
$\eta_{acb} = \eta_{abc}^*$: its real, $bc$-symmetric part is the injection under linearly
polarized light and vanishes under time reversal; its imaginary, $bc$-antisymmetric part
$\eta^{circ}_{abc} = (\eta_{abc} - \eta_{acb})/(2i)$ is the circular injection, present with
time reversal and broken inversion. Both come out of the same formula, and the component
convention is stated in the docstring.

The velocity-gauge $p = 0$ diagonal is not the same object: on the same fixture it gives
$-0.0859$ against the length-gauge $-0.0932$ at `nk=96` and $-0.0848$ against $-0.0922$ at
`nk=48`, a difference of order $\delta^0$ that does not shrink with the mesh. The sum rules
redistribute weight between the $p$ terms, so only the total is gauge invariant. A shift
current defined as "total minus the velocity-gauge diagonal" drifted by $12\,\%$ between
$\delta = 0.1$ and $0.05$ while the length-gauge one moved by $0.6\,\%$. So the shift
current is defined as

```
sigma^shift_abc(w) = sigma_abc(0; w, -w) - eta_abc(w)/delta
```

with both terms at the same $\delta$, which is finite as $\delta \to 0$ and the same in both
gauges once the mesh has converged. And the mesh matters when time reversal is broken: the
velocity and length totals at $(\omega,-\omega)$ on the Haldane fixture differ by $22\,\%$ at
`nk=24`, $6\,\%$ at `nk=48` and $0.2\,\%$ at `nk=96`, because the $1/\delta$ pole amplifies the
aliasing, while on the time-reversal symmetric gapped honeycomb lattice they agree to seven
digits at `nk=30` and the diagonal is zero identically. So the gauge test is tight only on
time-reversal symmetric fixtures, and loose at a converged `nk` on the Haldane one, with
$\eta$ tested through the closed form at fixed $\delta$ rather than by a scaling between two
deltas (the ratio of $\eta/\delta$ at two deltas is $2.20$, not $2$, because the Lorentzian
moves $\eta$ itself).

The one absolute benchmark of $\eta$ is the quantized circular photogalvanic effect of a
Weyl semimetal (de Juan, Grushin, Morimoto and Moore, arXiv:1611.05887): the trace of the
circular injection tensor over a frequency window where only one Weyl node absorbs is a
Chern number times a universal constant, independent of every convention. It is a 3D
workstation run on a lattice Weyl model, listed as a cuttable package; its implementer pulls
the TeX and derives the trace in the convention above.

### The length gauge at two frequencies

The independent path is the length gauge of `shg._sigma2_length` generalized to two
frequencies: $Y^c(\omega_2) = i f_{nm} r^c_{nm}/(\omega_2 + i\delta - \Delta\varepsilon_{nm})$,
$Z = [D^b, Y^c]$ with the generalized derivative from $[D^b, h^c] = h^{bc}$, and
$\sigma_{abc} = -q^3\sum h^a_{nm}Z_{mn}/(\omega_1 + \omega_2 + 2i\delta -
(E_m - E_n))$, averaged with the $(b,\omega_1)\leftrightarrow(c,\omega_2)$ partner. It is
exact for a cold insulator with a nondegenerate spectrum, including the diagonal $Z_{nn}$ that
is $\eta/\delta$. Selection rules for the tests: the shift current needs broken inversion and
allows time reversal (the gapped honeycomb lattice has it, with the $D_{3h}$ pattern); the
linear injection vanishes under time reversal and appears with a Haldane term.

## Higher orders at dc and at finite frequency

The general kernel at $\omega_j = 0$ gives the $n$-th order dc response at finite $\delta$,
with the Laurent structure above: the $1/\delta^n$ coefficient is $N_n/n!$ with
$\sigma^{ND,n} = \tau^n N_n$ from `nonlineardrude` (charge channel), which is the test. The
analytic $\tau$-decomposed third-order terms are the last package and are cut first if the
night runs short: the $\tau^3$ Drude from the band derivatives of `nonlineardrude._Channel`,
which works for any multiband Hamiltonian in the charge channel, and the $\tau^1$ term from
the Berry connection polarizability of Liu et al., arXiv:2106.04931, and Lai et al., Nat.
Nanotechnol. 16, 869 (2021). Its implementer pulls the TeX of 2106.04931 and derives the
prefactor in the conventions above before writing code; the Berry connection polarizability
is our $\tilde G$, validated by the static polarizability check above and by the symmetries
of the term, not by a Laurent coefficient, since at third order the one-pole terms mix. The
third-order $\tau^2$ term (the Berry curvature quadrupole) and the third-order $\tau^0$ term
are open.

For graphene's third harmonic the $D_{6h}$ identity $\sigma_{xxxx} = \sigma_{xxyy} +
\sigma_{xyxy} + \sigma_{xyyx}$ holds on a $C_6$-symmetric mesh at any resolution and is the
test. The feature at $3\omega = 2\mu$ of Passos's Fig. A is not: at $\mu = 0.1$,
$\delta = 0.011$ a mesh of `nk=120` has an energy spacing of $0.1$ to $0.15$, far above both
$\delta$ and $T$, and Passos's Appendix C shows the feature has zero weight as the broadening
goes to zero, so it is a property of the phenomenology and is not asserted.

## Fixtures

- Shibata's real two-band model, Eq. (real-qmd-model): $H = t_0\sin k_x\,\sigma_0 + (m +
  t_x\cos k_x + t_y\cos k_y)\sigma_x + \lambda\sin k_y\,\sigma_z$ with $(m, t_x, t_y, \lambda,
  t_0) = (2.2, 1.0, 0.7, 0.9, 0.5)$. Real, so $\Omega \equiv 0$ and $\sigma^{BCD} = 0$
  exactly, with a finite metric. Built as one site of the square lattice with the spin index
  as the orbital: `intra` $= m\sigma_x$, hopping along $+x$: $\frac{t_x}{2}\sigma_x -
  \frac{i t_0}{2}\sigma_0$ and its Hermitian partner along $-x$, along $+y$:
  $\frac{t_y}{2}\sigma_x - \frac{i\lambda}{2}\sigma_z$, set on the multicell hopping list as
  `test_shg.py` sets `h.intra`, and checked against $H(k)$ at a few k before use. His
  diagnostic at $\mu = 0.5$, $T = 0.05$: $\sigma^{intra}_{xyy} \simeq +1.16\times10^{-2}$,
  reproduced at `nk=250` and `nk=500` as $1.16245\times10^{-2}$ with the hoppings above and
  `h.shift_fermi(-0.5)`: `shift_fermi(e)` adds $e$ to the onsite energy, so $-\mu$ puts the
  chemical potential at $\mu$; the test checks this sign against the bands before asserting
  the value to 1 %. Gap $1.0$, a
  Laurent fixture.
- Shibata's tilted massive Dirac lattice model, Eq. (tilted-dirac-model), $H = t\sin k_x
  \sigma_0 + v_x\sin k_x\sigma_x + v_y\sin k_y\sigma_y + [m + b_x(1-\cos k_x) + b_y(1-\cos
  k_y)]\sigma_z$, $v_x = v_y = b_x = b_y = 1$, $m = 0.4$, $t = 0.3$: hopping along $+x$
  $-\frac{i t}{2}\sigma_0 - \frac{i v_x}{2}\sigma_x - \frac{b_x}{2}\sigma_z$, along $+y$
  $-\frac{i v_y}{2}\sigma_y - \frac{b_y}{2}\sigma_z$, `intra` $= (m + b_x + b_y)\sigma_z$,
  checked against $H(k)$ to $10^{-16}$. Finite BCD and metric; the closed d-vector forms
  $\Omega^\pm_{xy} = \mp\frac12\hat d\cdot(\partial_x\hat d\times\partial_y\hat d)$ and
  $G^\pm_{ij} = \frac14[\partial_i\theta\partial_j\theta + \sin^2\theta\,\partial_i\phi
  \partial_j\phi]$ pin $G$ and $\Omega$. Gap $0.8$, the Laurent fixture for $A_{-2}$ and
  $A_{-1}$ with $\mu = 0.6$.
- Ezawa's $d$-wave altermagnet with Rashba coupling (arXiv:2409.09241, Eq. (Hamiltonian)):
  $\frac{k^2}{2M}\sigma_0 + \lambda(k_x\sigma_y - k_y\sigma_x) + J(k_x^2 - k_y^2)\,n\cdot\sigma$,
  a $d_{x^2-y^2}$ form factor. `specialhamiltonian.dwave_altermagnet` is $J\sin k_x\sin k_y
  \sigma_z$, a $d_{xy}$ form factor rotated by 45 degrees, which moves the zeros of his law.
  The lattice fixture is built by hand: the square lattice with the first-neighbor hopping,
  the exchange $2J(\cos k_y - \cos k_x)\,n\cdot\sigma$ set through the multihopping dict
  (hopping $-J\,n\cdot\sigma$ along $\pm x$ and $+J\,n\cdot\sigma$ along $\pm y$), which is
  $J(k_x^2 - k_y^2)\,n\cdot\sigma$ at small k, and `h.add_rashba(lambda)` unrotated. His law
  at first order in $J$: $\sigma^{xx;x}_{Metric}(\Phi) = \sigma(0)\cos\Phi$ for
  $n = (\sin\Phi, \cos\Phi, 0)$ and the same for the Drude term, zero for $n \parallel x$ and
  $n \parallel z$ exactly by the mirror $M_yT$, both terms linear in $J$ with a sign flip under
  $J \to -J$. Test the profile against $\cos\Phi$ with a fitted amplitude, the two exact zeros,
  the linearity by halving $J$, and the sign flip. Do not use his absolute prefactors: his
  Eq. (HallAna) carries a $\tau$ in a $\tau^0$ quantity. Not a Laurent fixture (crossing at
  $\Gamma$).
- A $PT$-symmetric antiferromagnet with broken inversion and broken $C_3$ for the
  quantum-metric terms: the honeycomb lattice with a Neel exchange along $z$
  (`add_antiferromagnetism`), a Kane-Mele term (`add_kane_mele`) and uniaxial strain
  (`add_strain`), doped into a band with `shift_fermi`. Kramers degenerate at every k
  (splitting $4\times10^{-15}$, $|\Omega_1 + \Omega_2| \sim 10^{-14}$ measured without the
  strain). The first draft had Rashba instead of Kane-Mele, and Rashba is $P$-odd and
  $T$-even, so $PT$-odd: the lowest pair splits by $1.75$. Neel along $z$ gives
  $\varepsilon(k) \ne \varepsilon(-k)$ (measured $\sigma^{ND}_{yyy} = -2.95\times10^{-2}$ at
  `nk=120` and `nk=240` without strain, $\sigma^{BCD} = 0$, $\sigma^{disp} = -6.2\times10^{-3}$,
  $\sigma^{hall} = 0$ by $C_3$); an in-plane Neel vector with Kane-Mele has
  $\varepsilon(k) = \varepsilon(-k)$ and every $\tau$-even term vanishes. The strain breaks
  $C_3$ and keeps $PT$, so $\sigma^{hall}$ is finite there; its value is asserted nonzero
  only after doubling `nk`.
- A time-reversal symmetric metal with broken inversion and no $C_3$ for the Berry curvature
  dipole: the gapped honeycomb lattice under uniaxial strain (`h.add_strain`), doped. The
  strain breaks $C_3$; without it the dipole is zero exactly and the test must see both.
- A $C_3$ metal with broken time reversal for the two $C_3$ patterns of $\sigma^{ND}$,
  $\sigma^{disp}$ and $\sigma^{intra}$: the gapped honeycomb lattice with a Haldane term
  (`add_haldane`), doped. The first draft used the gapped honeycomb lattice without the
  Haldane term, where time reversal makes every $\tau$-even term vanish and the pattern test
  is $0 = 0$.
- The gapped honeycomb lattice and the three-sublattice triangular lattice of `test_shg.py`
  for the gauge comparisons at two frequencies and for the $D_{3h}$ patterns; the
  Haldane-gapped honeycomb lattice (mass $0.3$, $t_2 = 0.2$) for the injection current.
- The crystal of isolated molecules of `test_static_limit_matches_finite_field` for the
  absolute prefactor at third order: $\chi^{(3)}(0) = \sigma^{(3)}(0;0,0,0)/(3\delta)$, since
  $P^{(3)} = \sigma E^3/(-i\Omega)$ with $\Omega = 3i\delta$, against $\frac16 d^3P/dE^3$ by
  finite differences of the exact polarization.
- Decoupled layers stacked along $a_3$ for 3D: the 3D conductivity per unit length is the 2D
  one divided by the layer spacing.
- A lattice Weyl semimetal (two-band cubic model with two nodes) for the quantized circular
  photogalvanic effect, workstation only.

## The checks, each pairing two independent paths

- General kernel at $n = 1$ against `kubo.optical_conductivity` on an insulator at the same
  complex frequency, to the aliasing of the mesh (1e-8 at `nk=24` on the gapped honeycomb
  lattice is the expectation); at $n = 2$, $(1,1)$ against `shg.shg_conductivity` to 1e-12
  (same formula, same mesh); for a one-band model at $n = 2, 3, 4$ against the tadpole
  $(i^n q^{n+1}/n!)\sum f\,h^{(n+1)}/\prod(\omega_j + i\delta)$ with $h^{(n+1)}$ from
  `nonlineardrude._Channel.derivative`, which is an identity.
- Rotation covariance of the rank-4 tensor at third order (as `test_rotation_of_the_crystal`),
  the $D_{6h}$ relation of graphene's third harmonic, the permutation symmetry
  $\sigma_{abc}(\omega,2\omega) = \sigma_{acb}(2\omega,\omega)$, and the same for $(1,1,0)$
  and $(1,-1,0)$.
- Nonlinear Drude: the Fermi-surface form against `nonlinear_drude_conductivity` at $l = 2$
  (charge and spin channels on the $d$-wave altermagnet) and, where `_spin_channels` refuses
  (a Rashba metal), against `nonlineardrude._Channel(h.get_multicell(), ks).derivative`
  directly; at $l = 3$ for the third-order Drude.
- Berry curvature dipole: $\langle w\,u_a\Omega\rangle$ against $\langle f\,\partial_a
  \Omega\rangle$ with $\Omega$ from `anomaloushall.band_curvature_mesh` and a spectral
  derivative on the periodic mesh (reduced coordinates, then the Jacobian of
  `kubo._cell_factors`), and against the finite difference of the anomalous Hall sum under a
  rigid shift of the Fermi sea alone: $f(\varepsilon(k + \delta k))$ from the Hamiltonian
  with every hopping multiplied by $e^{i\,\delta k\cdot d}$ and $\Omega(k)$ from the
  unshifted one. Shifting both, as the first draft said, shifts the whole $H(k)$ and the
  mesh sum is invariant regardless of $D$.
- Quantum metric per band against `qgt.quantum_geometric_tensor_k(occ_idxs=[n],
  coordinates="cartesian")`; the band inverse mass $u_{ab}$ against
  `nonlineardrude._Channel`; $\tilde G$ against the static polarizability of an insulator
  from `kubo.optical_conductivity`.
- The two forms of $\sigma^{intra}$ ($f'\partial G$ and $f'' u G$) against each other.
- The Laurent coefficients of the dc recursion by the contour: $A_{-2} = \frac12\sigma^{ND}/
  \tau^2$ and $A_{-1} = \sigma^{BCD}/\tau$ on the tilted Dirac model, $A_{-3} =
  \sigma^{ND,3}/(6\tau^3)$ at third order; $A_0$ recorded above, not asserted.
- Gauge agreement velocity against length at $(\omega, -\omega)$ and $(\omega, 2\omega)$:
  to 1e-10 of the signal on the time-reversal symmetric insulators, to a loose tolerance at
  a converged `nk` on the Haldane fixture; $\eta$ against the closed form at fixed $\delta$.
- Degeneracy rule: a random unitary inside every degenerate block leaves every dc term
  unchanged, on a fixture that has degenerate blocks.
- Every option checked: unknown `term`, `qmd`, `channel`, a spin channel with spin-orbit
  coupling, $T = 0$ for a Fermi-surface routine, $\delta = 0$, a Nambu Hamiltonian, a
  component string of the wrong length, a multiplier tuple of the wrong length, an unknown
  `prec`, following the error conventions skill.

## The packages

Four waves, disjoint files inside a wave. Every test and benchmark runs on the GPU
workstation (a float32 card, so a double-precision device timing says nothing), and the
timings are taken pinned to the performance cores with no other job running. The most
valuable pieces come first so that the last waves can be cut.

Two coordination rules for packages that run at the same time in one tree. The signatures
of `shg._derivative_tensors`, `shg._eigenbasis_derivatives`, `shg._cartesian_derivative` and
`shg.shg_conductivity` are frozen during wave 1, since `dc2` and `kernel` import them while
`shg-twofreq-3d` edits the file; the identity test of the kernel against
`shg.shg_conductivity` is run at the end of the wave. In wave 2 the public functions of
`nonlinearkernel.py` are frozen, since `photocurrent` and `dc-crosscheck` call them while
`gpu` adds the backend inside the same file.

Wave 1, no dependencies, about 2.5 h wall clock.

- `dc2` (2.5 h): `conductivitytk/nonlineardc.py`, `tests/conductivity/test_nonlinear_dc.py`,
  and the docstring correction of `nonlineardrude.py`. The second-order dc terms above, all
  conventions, channels, block traces, the fixtures and checks that need no other new code.
- `kernel` (2.5 h): `conductivitytk/nonlinearkernel.py`,
  `tests/conductivity/test_nonlinear_kernel.py`. The general recursion with the numpy
  backend, 1D to 3D, complex frequencies in, the symmetrization, the adaptive k-batch, the
  general-$n$ susceptibility conversion.
- `shg-twofreq-3d` (1 h): `conductivitytk/shg.py`, `tests/conductivity/test_shg.py`. The length
  gauge at two frequencies as a module-level function the photocurrent package imports, the
  closed-form $\eta$ check against its diagonal, and the 3D guard lifted with the
  decoupled-layers test.

Wave 2, depends on wave 1, about 1.5 h.

- `photocurrent` (1.5 h): `conductivitytk/photocurrent.py`,
  `tests/conductivity/test_photocurrent.py`. Shift, linear and circular injection, sum and
  difference frequency on the general kernel, gauge checks, selection rules.
- `dc-crosscheck` (0.5 h): `conductivitytk/nonlineardcfit.py` (the contour extraction),
  `tests/conductivity/test_nonlinear_dc_crosscheck.py`. Asserts $A_{-2}$, $A_{-1}$ at second
  order and $A_{-3}$ at third, records $A_0$.
- `gpu` (1.5 h): the jax backend inside `nonlinearkernel.py`,
  `tests/conductivity/test_nonlinear_gpu.py`, the device benchmark. Routed on
  `gpu.get_gpu()`, precision per call through `gpu.resolve_prec`, `gpu` imported before any
  jax array as the gpu-backend skill says. Starts only with the maintainer's sign-off.

Wave 3, depends on wave 2, about 1.5 h; all three are cuttable, `wannierberri` first.

- `dc3` (1.5 h): third order in `nonlineardc.py` (the $\tau^3$ Drude for any multiband charge
  channel, the Berry-connection-polarizability $\tau^1$ term after reading 2106.04931),
  `tests/conductivity/test_nonlinear_dc3.py`.
- `cpge` (1 h): the quantized circular photogalvanic effect of a lattice Weyl semimetal in
  `tests/conductivity/test_cpge.py`, a 3D workstation run; the circular injection it tests
  lives in `photocurrent.py` from wave 2.
- `wannierberri` (1.5 h): the external benchmark of the Berry curvature dipole and the shift
  current, importing `nonlineardc` and `photocurrent` directly. The wheel is on PyPI (26.10);
  install with `pip install --target` into a scratch directory, never into the venv, with a
  20-minute cap on the dependency tree; write a `_tb.dat` of a pyqula model, compare.

Wave 4, depends on wave 3, about 2 h.

- `integration` (2 h): `conductivity.py`, `hamiltonians.py`, the guide page, the reference
  entries, `README.md`, `mkdocs.yml`, the examples, this file's status, the stale bullets of
  `second_harmonic_generation.md` and `nonlinear_spin_transport.md`, the index in
  `future_development/README.md`, `documentation/gpu_porting_plan.md`, `tests/documentation`,
  the suite recount. Documents `order=3` of the dc routine only if `dc3` landed.

About 15.5 agent-hours over 7.5 h of wall clock, inside the window to 08:00 with the
cuttable packages as the margin.

## The public API

One general function with keywords, thin wrappers that say what the caller wants, and
one-line delegators on `Hamiltonian`.

```python
conductivity.nonlinear_conductivity(h, order=2, energies=None, frequencies=None, nk=20,
        T=None, delta=0.1, components=None, degeneracy_tol=1e-6, prec="double")
# frequencies: the multipliers c_j, omega_j = c_j * energies; None is all ones (harmonic
# generation); (1,-1) is the rectified current; (1,1,0) the field-induced second harmonic;
# energies=[0.] with any multipliers is the adiabatic-switching dc value, not the
# tau-expanded one. components: strings of order+1 labels, "xyy", "xxyy"; the first is the
# current. returns (energies, array or dict), as shg_conductivity.
conductivity.nonlinear_susceptibility(h, ..., length_unit=1.0, energy_unit=1.0)
# chi^(n) = sigma^(n)/(-i Omega), Omega the sum of the complex frequencies, any order
conductivity.nonlinear_dc_conductivity(h, order=2, term="all", tau=1., qmd="shibata",
        channel="charge", nk=100, T=0.05, components=None, degeneracy_tol=1e-6)
# term: "all" (a dict with "drude", "bcd", "qmd" and "total", each with its power of tau
# included), or one of the registry names "drude", "bcd", "qmd", "qmd_hall", "qmd_disp",
# "qmd_intra"; qmd: "shibata" (default), "gao", "kaplan".
conductivity.berry_curvature_dipole(h, ...)      # D_a = <w u_a Omega_xy> in 2D, D_ab in 3D
conductivity.shift_current(h, ...)               # sigma(0; w, -w) - eta/delta
conductivity.injection_current(h, ...)           # eta_abc(w), complex: Re linear, Im circular
conductivity.shg_conductivity(...)               # unchanged, now also 3D
h.get_nonlinear_conductivity(), h.get_nonlinear_susceptibility(),
h.get_nonlinear_dc_conductivity(), h.get_berry_curvature_dipole(),
h.get_shift_current(), h.get_injection_current()
```

String options go through registries (`get_term_names()`, `get_qmd_conventions()`,
`get_channel_names()`), Hilbert-space guards through `check.py`, Nambu rejected as in
`kubo._orbital_positions`, precision through `gpu.resolve_prec`.

## What to optimize if time is left, and how to measure it

- The general kernel at $n = 2$ against the numba SHG kernel on 24 bands, 100 frequencies,
  `nk=40`: the numba path takes 2.2 s on 14 threads. If the array kernel is more than 3x
  slower, add a numba path for the equal-frequency case.
- The device kernel in single precision against numpy on the performance cores, at 2, 8, 24,
  64 and 100 bands, `nk=40`, orders 2 and 3, 1 and 100 frequencies, at $\omega \ge 0.2$ where
  single precision is meaningful: the crossover in bands and the error of single precision
  against double, recorded as conclusions here, without the host name.
- The dc routine at `nk=200`, 24 bands: the target is under 30 s on the CPU, the cost is the
  batched diagonalization and three einsums per k-batch.
- The memory of the vertices at fourth order with 100 bands, and whether the needed-suffix
  set keeps a few components cheap.
- The contour: whether 12 points suffice instead of 24 (`nk=150` already equals `nk=300`).

## Risks

- The quantum-metric conventions differ in the literature by gauge-invariant pieces, and the
  $\delta^0$ coefficient of the recursion matches none of them, as measured. The plan exposes
  the pieces, asserts only the pole coefficients, and defaults to `shibata` with the reasons
  recorded above.
- Fermi-surface sums converge slowly in `nk` at small $T$, and a fixture with a band touching
  near the Fermi level makes the clean-limit formulas meaningless; every fixture is checked
  for convergence by doubling `nk` before any nonzero value is asserted.
- The gauge comparison at $(\omega,-\omega)$ with broken time reversal converges slowly in
  `nk` because of the $1/\delta$ pole; the tests know this and use the time-reversal symmetric
  fixtures for the tight tolerance.
- The block-trace rule for degenerate multiplets is a regularization, tested by random
  unitaries, not a derivation.
- Nine packages in the night window is tight; wave 3 is the margin, `wannierberri` first.

## Parked, deliberately

- The thermal (spin-Nernst, spin-Seebeck) channel stays parked, see
  `nonlinear_spin_transport.md`.
- The spin current at finite frequency (a spin-current vertex in the recursion; spin
  photocurrents). The spin channel is dc-only through the spin-conserving block
  decomposition.
- Routing `shg_conductivity` through the general kernel: `shg.py` keeps the numba kernel and
  the triangle and qtci integrations, which are SHG-only.
- A triangle or qtci integration of the general kernel; the grid is the only integration.
- The 3D lift of `nonlineardrude.py`; `nonlineardc` covers the 3D Drude term.
- The third-order $\tau^2$ (Berry curvature quadrupole) and $\tau^0$ dc terms.
