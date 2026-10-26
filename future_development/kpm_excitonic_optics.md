# The Chebyshev resolvent of the excitonic optics: what is built and what is left open

The excitonic absorption and second harmonic (`conductivitytk/excitonicoptics.py`) take
`method="dense"|"kpm"` since 26 October 2026. With `"kpm"` the resolvent of the Tamm-Dancoff
BSE block A is never formed through eigenpairs: `bsetk/chebyshev.py` expands
(z - A)^-1 in Chebyshev polynomials of A at the complex energy z = w + i delta, driven by
`KernelFactorization.matvec`, so the cost is linear in the number of pairs and the memory is
the pair basis plus the low-rank factors. This file records why it looks the way it does,
what was measured, and what was deliberately not built.

## The formula, and why there is no kernel

With the spectrum of A inside [c - s, c + s], A' = (A - c)/s and z' = (z - c)/s,

    (z - A)^-1 = (1/s) sum_n (2 - delta_n0) q^n / r T_n(A'),   r = sqrt(z'^2 - 1),  q = z' - r,

on the branch |q| < 1 (Weisse et al. arXiv:cond-mat/0504627, continued to complex energy as in
Ferreira and Mucciolo arXiv:1507.00488 and Braun and Schmitteckert arXiv:1310.2724). The
imaginary part of z damps the series by itself, so no Jackson kernel is used, and the converged
result is the Lorentzian-broadened dense result exactly, not a line shape of equal width. That
is what makes every check an equality on the same mesh. The number of moments is computed from
the largest |q| of the requested energies for a truncation of 1e-9 (`chebyshev.tol`), which is
about 20 s/delta at the center of the spectrum and does not depend on the number of pairs.

The spectral interval comes from two `eigsh` calls on the factorized operator with a
deterministic start vector, widened by 2%. The ARPACK objection of `iterative.py` (it cannot
resolve a degenerate multiplet) does not apply: an extreme eigenvalue is wanted, not its
eigenspace. The diagonal moments <d|T_n(A')|d> are checked against |d|^2 by
`kpmtk/scaleguard.check_scale` on every block of the recursion, which is the backstop if the
interval misses an edge (pinned by `test_chebyshev_scale_is_guarded`). A polynomial of A never
selects an eigenvector, so the four-fold degenerate exciton levels of a spinful model without
spin-orbit coupling, which broke ARPACK and the seeded LOBPCG (see `bse_excitons.md`), play no
role; `test_chebyshev_route_equals_the_dense_one[True]` is that case.

The linear response needs only d_a^+ (z - A)^-1 d_c, so it is done from the nine moment
sequences of one recursion, for every frequency at once (`resolvent_braket`). The second
harmonic needs the vectors, since the position operator sits between two resolvents: each SHG
term is d^+ (z1 - A)^-1 X (z2 - A)^-1 d, two resolvents and not three (the three-resolvent
structure belongs to the velocity gauge of `shg.py`). The four resolvent sets of a frequency
batch, at w, -2w, conj(2w) and -conj(w), come from one recursion on the three dipole vectors
(`resolvent_apply`), the vectors being accumulated in blocks of 32 so that the sum over n is a
matrix product. The dense route was moved onto the same interface, which also stopped it from
recomputing the left resolvents once per component.

## What was measured

Gapped honeycomb lattice with the Coulomb tail of the excitons page, nk=18: the two routes agree
to 2e-10 (linear) and 3e-11 (SHG) of the maximum at delta=0.1 and 0.05; the kpm route takes 0.2
to 0.4 s against 0.8 to 1.6 s dense. At nk=60, 131 frequencies, the absorption takes 7 s and the
SHG susceptibility 31 s, with the exciton peak (1.84 t, 6.88 e^2/h) and the 2w peak (0.92 t,
0.244 nm^2/V) unchanged from nk=30 to six digits.

Spinful three-band MoS2 with a model Coulomb tail (2/sqrt(r^2+1) eV, r in Angstrom), 16
frequencies of chi_yyy at delta=0.05: dense 15 s at nk=18, kpm 2.6 s at nk=18 (agreeing to
2.5e-10), 31 s at nk=36, 104 s at nk=60.

The all-band kernel="none" comparison with the velocity gauge of `shg.py`, which the dense BSE
could not make at nk>24, on 16 frequencies from 0 to 1.5 eV:

- Monolayer MoS2 (spinful, SOC, every band). The velocity gauge is converged at nk=180 to 2e-6
  (delta=0.05) and 2e-4 (delta=0.025). The excitonic route converges toward it, with a largest
  error, relative to the maximum of |sigma_yyy|, of 1e-2, 6e-4, 5e-5, 1e-5, 2e-6 at nk = 24, 48,
  72, 96, 120 for delta=0.05, and 4e-2, 5e-3, 8e-4, 4e-4, 2e-4 for delta=0.025. On the same mesh
  the two gauges differ by 1e-3 at nk=24, 4e-6 at nk=48 and 1e-11 at nk=120, so what is left at
  a given mesh is the mesh, shared by both. C3 (yxx = -yyy) holds to 1e-14 on every mesh. The
  kpm run at nk=120 (115,200 pairs) takes 27 s at delta=0.05.
- MoS2/MoS2/WSe2/WSe2 tetralayer, 2H, t_perp=0.1 eV, spinful, every band (128 pairs per
  k-point), delta=0.05. The velocity gauge at nk=120 is the reference (2e-3 from nk=180). The
  excitonic route has a largest error, relative to the maximum of |sigma_yyy|, of 45%, 7%, 8%,
  2.1%, 0.7% at nk = 24, 36, 48, 72, 96, always at the top of the window, 2w = 3 eV; the velocity
  gauge itself is 39%, 9%, 6% off on the meshes 24, 36, 48, so the slow part is the resolution
  delta >= v dk/2 and not the gauge. C3 holds to 1e-13. Time: 22 s at nk=24, 34 s at 36, 63 s at
  48, 200 s at 72 and 420 s at 96 (1.18 million pairs, 12 GB peak at wbatch=4); the velocity
  gauge at nk=120 takes 26 s, so for a calculation without interaction it stays the tool.

## Left open

The numbers of the first two items were measured while planning, on the dense matrices of
the same models, before the implementation.

- **Non-Tamm-Dancoff linear absorption.** The Chebyshev resolvent of the pseudo-Hermitian full
  matrix H = [[A,B],[-B^+,-conj(Abar)]] converges at the same geometric rate, with the half
  width doubled by the +-E branches: against a direct solve at nk=12 (288x288, eigenvector
  condition number 11), 2.9e-2, 1.5e-3, 1.9e-6 at N = 400, 800, 1600, delta=0.05. So it costs
  about five times the Tamm-Dancoff one and needs no Lanczos in the S@H inner product, which is
  what `bse_excitons.md` thought was missing; the Chebyshev recursion is not an
  orthogonalization and has no metric to preserve. The trap is that -B^+ applies as
  sum_t conj(c_t)|R_t><L_t|, not as the conjugate of the B matvec. What has to come first is a
  dense non-Tamm-Dancoff absorption with the X and Y amplitudes as the exact reference, which
  `excitonicoptics.py` does not have. The SHG stays Tamm-Dancoff: neither source paper
  (arXiv:2310.09674, arXiv:1804.04869) has a non-Tamm-Dancoff formula.
- **The term count is the wall for large cells.** `density_interaction` couples every
  spin-orbital pair of two sites, so nterm is 240 for the three-band monolayer and 1104 for the
  tetralayer (with exchange) at the default rcut=5, 9816 at rcut=10. With every band and the
  interaction on, the tetralayer at nk=48 has 295k pairs, 10 GB of factors and 4e9 flop per
  matvec on three columns, so a spectrum takes an hour or two and nk=120 (65 GB) is out of reach.
  The fix is the direct term as a convolution on the mesh, per orbital pair: gather
  z_ab(k) = sum_{vc} u2[a] conj(w1[b]) x over the bands, convolve with the tabulated W_ab(q)
  that `kernel.interaction_tensor` builds by FFT, scatter back. Cost O(norb^2 (npair + nk^d log
  nk)) per column and memory O(norb^2 nk^d), which beats the low-rank apply by the number of
  lattice vectors inside rcut (2x at rcut=5, 17x at rcut=10 on the tetralayer) and takes a
  tabulated RPA-screened W without truncation, which the factorization refuses. The engine
  takes a matvec callable, so the two applies can coexist behind it.
- **An exciton density of states** by a stochastic trace with the Jackson kernel, the analogue
  of `kpm.random_trace`, on the same engine. Not asked for.
- **The GPU port** of the recursion, a (nterm,npair)@(npair,ncol) GEMM pair per step, the shape
  `gpu-backend` calls favourable; it is a tier of the porting plan that wants sign-off, and the
  single-precision drift of the recursion at N in the thousands has to be measured first.

Deliberately not built: the two-variable Chebyshev expansion of the second-order conductivity
(Joao and Lopes arXiv:1810.03732), N^2 moments for disordered independent particles, when
`shg.py` is exact and the excitonic SHG needs no two-variable moment tensor because the
frequencies factor out of the vectors; and a truncation of the exciton sum to an energy window,
pointless once the resolvent is matrix-free.

The benchmark is the in-repository dense BSE (the maintainer's decision, 26 October 2026): only
the numerical route is new, the formalism is the one already pinned there. The external
structural precedents for absorption without diagonalizing the BSE are the Haydock solver of
Yambo (arXiv:0810.3118, arXiv:0809.3389) and the structure-preserving Lanczos of BerkeleyGW
(arXiv:1611.02348).
