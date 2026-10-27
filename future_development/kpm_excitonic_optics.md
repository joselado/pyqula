# The Chebyshev resolvent of the excitonic optics: what is built and what is left open

The excitonic absorption and second harmonic (`conductivitytk/excitonicoptics.py`) take
`method="dense"|"kpm"` since 26 October 2026. With `"kpm"` the resolvent of the Tamm-Dancoff
BSE block A is never formed through eigenpairs: `bsetk/chebyshev.py` expands
(z - A)^-1 in Chebyshev polynomials of A at the complex energy z = w + i delta, driven by a
matrix-free product with A, so the cost is linear in the number of pairs. The product is the
convolution apply of `bsetk/convolution.py`, whose memory is a few orbital matrices per k-point,
and with `gpu.set_gpu(True)` the whole recursion runs on the device (`bsetk/chebyshevjax.py`).
This file records why it looks the way it does, what was measured, and what was deliberately
not built.

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

The spectral interval comes from one Lanczos run without reorthogonalization, which gives both
edges at once, from a fixed-seed random start vector, extended 16 steps at a time until both edges
move by less than 1e-5 of the width, and widened by 2%. On the tetralayer at nk=36 that is 48
steps, with the edges within 6e-5 of the width of the converged ones, far inside the margin. It
replaced two ARPACK `eigsh` calls at a tolerance of 1e-4, whose host work was most of their cost:
on the tetralayer at nk=48 they took 1.9 s with the device matvec, against 0.07 s for the same
Lanczos compiled on the card, and 4.6 s on the CPU against about 2 s. The loss of orthogonality
only duplicates converged Ritz values, so the extreme ones are untouched. The ARPACK objection of
`iterative.py` (it cannot resolve a degenerate multiplet) does not apply either: an extreme
eigenvalue is wanted, not its
eigenspace. The diagonal moments <d|T_n(A')|d> are checked against |d|^2 by
`chebyshev.check_moments` (on `kpmtk/scaleguard`) on every block of the recursion, which is the backstop if the
interval misses an edge (pinned by `test_chebyshev_scale_is_guarded`). A polynomial of A never
selects an eigenvector, so the four-fold degenerate exciton levels of a spinful model without
spin-orbit coupling, which broke ARPACK and the seeded LOBPCG (see `bse_excitons.md`), play no
role; `test_chebyshev_route_equals_the_dense_one[True]` is that case.

The linear response needs only d_a^+ (z - A)^-1 d_c, so it is done from the nine moment
sequences of one recursion, for every frequency at once (`resolvent_braket`). A is Hermitian, so
T_m T_n = (T_{m+n} + T_{|m-n|})/2 gives mu_2n = 2 <v_n|v_n> - mu_0 and mu_2n+1 =
2 <v_n+1|v_n> - mu_1 from vectors of half the order, which halves the matvecs; the loop keeps
only the last two vectors and takes the two 3x3 products of each new one by BLAS, with the
conjugation inside `zgemm` rather than on a copy. The second
harmonic needs the vectors, since the position operator sits between two resolvents: each SHG
term is d^+ (z1 - A)^-1 X (z2 - A)^-1 d, two resolvents and not three (the three-resolvent
structure belongs to the velocity gauge of `shg.py`). The four resolvent sets of a frequency
batch, at w, -2w, conj(2w) and -conj(w), come from one recursion on the three dipole vectors
(`resolvent_apply`), the vectors being accumulated in blocks of 32 so that the sum over n is a
matrix product. The recursion does not depend on the frequency, so every batch of frequencies
runs it again; `wbatch=None` takes as many frequencies as fit in 2 GB
(`excitonicoptics.batch_memory`), which put the 131 frequencies of the example in one batch and
took it from 12 s to 8 s. The budget is host memory; on the device the result of a batch lives
on the card, about 1 GB in double for the tetralayer at nk=96, where the budget picks four
frequencies. With `kernel="none"` the block is diagonal and the resolvent is
taken exactly, without any expansion. The dense route was moved onto the same interface, which also stopped it from
recomputing the left resolvents once per component.

## The product with A: a convolution on the mesh

The first version drove the recursion with `KernelFactorization.matvec`, one rank-one
projector per non-zero entry (a,b,d) of the real-space interaction. That costs nterm x npair
per column with nterm = norb^2 times the lattice vectors inside the cutoff, 276 for the
three-band monolayer and 1104 for the TMD tetralayer at rcut=5, and the factors themselves were
the memory wall (10 GB for the tetralayer at nk=48 with every band). The same direct term is a
convolution over the mesh: gather the pair vector into the orbital matrix
Z_ab(k) = sum_vc C_c(k)_a x_kvc conj(C_v(k)_b), convolve each channel (a,b) with W_ab tabulated at
the mesh differences k - k_0 by FFT, and project back with conj(C_c(k)_a) C_v(k)_b. The table is
the real-space dictionary folded onto the nk-periodic supercell, which is what the Bloch phases
of the factorization do as well, so the two applies are equal to machine precision
(`tests/bse/test_bse_convolution.py` against `kernel.build_blocks`, 1e-13, every kernel, the
chain, the honeycomb lattice, an onsite U whose channels miss the orbital diagonal, the
three-band TMD). The exchange term at Q=0 is the orbital diagonal of the same Z summed over the
mesh and contracted with W(0), so it costs nothing extra. The cost is O(npair norb + nch N log N)
per column, independent of the range of the interaction, and a tabulated RPA-screened W
(`screening="rpa"`, also with `nkW` a multiple of `nk`) enters as it is, which the factorization
refuses; the kpm and dense routes agree to 4e-10 with it, for the absorption and the SHG.

Only the channels where W does not vanish are gathered and transformed (144 of 576 for the
tetralayer), plus the orbital diagonal when the exchange term is on. The gather and the scatter
are two numba `prange` kernels over k, `_gather` and `_scatter`, which write into buffers kept
per number of columns: a fresh orbital-matrix array page-faults on every call, and reusing it was
1.7x on the gather alone. Measured per matvec on three columns, idle: on the MoS2 monolayer at
nk=60 the low-rank apply took 7 ms, the einsum convolution 19 ms, the numba one 4.4 ms; on the
tetralayer at nk=48, einsum 56 ms and numba with the channel selection 25 ms, against 85 to 146
ms for the low-rank apply at nk=24 already. The position operator of the SHG does the same
gather, FFT and scatter with every channel and the bond vector in place of W, and moved onto the
same kernels went from 1.2 s to 0.19 s per call on the tetralayer at nk=48. The low-rank
factorization itself got one fix that the iterative and the quantics solvers share:
`right.conj()@X` copied the whole (nterm,npair) factor on every call, 3.0 of 4.3 s of the
first profile, and is now `conj(right@conj(X))`.

The recursion step 2 (A v - c v)/s - v_n-1 is one in-place numba loop on the output of the
matvec (`chebyshev._combine`), and the starting block is made C-ordered: it arrived as the
transpose of the dipole array, numpy kept that layout through every vector, and every matvec
then copied its input.

## The device path

`bsetk/chebyshevjax.py` writes the matvec in jax and runs the recursion in compiled blocks of 16
steps (`jax.lax.scan`), folding each block into the result with one matrix product, so the host
only sees the diagonal moments of the scale guard; the brakets of the absorption use the halved
recursion as on the CPU, and the Lanczos run of the bounds is compiled on the device too, so
that only its tridiagonal elements come back. It routes on
`gpu.get_gpu()`, and precision is per call, `kpm_prec="double"` (default) or `"single"`; on the
CPU `"single"` raises. Unlike the numba kernels, the device matvec gathers and scatters whole
orbital matrices with dense contractions and picks the channels only for the FFT: a
channel-by-channel scatter lowers to atomic adds, and was 3x slower in single precision on the
tetralayer (8.4 ms against 2.7 ms per matvec at nk=48, on an RTX A2000).

The kernels are module-level jitted functions that take the device arrays of the block as
arguments, with the shapes as a static argument, rather than closures over the arrays. With
closures every call made a new block, so jax traced and compiled everything again: 0.45 s for
the Lanczos run and 0.7 s for the recursion of each call on the tetralayer at nk=48 in single
precision, out of 3.2 s for the absorption. Now a second call on the same mesh reuses the
compiled code, and only the first call of a process compiles.

Every contraction asks for `Precision.HIGHEST`. Without it jax runs float32 products on an
Ampere or later card in TF32, with a ten-bit mantissa, and the single-precision resolvent came
out 8e-4 away from the double one on the A2000 against 4e-6 on jax's CPU backend; with it, 3e-6.
This holds for any float32 jax product in the package, not only this one.

## What was measured

The timings recorded here first (31 s at nk=36 and 104 s at nk=60 for the MoS2 SHG below) were
taken while the machine was loaded; idle, the same code took 4.8 s and 18.2 s. Every number
below was taken idle.

Gapped honeycomb lattice with the Coulomb tail of the excitons page, nk=18: the two routes agree
to 2e-10 (linear) and 3e-11 (SHG) of the maximum at delta=0.1 and 0.05. At nk=60, 131 frequencies,
the exciton peak (1.84 t) and the 2w peak (0.92 t) are unchanged from nk=30 to six digits, and the
example (`examples/2d/excitonic_optics_kpm`, both meshes, both responses) runs in 8 s.

Spinful three-band MoS2 with a model Coulomb tail (2/sqrt(r^2+1) eV, r in Angstrom), 16
frequencies of chi_yyy at delta=0.05, and the absorption at 301 energies, on a 14-thread
workstation: with the first version 0.7, 4.8 and 18.2 s for the SHG at nk = 18, 36, 60 and 29.9 s
for the absorption at nk=60; after the conj fix, the Lanczos tolerance and the halved brakets
0.3, 1.3, 3.6 and 4.8 s; with the numba convolution 0.3, 0.8, 2.5 and 3.1 s. The results did not
move by more than 3e-13 through any of it.

MoS2/MoS2/WSe2/WSe2 tetralayer with the same Coulomb tail, every band (128 pairs per k-point),
SHG at 16 frequencies, delta=0.05: with the low-rank apply the plan put nk=48 at an hour or two
and 10 GB; with the einsum convolution 13.8 s at nk=24, 49.6 s at 36, 75.0 s at 48 (3.6 GB); with
the numba kernels and the channel selection 7.8 s at 24, 36 s at 48 (3.4 GB) and 102 s at 72
(7.1 GB), before the last two changes to the recursion step and the brakets, which took the
absorption at nk=48 from 30 s to 23 s. The spectra of all these versions agree to 1e-13.

On a workstation with an RTX A2000 12 GB (a consumer Ampere card, float64 at about 1/70 of
float32) and the six performance cores of an i5-12600K for the CPU column, the whole calls
end to end (SHG at 16 frequencies / absorption at 301 energies), with the interaction:

| case | CPU | GPU double | GPU single | single against CPU |
| --- | --- | --- | --- | --- |
| MoS2, nk=60, delta=0.01 | 12.5 / 7.7 s | 8.9 / 6.9 s | 3.8 / 3.0 s | 1.3e-5 / 1.9e-5 |
| tetralayer, nk=48 | 23.9 / 16.2 s | 15.4 / 12.3 s | 7.5 / 5.4 s | 2.7e-5 / 4.3e-5 |
| tetralayer, nk=72 | 55.0 / 36.5 s | 52.9 / 27.2 s | 16.6 / 12.5 s | 1.3e-5 / 4.5e-5 |

Double precision on the card agrees with the CPU to 1e-9 and gains little, as expected from a
card this weak in float64; single precision is 2.6 to 3.3 times faster, with an error of 1e-5
to 5e-5 relative to the maximum of the response, the absorption being the worse one. The
resolvent alone gains more (the tetralayer at nk=48: 21.9 s on the CPU, 10.3 s in double and
5.4 s in single on the card, measured with the first version of the CPU kernels and with
`Precision.HIGHEST`); what is left is the pair basis, the position operator on the host and the
compilation, about 1.3 s per process. On a data-centre card the double column
would move, the single one less.

The all-band kernel="none" comparison with the velocity gauge of `shg.py`, which the dense BSE
could not make at nk>24, on 16 frequencies from 0 to 1.5 eV (taken with the first version, whose
results the later ones reproduce to 1e-13; with kernel="none" the resolvent is now diagonal and
exact, so these runs would take seconds):

- Monolayer MoS2 (spinful, SOC, every band). The velocity gauge is converged at nk=180 to 2e-6
  (delta=0.05) and 2e-4 (delta=0.025). The excitonic route converges toward it, with a largest
  error, relative to the maximum of |sigma_yyy|, of 1e-2, 6e-4, 5e-5, 1e-5, 2e-6 at nk = 24, 48,
  72, 96, 120 for delta=0.05, and 4e-2, 5e-3, 8e-4, 4e-4, 2e-4 for delta=0.025. On the same mesh
  the two gauges differ by 1e-3 at nk=24, 4e-6 at nk=48 and 1e-11 at nk=120, so what is left at
  a given mesh is the mesh, shared by both. C3 (yxx = -yyy) holds to 1e-14 on every mesh.
- MoS2/MoS2/WSe2/WSe2 tetralayer, 2H, t_perp=0.1 eV, spinful, every band, delta=0.05. The
  velocity gauge at nk=120 is the reference (2e-3 from nk=180). The excitonic route has a largest
  error, relative to the maximum of |sigma_yyy|, of 45%, 7%, 8%, 2.1%, 0.7% at nk = 24, 36, 48,
  72, 96, always at the top of the window, 2w = 3 eV; the velocity gauge itself is 39%, 9%, 6%
  off on the meshes 24, 36, 48, so the slow part is the resolution delta >= v dk/2 and not the
  gauge. C3 holds to 1e-13.

After the review of 26 October 2026 (the Lanczos bounds on the CPU and the device, the
compiled-once device kernels, the leaner pair basis and the regrouped position operator), on
the same workstation, tetralayer at nk=48, second call in the process, SHG yyy at 16
frequencies / absorption at 301 energies:

| route | before | after |
| --- | --- | --- |
| CPU | 24.1 / 16.0 s | 20.5 / 12.7 s |
| GPU double | 13.9 / 10.4 s | 10.0 / 6.6 s |
| GPU single | 6.7 / 5.3 s | 3.4 / 2.1 s |

The first call of a process pays the compilation, 4.6 / 3.1 s in single precision. The spectra
agree with the code before to 1e-12 in double precision, and single precision stays within 8e-6
(SHG) and 4e-5 (absorption) of the CPU. In single precision what is left is the recursion, 1.7 /
1.4 s, the position operator on the host, about 0.8 s, and the pair basis, 0.4 s (1.1 s before,
when it diagonalized k and k+Q separately at Q=0 and filled its arrays in a Python loop). On the
CPU of that workstation one Lanczos step costs about 30 ms against 9 ms for the same matvec
applied repeatedly to one vector; it depends on the content of the vector and not on the
allocation (fixed buffers changed nothing), and on the 14-thread workstation the same run takes
48 steps in 0.27 s at nk=36, so the cause was not found. With every component of the SHG, the
default, the tetralayer at nk=36 went from 34.2 s to 18.7 s on the 14-thread workstation, mostly
from the position operator, whose 81 applies per batch (one per term and triplet) had cost 17.1
s: they are now at most 6 forward and 18 backward transforms, made one source at a time so that
the memory per frequency does not grow.

One caveat on the tetralayer numbers. `density_interaction` cuts the Vr tail at `rcut=5.0` by
default, and the layers of this stack are 6.13, 6.25 and 6.36 apart, so the benchmark has no
interlayer electron-hole interaction at all: only 144 of the 576 orbital channels are non-zero,
the intralayer ones, and interlayer excitons are not bound. An `rcut` that reaches the next
layer is the physical setting for a heterostructure, and it multiplies the channels of the FFT
by up to four, so the timings above understate that case.

## Left open

The numbers of the first item were measured while planning, on the dense matrices of the
same models, before the implementation.

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
- **The convolution apply for the eigensolvers.** `bsetk/iterative.py` and `bsetk/qtt.py`
  still use the low-rank factorization, with its memory and its refusal of a tabulated W. At
  Q=0 `ConvolutionKernel` has the interface the iterative solver needs (npair, diag, matvec);
  finite Q, the antiresonant and the coupling blocks are not built, and the coupling block is
  what the non-Tamm-Dancoff item above needs.
- **The position operator on the device.** On the GPU path the SHG still brings the resolvent
  vectors back to the host for the position operator, a forward transform per source vector and
  a backward one per direction it is needed in, about 0.8 s for one component on the tetralayer
  at nk=48; together with the pair basis it is most of what is left on the host.
- **A channel-wise device matvec.** Gathering only the channels where W is non-zero, with the
  mesh on the last axes of the FFT and the scatter as a contraction with a one-hot matrix, gave
  2.16 ms against 2.67 ms per matvec in single precision on the tetralayer at nk=48, but 23.6 ms
  against 16.5 ms in double, so it was not adopted.
- **An exciton density of states** by a stochastic trace with the Jackson kernel, the analogue
  of `kpm.random_trace`, on the same engine. Not asked for.

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
