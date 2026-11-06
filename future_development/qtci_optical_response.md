# QTCI and the triangle method for the optical conductivity

This note records where quantics tensor cross interpolation (QTCI) helps the
k-integral of an optical response and where it does not, with the
measurements that decided it. The outcome is in the code as
`integration="triangle"` and `integration="qtci"` of
`h.get_optical_conductivity()` (`conductivitytk/kubosimplex.py`, tests in
`tests/conductivity/test_optical_simplex.py`). What is left open is at the end.

All numbers are for the spinless honeycomb lattice with a sublattice
imbalance of 0.3 (gap 0.6 t), Re sigma_xx, T = 1e-3 t, and the error is
measured against the uniform mesh at nk = 8192 (67M diagonalizations),
which itself agrees with the converged triangle and qtci results to 1e-4 to
3e-4. Below that level the methods cannot be ranked with this reference.

## QTCI of sigma itself: a loss

The first idea was to interpolate the integrand of sigma(omega) on a
quantics grid of (kx, ky), one interpolation per frequency. The resonant
shell E_c - E_v = omega has width delta, so the integrand is sharp, and the
hope was that QTCI refines only near the shell.

- At delta = 1e-2 it loses: the uniform mesh at nk = 512 (262k points,
  error 1e-3) beats QTCI at R = 12 (378k evaluations, error 4.5e-3).
- At delta = 1e-3 it is comparable, not better: R = 12 at tolerance 1e-5
  gives 6e-4 with 365k diagonalizations, against 3e-3 for the uniform mesh
  at nk = 2048 (4.2M). The error is controlled by the tolerance (it comes
  out 20 to 70 times the requested value) and not by the rank cap (raising
  `maxbonddim` from 400 to 800 changes nothing, the rank saturates at 115
  for R = 10 and 256 for R = 12). Seeding the pivots on the resonant shell
  instead of at random points barely helps.
- Each frequency needs its own interpolation, while the uniform mesh gives
  all frequencies from one diagonalization, so for a spectrum this is a
  loss in every regime.
- Interpolating the joint function f(kx, ky, omega), with omega as a third
  quantics variable, is a clear loss: at Rk = 8, Rw = 6, delta = 1e-2 it
  diagonalized 73% of the k-points at rank 404, with an error of 9e-3.

## The triangle method: the real gain

The sharp part of the integrand is only the resolvent 1/(z - dE(k)); dE and
the weights are smooth. Interpolating dE linearly inside each triangle of
the mesh and integrating the resolvent exactly (Hermite-Genocchi, the 2D
analogue of the analytic tetrahedron method of Lambin and Vigneron, PRB 29,
3430 (1984)) gives every frequency from one mesh, at any delta:

| method | diagonalizations | max relative error, omega in [0.65, 3] t |
|---|---|---|
| uniform nk = 4096 | 16.8M | 4e-3 |
| triangle nk = 256 | 65k | 1e-3 |
| triangle nk = 1024 | 1.0M | 3e-4 |
| qtci R = 8 + triangle | 35k | 1e-3 |
| qtci R = 10 + triangle | 69k | 3e-4 |
| qtci R = 12 + triangle | 81k | 3e-4 |

The triangle method alone gives about a factor 1000 in diagonalizations
against the uniform mesh at equal error. QTCI of the smooth fields (dE and
the weights, one interpolation each, sharing one cache of diagonalizations)
adds a factor 2 at 1e-3 and about 15 at 3e-4, because the number of
diagonalizations saturates with R: ranks stay at about 43 for dE and 60 for
the weight. At delta = 1e-2 the triangle result at nk = 512 agrees with the
converged uniform mesh at nk = 2048 to 1e-4 at all six frequencies, which is
the independent check of the formula.

The triangle result converges at second order in 1/nk (the Hall
conductivity at omega = 0 has errors 4.8e-3, 1.2e-3, 5.4e-4 at nk = 30, 60,
90). For a smooth periodic integrand, a frequency inside the gap or a large
delta, the uniform mesh converges exponentially and is the better choice;
the docstring and the guide say so.

The weights are averaged over degenerate multiplets at each k before they
are interpolated. For exact degeneracies at every k (a spinful copy) this
changes nothing in the triangle sum, because the pairs of a multiplet share
the same dE and the sum is linear in the weights; it matters for an
interpolation in which a single w_nm would depend on the basis that eigh
picks. On the spinful Haldane model, switching the averaging off changed
the triangle result by 8e-13 and the qtci result by 7e-8, so on the models
tried the unaveraged fields were smooth as well (possibly because eigh
returns pure spin states for a spin-block Hamiltonian; not checked). Only
the direct basis-rotation test exercises the averaging.

## What is left open

- **Cost per frequency.** The kernel evaluates the triangle sum frequency by
  frequency (prange over frequencies). On a 1024 x 1024 mesh with 701
  frequencies this takes about three minutes for two bands, and it is the
  dominant cost of both paths. The fix is to build the weighted joint DOS
  J(E) once (the density of a linear function on a triangle is piecewise
  linear in E) and convolve it with the resolvent, which removes the
  dependence on the number of frequencies. Not built.
- **Many bands.** The fields are ordered by band index, so they have kinks
  where bands cross. The triangle sum still converges (the crossings are a
  set of measure zero), but the QTCI rank grows. Not measured beyond two
  bands.
- **Metals.** The occupation factors jump at the Fermi surface within a
  width T. The vertex-mean weight is first order there, and in the qtci path
  the step costs rank. The Drude term is the same in all three
  integrations. Linear vertex weights (a Blochl-type correction) would make
  the interband part second order; not built.
- **Second harmonic generation and the shift current** (`conductivitytk/shg.py`)
  have the same structure with 1/(z - dE)^2 and two resonant shells. The
  same Hermite-Genocchi construction works with one more antiderivative.
  This is the most useful next step, because SHG spectra are usually
  over-broadened to converge the k-sum.
- **3D.** The Kubo module has no 3D branch (`current.derivative` has none).
  With one, tetrahedra need the third divided difference of
  psi_3, and the gain over a uniform nk^3 mesh should be larger still.
