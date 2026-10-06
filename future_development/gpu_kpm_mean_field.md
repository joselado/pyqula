# The KPM mean field on the GPU

Status, 5 October 2026: **built**, after the maintainer's sign-off. The
KPM mean field (`h.get_mean_field_hamiltonian_kpm`, and
`integration="kpm"` of `h.get_mean_field_hamiltonian`) did not run on the
device in any useful sense, a prototype of the shape that would was timed
on a consumer card, and that shape is now the engine on both backends:
`kpmtk/pairmomentsjax.py`, called by
`kpmtk/densitymatrix_kpm.py::_dm_kpm_from_needed` for the density matrix
and by `_kpm_dos_moments` for the Fermi search when the switch is set,
with `kpm_prec` (single on the GPU, double on the CPU by default) threaded
through `Vinteraction_kpm` and `VJinteraction`. The first sections below
are the measurements that led to it, the last one what the port measures
and what it leaves open.

## What runs where today

A KPM mean-field iteration has two costs: the density-matrix entries the
interaction reads, `kpmtk/densitymatrix_kpm.py::_dm_kpm_from_needed`, and
the Fermi search, `get_fermi4filling_kpm`, whose density of states comes
from `kpm.full_trace`. Only the second consults `gpu.get_gpu()`, through
`kpm_moments_batch`. The first runs one numba Chebyshev recursion per
canonical pair $(i,j)$ and per k-point, in complex128, in a Python loop,
so `gpu.set_gpu(True)` leaves the dominant cost exactly where it was.

The measurement is a honeycomb island (`islands.get_geometry(name="honeycomb",
n=..., nedges=6)`) at `nk=1`, with Rashba coupling 0.2, a spinful onsite $U$
(the non-collinear case) or a Nambu Hamiltonian with attractive $U$ and an
in-plane field (the superconducting case), `npol=200`. The machine is a
consumer RTX A2000 (12 GB, float64 at about 1/70 of float32) with the six
performance cores of an i5-12600K as the CPU side. Per SCF iteration, warm:

| case | orbitals | ED | KPM density matrix | Fermi search, CPU | Fermi search, `set_gpu(True)` |
|---|---|---|---|---|---|
| Hubbard | 768 | 1.2 s | 5.4 s | 0.25 s | 0.46 s |
| Hubbard | 1728 | 4.1 s | 27.2 s | 1.28 s | 1.68 s |
| Hubbard | 3072 | see below | 87 to 89 s | 4.1 s | 4.7 s |
| BdG | 1536 | 2.8 s | 25.7 s | 0.27 s | 0.49 s |
| BdG | 3456 | 35.6 s | 134 to 143 s | 1.31 s | 1.70 s |

The density matrix grows as $N^2$ (number of pairs times the cost of a
matvec) and ED as $N^3$, so with the present engine the crossover sits
near $10^4$ orbitals. The ED time of the 3072-orbital Hubbard island read
127 s per iteration, out of line with the larger BdG island, and is not to
be read as a crossover.

The existing batched GPU kernel (`kpmjax.kpm_moments_batch_gpu`, the one
the Fermi search reaches) is no faster than numba on this card in either
precision, which is the second half of the picture. `kpm.full_trace` at
`npol=200`, warm:

| orbitals | numba, 12 threads | card, complex128 | card, complex64 |
|---|---|---|---|
| 192 | 0.016 s | 0.023 s | 0.022 s |
| 768 | 0.25 s | 0.28 s | 0.19 s |
| 1728 | 1.12 s | 1.31 s | 0.97 s |

## The block recursion

The per-pair loop repeats work: the Hubbard interaction needs three pairs
per site from two starting columns, and the batched kernel runs all 1728
starting vectors of the 1728-orbital island through the same 400 steps in
1.1 s while the pair loop takes 27 s. The way we remove this is by running
one recursion on a dense block $V$ whose columns are every distinct starting
column $e_j$, $V_{n+1}=2HV_n-V_{n-1}$, and reading at each step the entries
$(T_n(H))_{ij}$ of every needed pair with one gather. The energy grid, the
Jackson basis, the Fermi weights and the assembly stay as they are. In
complex128 this reproduces the library's density matrix to 2e-16.

How the sparse product is done decides everything on the card. jax's
default lowering of a `BCOO` matrix times a dense block is a gather and a
scatter-add, whose atomics cost the same in either precision, and the
cuSPARSE lowering flag (`jax_bcoo_cusparse_lowering`) changes nothing
inside the loop. A tight-binding $H$ has a few entries per row, so the
GPU-friendly form is ELL: the entries of row $r$ padded to a fixed width $K$,
and $HV$ a sum over the $K$ slots of whole-row gathers, with no atomics.
One density-matrix evaluation, warm, against the library's numba engine:

| case | orbitals | library | card, BCOO, c64 / c128 | card, ELL, c64 | card, ELL, c128 | jax CPU, ELL, c64 / c128 |
|---|---|---|---|---|---|---|
| Hubbard | 768 | 5.4 s | 0.54 / 0.47 s | 0.17 s | 0.39 s | |
| Hubbard | 1728 | 27.2 s | 2.56 / 2.24 s | 0.46 s | 1.63 s | 2.45 / 4.97 s |
| Hubbard | 3072 | 87 s | 7.4 / 7.1 s | 1.65 s | 5.16 s | |
| BdG | 1536 | 22.4 s | 2.1 / 1.7 s | 0.34 s | 1.28 s | |
| BdG | 3456 | 134 s | 10.5 / 8.3 s | 1.48 s | 6.10 s | |

So the block layout alone gives 10 to 16x, ELL in single precision on the
card gives 50 to 90x over the library at 1500 to 3500 orbitals, and the
same ELL kernel on jax's CPU backend gives 5 to 11x with no device at all.
Single precision pays on the card only once the product stops being
atomics-bound, which is why it bought nothing in the BCOO column.

The kernel, as timed (`data` and `cols` of shape $(N,K)$, `V0` the block of
starting columns, `rows` and `cidx` the row and the block column of every
needed pair, `nm=2*npol`):

```python
@partial(jax.jit, static_argnums=(5,))
def _recursion_ell(data, cols, V0, rows, cidx, nm):
    def hv(a):
        out = data[:, 0, None]*a[cols[:, 0]]
        for s in range(1, cols.shape[1]): out = out + data[:, s, None]*a[cols[:, s]]
        return out
    V1 = hv(V0)
    mus = jnp.zeros((nm, rows.shape[0]), V0.dtype)
    mus = mus.at[0].set(V0[rows, cidx]).at[1].set(V1[rows, cidx])
    def body(i, c):
        am, a, mus = c
        ap = 2*hv(a) - am
        return a, ap, mus.at[i].set(ap[rows, cidx])
    _, _, mus = jax.lax.fori_loop(2, nm, body, (V0, V1, mus))
    return mus
```

with the ELL arrays built from the CSR form of $H(k)/s$ by padding each row
with zero entries pointing at the row itself. The library's pair $(i,j)$
holds $\langle e_i|T_n(H)|e_j\rangle$, meaning that the starting column is
$j$ and the projection is on $i$; the prototype's `cidx` follows that.

## Single precision

Single precision costs nothing measurable here. On frozen self-consistent
Hamiltonians (the 120 degree state of the triangular Hubbard model, a dilute
s-wave superconductor, a Rashba superconductor in an in-plane field) the
complex64 density matrix differs from complex128 by 1e-9 to 1e-8 at
`npol` from 200 to 1200, against a KPM truncation error of 2e-6 to 2e-2.
Along full SCF runs the two precisions take the same number of iterations
and agree to every printed digit in all four 2D cases (moments, gaps,
pairing, the d-vector non-unitarity of a triplet state), and on the card
the islands agree to 8e-8 in the moment and 4e-8 in the pairing. This was
checked up to `npol=1200` on frozen Hamiltonians and `npol=600` in an SCF,
not beyond.

One thing a single-precision path has to carry: the scale guard. A
complex64 recursion drifts by about $n\epsilon$, and a moment of a state
near zero energy, where $|T_{2n}(0)|=1$, then exceeds one by more than the
double-precision tolerance of `kpmtk/scaleguard.py`. That file already
holds a single-precision tolerance, but `_check_scale_covers_spectrum`
calls `moments_within_bound` without `kpm_prec`, so a port has to pass it
through, or the dilute s-wave case stops with "the Chebyshev moments of
H(k) diverge".

## What was built, and what is left open

The engine is the ELL kernel above, with three changes from the prototype.
Every k-point goes in one call (a `vmap` over k), with the ELL pattern
taken as the union over the mesh, since an entry whose Bloch sum cancels
at one k drops out of the CSR form of $H(k)$ and would otherwise change the
width, and with it the compiled kernel, from one k to the next. The energy
integral against the Fermi weights is contracted with the moments on the
device, in double precision whatever the precision of the recursion, so
that only one number per pair and k comes back. And the starting columns
and the k-points are split into calls of at most `_MAX_BLOCK` entries,
three blocks of which are live at a time. The scale guard receives
`kpm_prec`. In double precision the engine reproduces the per-pair numba
one to 1e-16, and its trace reproduces `kpm.full_trace` to 1e-15
(`tests/scf/test_kpm_block_density_matrix.py`).

Per SCF iteration on the same islands, density matrix plus Fermi search,
warm (the Fermi search on the CPU is still numba's `full_trace`):

| case | orbitals | per-pair engine | CPU, double | card, single | card, double | ED |
|---|---|---|---|---|---|---|
| Hubbard | 768 | 5.6 s | 1.1 s | 0.27 s | 0.87 s | 1.2 s |
| Hubbard | 1728 | 28.5 s | 6.2 s | 0.82 s | 3.3 s | 4.1 s |
| Hubbard | 3072 | 92.8 s | 18.0 s | 2.8 s | 10 s | |
| BdG | 1536 | 26.0 s | 4.2 s | 0.50 s | 1.8 s | 2.8 s |
| BdG | 3456 | about 140 s | 20.1 s | 1.9 s | 8.5 s | 35.6 s |

So the CPU gains 5 to 7x and the card 20 to 75x over the per-pair engine,
and the card is faster than exact diagonalization at every island measured,
from 768 orbitals, while the CPU overtakes it between two and three and a
half thousand. For a small cell on a k-mesh (the 120 degree triangular cell,
6 orbitals on 144 k-points) an iteration is about 0.08 s on either backend,
limited by dispatch rather than arithmetic.

Left open:

- The moment doubling that numba's batched kernel uses,
  $T_{2n}=2T_n^2-T_0$ and its odd partner, would halve the recursion for
  the pairs whose two indices are both starting columns, which for an
  onsite interaction is all of them, at the cost of inner products between
  block columns. Not tried.
- The jax CPU backend takes its thread count from XLA, so
  `parallel.set_enabled(False)` does not make this engine serial.
- Past roughly $10^4$ orbitals the starting columns are split into several
  calls; that path is tested at small size with a forced budget, not timed
  at scale.
- A data-centre card, where float64 is not 1/70 of float32, was not tried.
- `h.get_mean_field_hamiltonian(integration="kpm")` still refuses a Nambu
  Hamiltonian, which only `h.get_mean_field_hamiltonian_kpm` accepts.
