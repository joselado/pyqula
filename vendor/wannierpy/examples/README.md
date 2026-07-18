# Examples

Six self-contained scripts, each defining (or importing) a Bloch
Hamiltonian and computing the `M_matrix`/`A_matrix`/`eigenvalues` overlap
data `wannier90.run()` needs directly from it (no `.mmn`/`.amn`/`.eig`
files, no DFT interface) -- see `_tb_utils.py` for how, and why.

| Script | System |
|---|---|
| `0d_molecule.py` | 4-site open linear chain (no periodicity) |
| `1d_ssh_chain.py` | Su-Schrieffer-Heeger dimerized chain (2 orbitals/cell) |
| `2d_square_lattice.py` | Two-orbital checkerboard-like square lattice |
| `3d_cubic_lattice.py` | Two-orbital CsCl-like simple cubic lattice |
| `pyqula_ladder.py` | Lowest band of a two-leg ladder, Hamiltonian imported from [pyqula](https://github.com/joselado/pyqula) |
| `pyqula_ladder_both_bands.py` | Same ladder, both bands (contrast with the single-band case above) |

Run any of them directly:

```bash
python examples/1d_ssh_chain.py
```

(no install needed -- each script adds the repo root to `sys.path` itself
if `wannierpy` isn't already installed).

## What they show, and why the converged spread is exactly zero

Every example keeps `num_wann == num_bands` (every tight-binding orbital
becomes one Wannier function, no disentanglement). For a *complete,
untruncated* discrete tight-binding manifold like these, the maximally
localized Wannier functions are provably **exactly** the original
tight-binding sites, with **exactly zero** spread -- a real mathematical
fact (Omega_I is a gauge invariant of the raw overlap data alone, and a
full eigenbasis is always unitary), not a limitation of this package. The
1D/2D/3D examples still demonstrate real work: they seed the calculation
with deliberately imperfect (differently-sized Gaussian) trial orbitals,
print the spread *before* any CG minimisation, and let `wannier90.run()`
minimise it down to that exact answer -- see `_tb_utils.py`'s module
docstring for the full derivation, including why naive choices (trial
orbitals equal to the exact eigenbasis, or any other *fixed* trial matrix)
trivially -- and uninformatively -- give zero spread from the very first
iteration, with no minimisation happening at all.

`0d_molecule.py` is simpler (no trial-orbital tricks needed) but has its
own caveat: a single k-point has no Brillouin-zone variation for the
Wannier centre/spread formulas to extract *position* information from, so
while it correctly exercises the full pipeline, the reported centres don't
resolve individual site positions -- see its module docstring.

`pyqula_ladder.py` is different again: it pre-selects a genuine subset of
bands (the lowest of 2, via `build_overlaps`'s `band_indices` -- the
manual equivalent of Wannier90's `exclude_bands`) rather than keeping the
full manifold, which is *not* subject to the same "any fixed trial
trivially collapses" algebra -- and yet it *still* converges to exactly
zero spread, this time for a genuine physical reason (the ladder's
leg-exchange symmetry pins the lowest band's Bloch eigenvector to be
exactly k-independent) rather than an algebraic one. See its own module
docstring for the full explanation, including why the reported rung-axis
centre is always 0 regardless of the true (a)symmetry of the state -- a
ladder is only periodic along its length, so there's no b-vector across
the rungs for the centre formula to extract that information from.

`pyqula_ladder_both_bands.py` Wannierizes both of the ladder's bands
instead (`num_wann == num_bands == 2`) -- back to the "complete manifold"
case, so it uses the same Gaussian-trial-orbital, before/after pattern as
the 1D/2D/3D examples (and shares the same rung-axis centre caveat as
`pyqula_ladder.py`, for the same reason).

## Adapting these

To Wannierize your own Hamiltonian -- hard-coded or, like
`pyqula_ladder.py`, from another package's Bloch Hamiltonian generator:
write/obtain `hamiltonian_k(k_frac) -> Hermitian ndarray` in the "periodic
gauge" (hoppings enter only via `exp(i*2*pi*k.R)` for integer lattice
vectors `R`, no sub-cell position phases -- true of `pyqula`'s
`h.get_hk_gen()`, and of most other tight-binding packages' Bloch
generators too), then call `_tb_utils.build_overlaps` with it. Two
independent knobs there are worth knowing about:

- `band_indices` pre-selects a subset of bands at every k (the manual
  equivalent of Wannier90's `exclude_bands`) instead of keeping the full
  local Hilbert space -- unlike the full-manifold case, this genuinely can
  give non-trivial spreads for a fixed trial (`trial_vectors`), though it
  isn't guaranteed to (see `pyqula_ladder.py` for a case where it still
  comes out exactly zero, for a real physical -- not algebraic -- reason).
- If you want genuinely non-trivial spreads and don't already have a
  natural band subset to select, real disentanglement is the other route:
  make your Hamiltonian bigger than `num_wann` bands, with a genuinely
  entangled (not simply weakly/perturbatively coupled, and not a single
  k-point) coupling between the bands you keep and the ones you don't --
  see `_tb_utils.py`'s module docstring for why weaker attempts still tend
  to land on an exact zero-spread answer for simple/smooth toy models.
