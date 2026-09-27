---
name: wannierization
description: pyqula's Wannierization (wanniertk/), what h.get_wannier_hamiltonian() returns, the disentanglement window keywords and which combinations raise NotImplementedError, and the bundled pure-Python wannierpy port. Load this before working on anything under src/pyqula/wanniertk/, before changing or calling get_wannier_hamiltonian, and whenever a task mentions Wannier functions, Wannier90, disentanglement, frozen windows, num_wann, or building a smaller real-space Hamiltonian that reproduces a subset of bands.
---

# Wannierization

## What `get_wannier_hamiltonian` does

`h.get_wannier_hamiltonian(bands=[a,b], nk=...)`
(`src/pyqula/wanniertk/wannierize.py`) Wannierizes a fixed, contiguous range of `h`'s bands
-- 0-indexed, both ends inclusive, Wannierized jointly as one group -- and returns a new,
smaller multicell `Hamiltonian` whose real-space hoppings exactly reproduce that band
subspace on the wannierization k-mesh.

## The default initial guess

Without `trial_vectors=`, the minimization starts from `num_wann` of `h`'s own orbitals,
the ones the SCDM column selection (Damle, Lin and Ying, arXiv:1507.03354) picks for the
selected bands on the mesh (`_default_trial_vectors`), in ascending orbital order (the
identity for a full manifold). It is deterministic, so repeated calls give the same
Wannier functions. Do not go back to a random draw: for the gapped honeycomb valence band
it stopped in a local minimum (spread 2.5 instead of 0.32) in a quarter of the calls
(`tests/wannier/test_default_trial_vectors.py`). Disentanglement picks from the frozen
window instead (`_default_disentanglement_trial_vectors`), and Nambu Hamiltonians keep
their own defaults (identity for the full manifold, electron-hole-paired orbitals
otherwise).

## Where the hoppings sit

The mesh fixes each hopping only up to a translation by the `nk` supercell, so
`_mesh_to_real_space` puts them on the cells of the Wigner-Seitz cell of that supercell
(real lattice metric, `_wigner_seitz_cells`), each divided by its degeneracy `ndegen`:
Wannier90's `hamiltonian_wigner_seitz` construction, with the same cells as the bundled
port `wannierpy/_engine/ws_vectors.py` but found class by class, so it also covers a
skewed, anisotropic supercell (`nk=[12,2]` on the honeycomb lattice) where Wannier90's
two-supercell search, and the port, fail. The cell set is inversion symmetric, so
`H_wan(k)` is Hermitian at every k. Do not go back to a plain `fftfreq` box of cells: for
an even `nk` it holds `R=-nk/2` without `+nk/2`, and `H_wan(k)` off the mesh is not
Hermitian (`tests/wannier/test_wigner_seitz_hoppings.py`).

`wannier_functions` is different on purpose (`_mesh_to_wannier_functions`): a Wannier
function is a function, so every class of cells appears once (the `fftfreq` box), since
splitting an amplitude over degenerate images would break its normalization.
Wannier90's `use_ws_distance` refinement (the Wigner-Seitz test on `R + tau_n - tau_m`
per pair of Wannier functions) is not implemented; it only matters for several Wannier
functions centred far apart within the cell.

How far the interpolation between mesh points is converged depends on the Wannier
functions, not on these cells: a band group that comes close to the rest of the spectrum
somewhere (the low-energy BdG pair of a chain with a small gap to the next band) has
wide Wannier functions and needs a dense mesh, and its `wannier_spread_total` keeps
growing with `nk` until the mesh resolves that region.

## Disentanglement

Passing `num_wann=` smaller than the selected range, together with the `dis_win_min`,
`dis_win_max`, `dis_froz_min` and `dis_froz_max` window keywords, turns on
Souza-Marzari-Vanderbilt disentanglement, which the bundled port already implements.

Outside a frozen window the reproduction is of the *optimal subspace* rather than exact.
That is the correct behaviour and what the tests assert -- do not treat it as a bug to fix.

Disentanglement combined with `has_eh`, `symmetries=` or `auto_split_clusters` raises
`NotImplementedError` naming the combination.

## The backend

It is built on [wannierpy](https://github.com/joselado/wannierpy)'s pure-Python Wannier90
port, bundled directly in this repo at `src/pyqula/wanniertk/wannierpy/`. There is no
Fortran source and no compiled extension: the pure-Python backend needs neither, and its
only dependency is numpy, which pyqula already requires. `wannierize.py` imports it
normally, not as an optional backend.

## Where to look

- `examples/wannier/get_wannier_hamiltonian/main.py` -- a runnable demo
- `tests/wannier/` -- correctness tests, exact-reproduction checks against the original
  spectrum
