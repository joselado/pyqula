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
