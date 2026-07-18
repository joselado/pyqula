# wannierpy

Compute maximally-localised Wannier functions from a DFT (or other) code's
overlap/projection/eigenvalue output -- the same calculation
[Wannier90](http://www.wannier.org)'s "library mode"
(`wannier_setup`/`wannier_run`) provides, in pure Python.

`pip install wannierpy` gets you a pure-Python/NumPy implementation with no
compiled dependencies -- no Fortran compiler, no Wannier90 source tree
required. An optional Fortran backend (linking Wannier90's own
`libwannier.a` via a compiled extension) is also available for anyone who
wants to cross-check against, or fall back to, the reference implementation;
see "Optional: Fortran backend" below.

This package does **not** compute overlaps/projections/eigenvalues itself
-- those (`.mmn`/`.amn`/`.eig` files) still come from a DFT code's
`pw2wannier90`-style interface. `wannier90.io_helpers` parses those standard
Wannier90 file formats into the arrays `wannier90.run()` expects -- or build
those arrays yourself and skip files entirely, see "Everything through
memory" below.

## Installing

```bash
pip install wannierpy
```

The only dependency is NumPy. No compiler, no external Wannier90 source
tree -- this installs the pure-Python backend, described below.

## Usage

```python
import numpy as np
import wannier90
from wannier90 import io_helpers

# 1. wannier_setup: algorithmic parameters (num_wann, projections,
#    exclude_bands, disentanglement window, ...) are passed directly as
#    Python objects -- nothing is read from disk.
setup_result = wannier90.setup(
    "gaas", mp_grid, kpt_latt, real_lattice, num_bands_tot,
    atom_symbols, atoms_cart, gamma_only=False, spinors=False,
    win_keywords={
        "num_wann": 8, "num_iter": 1000, "conv_tol": 1e-10,
        "dis_win_max": 24.0, "dis_froz_max": 14.0, "dis_num_iter": 1200,
    },
    exclude_bands=range(1, 6),  # or a pre-formatted string, e.g. "1-5"
    projections=["f=0.25,0.25,0.25 : s", "f=0.25,0.25,0.25 : p"],
    backend="python",
)

# 2. Parse the DFT interface's overlap/projection/eigenvalue files. nnlist/
#    nncell (from setup_result) are needed to interpret the .mmn file.
M_matrix = io_helpers.read_mmn("run_dir/gaas.mmn", setup_result.num_bands,
                                num_kpts, setup_result.nntot,
                                setup_result.nnlist, setup_result.nncell)
A_matrix = io_helpers.read_amn("run_dir/gaas.amn", setup_result.num_bands,
                                num_kpts, setup_result.num_wann)
eigenvalues = io_helpers.read_eig("run_dir/gaas.eig", setup_result.num_bands, num_kpts)

# 3. wannier_run: must be passed the SetupResult from step 1.
run_result = wannier90.run(
    "gaas", setup_result, mp_grid, kpt_latt, real_lattice,
    atom_symbols, atoms_cart, M_matrix, A_matrix, eigenvalues,
    backend="python",
)
print(run_result.wann_centres, run_result.wann_spreads)
```

`real_lattice`/`atoms_cart` are in Angstrom; `kpt_latt` is in fractional
(reciprocal-lattice) coordinates, shape `(3, num_kpts)`. See
`tests/test_engine_run.py` (pure Python) or `tests/test_gaas.py` (Fortran)
for complete runnable examples using the GaAs case shipped with upstream
Wannier90, or `examples/` for four self-contained, hard-coded
tight-binding models (0D/1D/2D/3D) that build `M_matrix`/`A_matrix`
directly from a Bloch Hamiltonian instead of reading `.mmn`/`.amn` files.

## Everything through memory

Nothing in this API requires touching disk. `M_matrix`/`A_matrix`/
`eigenvalues` are just numpy arrays -- `io_helpers.read_mmn`/`read_amn`/
`read_eig` are one way to build them (from files a DFT interface wrote),
not the only way; build them yourself (e.g. from overlaps you computed
directly in Python) and pass them to `run()` exactly the same way.

`num_wann` is the one keyword always required in `win_keywords`; `mp_grid`,
`num_bands`, and the cell/atoms/k-points blocks aren't accepted there
because they're redundant with `setup()`'s own arguments (and Wannier90's
own library mode reads-and-ignores them too, for the same reason -- see
`write_win`'s docstring for the Fortran-backend equivalent).

## Status: what's implemented

The pure-Python backend covers the core calculation (k-mesh determination,
projections parsing, disentanglement, spread-functional minimisation) and
most non-default Wannier90 options: `guiding_centres`, explicit
`shell_list`, `select_projections`, spinor projections, selective
localization with centre constraints (`slwf_num`/`slwf_constrain`,
"SLWF+C"), `fixed_step`, `precond`, and site symmetry (`lsitesymmetry`, via
`run(dmn=...)` and a new `io_helpers.read_dmn`). All of it is validated
against upstream Wannier90's own reference data, not just internal
consistency -- see `CLAUDE.md`'s "Pure-Python backend" section for the full
list and exact test fixtures.

**Not yet implemented** (raises `NotImplementedError`, never a silently
wrong answer): `gamma_only=True` (needs a dedicated real-arithmetic engine
-- confirmed by testing that reusing the general complex-arithmetic code
does *not* just run slower, it gives wrong answers). See `CLAUDE.md` for
the detailed reasoning.

**Performance**: plain NumPy, not yet JIT-compiled -- benchmarked at ~2x
the Fortran extension's wall time on a small test case, i.e. practical for
small-to-moderate systems as-is.

## Optional: Fortran backend

Pass `backend="fortran"` (or set `WANNIER90_BACKEND=fortran`) to use a
compiled extension that links Wannier90's own `libwannier.a` instead --
useful for cross-checking results, or if you specifically need a code path
that exactly matches upstream Wannier90's Fortran numerics. This is *not*
part of the `pip install wannierpy` experience: building it requires

* a git clone of this repository (the build script isn't included in the
  PyPI sdist/wheel),
* a Fortran compiler and LAPACK/BLAS development headers (e.g.
  `apt install gfortran libblas-dev liblapack-dev` on Debian/Ubuntu),
* a local Wannier90 3.1.0 source tree.

```bash
git clone <this-repo>
cd wannierpy
pip install -e .
WANNIER90_SRC=/path/to/wannier90-3.1.0 python scripts/build_fortran_extension.py
```

The script looks for the Wannier90 3.1.0 source, in order: the
`WANNIER90_SRC` environment variable, `vendor/wannier90-3.1.0/` inside this
package, or a `wannier90-3.1.0/` sibling directory next to it (the layout
used while developing this package against a local Wannier90 checkout).

Once built, `backend="fortran"` and `backend="python"` are drop-in
compatible at the API level (same functions, same `SetupResult`/
`RunResult` shape) -- see `tests/test_gaas.py` for the Fortran-backend
golden test.

### Process isolation (read this before setting `in_process=True` on the Fortran backend)

By default, every Fortran-backend `wannier90.setup()`/`wannier90.run()`
call runs in a fresh subprocess. This isn't just a safety nicety -- two
real constraints in the underlying Fortran library make it necessary (the
pure-Python backend has neither issue, and never needs subprocess
isolation):

1. **Fatal errors call `STOP`, not an exception.** Wannier90's error handler
   (`io_error` in `src/io.F90`) calls Fortran `STOP` (or `exit(1)`, since
   the build sets `-DEXIT_FLAG`) on any internal error -- bad input,
   mismatched shapes, a singular matrix, a missing `.win` file. Linked
   in-process, that kills the whole Python interpreter with no exception
   raised. Isolating each call in a subprocess turns that into a normal
   `wannier90.WannierError`, with the tail of `<seedname>.wout` attached for
   context.

2. **`wannier_run` silently depends on `wannier_setup` having run in the
   *same process* just before it.** This was found empirically while
   building this package (it isn't documented upstream): despite
   `wannier_run` taking what looks like a self-sufficient set of arguments,
   it relies on a module-global flag (`library_param_read_first_pass` in
   `src/parameters.F90`) that only `wannier_setup` initializes correctly.
   Calling `wannier_run` alone in a fresh process reliably crashes with
   `SIGFPE` partway through disentanglement. Because of this,
   `wannier90.run(backend="fortran")` requires the `SetupResult` from its
   matching `setup()` call and, in subprocess mode, replays that exact
   `wannier_setup` call (silently, output discarded) in the worker before
   calling `wannier_run`.

Pass `in_process=True` to skip the subprocess and call the extension
directly -- only do this if you have your own process isolation (e.g. one
calculation per worker in a larger batch job already), and always call
`setup()` then `run()` in that same process in order, matching the only
usage pattern Wannier90 upstream actually tests
(`test-suite/library-mode-test/test_library.F90`).

Because of constraint 2, and because `wannier_setup`/`wannier_run` write
into module-global Fortran state, don't call `setup()`/`run()` more than
once (i.e. more than one calculation) in the same process, in-process mode
or not, with `backend="fortran"`.

### Serial only (Fortran backend)

Library mode is documented upstream as serial-only: even a `libwannier`
built with `COMMS=mpi` must be called from a single MPI rank. This package
always builds the serial variant and has no `mpi4py` integration. The
pure-Python backend has no MPI dependency either way.

### Library mode only reads a subset of `.win` (Fortran backend)

In library mode, wannier90 reads and then *ignores* `mp_grid`, `num_bands`,
and the `unit_cell_cart`/`atoms_frac`/`kpoints` blocks in `.win` -- those
come from `setup()`'s own arguments instead. Only `num_wann` is
unconditionally required. If you hand-author a `.win` file for the Fortran
backend (rather than passing `win_keywords`/`exclude_bands`/`projections`
directly, which works for both backends), those blocks are pointless to
include.

## License

Wannier90 is GPLv2, with no linking exception. The Fortran backend links
`libwannier.a` into a compiled extension, making that combined work GPLv2.
The pure-Python backend is a from-scratch reimplementation written by
closely following the Fortran algorithm rather than machine-translating it,
but out of caution it's released under the same GPLv2 terms as the rest of
this package rather than a more permissive license.
