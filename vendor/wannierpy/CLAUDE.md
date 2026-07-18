# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python bindings for [Wannier90](http://www.wannier.org)'s "library mode"
API -- the `wannier_setup`/`wannier_run` Fortran subroutines in
`src/wannier_lib.F90`, wrapped via a hand-written `f2py` signature file
(`wannier90.pyf`) into a compiled extension. This is a separate project
(its own git repo) from the `wannier90-3.1.0/` sibling directory it wraps;
that directory is the upstream Fortran source and is documented by its own
`CLAUDE.md`, not this one.

Two backends implement the same `setup()`/`run()` API, selected via
`backend="fortran"|"python"` (default `"fortran"`): the compiled extension
above, and `wannier90/_engine/`, a from-scratch pure-Python/NumPy
reimplementation of the same calculation with no Fortran dependency at
runtime -- see "Pure-Python backend" under Architecture for its status
(what's ported and validated vs. what still isn't).

## Build / install / test

```bash
pip install -e ".[test]"     # pure Python -- no compiler, no Wannier90 source needed
python -m pytest tests/ -v
```

That's the whole install for the pure-Python backend (`backend="python"`,
see below) -- this is a plain PyPI-style package now (`pyproject.toml`
only, no `setup.py`), and `pytest` without the Fortran extension built will
just skip the `backend="fortran"`-only tests (`test_gaas.py`) and the
fortran-vs-python parity assertions inside the `test_engine_*.py` files
that build both backends to compare (they still exercise the pure-Python
side on its own).

**To also build the Fortran backend** (optional, `backend="fortran"`;
requires a Fortran compiler and LAPACK/BLAS dev headers): run
`scripts/build_fortran_extension.py`, which looks for the Wannier90 3.1.0
source in order: the `WANNIER90_SRC` env var, `vendor/wannier90-3.1.0/`
inside this repo, or a `wannier90-3.1.0/` sibling directory (the layout used
in this dev checkout -- there's no vendored/pinned copy or git submodule
yet).

```bash
WANNIER90_SRC=/path/to/wannier90-3.1.0 python scripts/build_fortran_extension.py
```

It writes a `-fPIC -DEXIT_FLAG` `make.inc` there if none exists, runs `make
lib COMMS=serial`, then `f2py -c wannier90.pyf -lwannier -llapack -lblas`
with `LIBRARY_PATH` pointed at the built `libwannier.a` (see "f2py gotcha"
below), and copies the resulting `_wannier90*.so` into the installed
`wannier90/` package directory (so run `pip install -e .` first). This
script is *not* shipped in the PyPI sdist/wheel -- it only exists in a git
checkout, matching the decision to keep the PyPI release pure-Python-only
(see "Pure-Python backend" below for why: no source tree is available to
build against for a plain `pip install`, and vendoring Wannier90's GPL
Fortran source into the sdist was deliberately avoided).

To iterate on the extension alone without rerunning the whole script, that
same `f2py -c` command works standalone once `libwannier.a` exists
(`cd ../wannier90-3.1.0 && make lib COMMS=serial`).

**Verifying an install actually works** requires testing from outside this
source tree (e.g. a fresh venv + `cd /tmp && pytest .../tests/test_gaas.py`)
-- running pytest from here picks up the local `wannier90/` directory via
cwd on `sys.path` regardless of what's actually installed in site-packages,
which will hide a broken build.

`tests/test_gaas.py` is the golden test for the Fortran backend: it
reproduces the GaAs reference case from
`../wannier90-3.1.0/test-suite/library-mode-test/` against
`ref/results_ref.dat`, in three ways (`in_process` vs subprocess execution,
and file-based vs fully-in-memory `.win` construction) -- it's both the
correctness check and the executable spec for the whole API.
`tests/test_engine_*.py` are the pure-Python backend's tests (see "Pure-Python
backend" below) and `tests/conftest.py` holds shared GaAs fixtures for them.

## Architecture

Three layers, in `wannier90/`:

- **`_wannier90*.so`** -- the raw f2py extension. Its two functions,
  `wannier_setup`/`wannier_run`, mirror the Fortran subroutines' argument
  lists exactly as declared in `wannier90.pyf`, with redundant dimension
  arguments (`num_kpts_loc`, `num_bands_loc`, ...) hidden and inferred from
  array shapes. Never call this directly -- go through `api.py`.
- **`api.py`** -- the public `setup()`/`run()` functions, `SetupResult`/
  `RunResult` dataclasses, and the subprocess-isolation machinery
  (`_call`). This is where almost all the non-obvious behavior lives; read
  its module docstring before changing anything here.
- **`io_helpers.py`** -- optional convenience: parsers for the standard
  `.mmn`/`.amn`/`.eig` files a DFT interface produces, plus `write_win`
  (materializes a `.win` from Python data) and `reciprocal_lattice`. Nothing
  in `api.py` requires going through this module -- `M_matrix`/`A_matrix`/
  `eigenvalues` are just numpy arrays, however you build them.

### Pure-Python backend (`wannier90/_engine/`)

In addition to the Fortran extension, `wannier90/_engine/` is a from-scratch
pure-Python/NumPy reimplementation of the same library-mode calculation,
selected via `setup()`/`run()`'s `backend="fortran"|"python"` kwarg (or the
`WANNIER90_BACKEND` env var), default `"fortran"`. It mirrors
`wannier_lib.F90`'s call graph one-to-one for auditability:

- `kmesh.py` -- `kmesh_get` (b-vector shells/weights)
- `projections.py` -- projections-block parsing
- `overlap.py` -- `overlap_project` (Lowdin initial guess, no-disentanglement path)
- `disentangle.py` -- `dis_main` (disentanglement)
- `wannierise.py` -- `wann_main` (spread minimization)

**Design difference from the Fortran path, deliberate, not an oversight**:
every function here takes plain arguments and returns plain values -- no
`.win` file, no module-global state. `win_keywords`/`exclude_bands`/
`projections` are parsed in-memory (`params.py`) directly from the Python
objects `setup()`/`run()` already accept; hand-authored `.win` files on disk
are a `backend="fortran"`-only feature. Consequently the python backend
ignores `cwd`/`in_process` (both are Fortran-subprocess concerns) and needs
no subprocess isolation -- Python exceptions propagate normally, and there's
no equivalent of the Fortran replay-on-`run()` hazard described below
(`SetupResult._engine_state` carries context through directly instead).

Validated end-to-end against upstream's own reference data, not just
against the compiled extension -- see `tests/test_engine_*.py`, each
naming which upstream fixture it checks against:

| Fixture (referenced in place, not vendored) | Exercises |
|---|---|
| `library-mode-test/` (`ref/results_ref.dat`) | core engine: disentanglement, GaAs |
| `test-suite/tests/testw90_example05/` (diamond) | no-disentanglement path (`overlap_project`) |
| `test-suite/tests/testw90_example26/` (GaAs) | selective localization ("SLWF+C") |
| `test-suite/tests/testw90_precond_1/` (GaAs) | preconditioned CG (`precond`) |
| `test-suite/tests/testw90_na_chain_gamma/` (Na chain) | **negative** result -- see gamma_only below |
| `test-suite/tests/testw90_example21_As_sp/` (GaAs) | site symmetry (`lsitesymmetry`) -- self-consistency only, see above |

#### Implemented

Core: `kmesh_get`, projections parsing, `overlap_project`, `dis_main`
(disentanglement), `wann_main` (Wannierisation/CG spread minimization).

Beyond the core: `guiding_centres`/`wann_phases` (branch-cut fixing),
`kmesh_shell_fixed` (explicit `shell_list`), `select_projections`, spinor
projections (`(u)`/`(d)`/`[qaxis]`), selective localization with centre
constraints (`slwf_num`/`slwf_constrain`/`slwf_lambda`/`slwf_centres`,
"SLWF+C", Vitale et al. PRB 90, 165125), `fixed_step`, `precond`
(preconditioned CG via `ws_vectors.wigner_seitz_vectors` +
`wannierise._precond_direction`), and site symmetry (`lsitesymmetry`,
`_engine/sitesym.py`, R. Sakuma, PRB 87, 235109 (2013)) -- restricts
disentanglement/Wannierisation to irreducible k-points and reconstructs the
rest of the Brillouin zone via symmetry operations' representation
matrices, given an external `.dmn` file (`io_helpers.read_dmn`; not
generated by this package or by Wannier90 itself -- produced by an external
symmetry-detection tool, e.g. a `pw2wannier90` interface with spglib
support). Not supported together with frozen-window disentanglement
(matches upstream). `run(dmn=...)` is the public entry point (accepts a
`.dmn` path, a `DmnData`, or a pre-built `sitesym.SymmetryData`);
`backend="fortran"` doesn't need this argument at all -- it reads
`site_symmetry`/the `.dmn` file itself, same as any other `.win` keyword.
Validated in `tests/test_engine_sitesym.py` via trivial-symmetry equivalence
against the non-symmetric code paths (on real disentangling and
non-disentangling GaAs data) and exact D-matrix propagation
self-consistency on `test-suite/tests/testw90_example21_As_sp/GaAs.dmn`
(real, non-trivial symmetry) -- *not* against that fixture's own
`benchmark.out`, which appears to be a stale pairing with the checked-in
`.mmn`/`.amn` (see the test file's module docstring for the reasoning:
Omega_I is a mathematical invariant of the raw overlap data, confirmed
identical between the symmetric and non-symmetric code paths and uniform
across every symmetry star, yet still doesn't match the benchmark's
reported value -- inconsistent with a bug in this port without also
breaking those from-first-principles checks).

#### Not implemented

All of these raise `NotImplementedError` (never silently wrong):

- **`gamma_only=True`** -- explicitly rejected, not just unported, and
  worth reading the reasoning before touching it: feeding the general
  complex-arithmetic `dis_main`/`wann_main` the gamma-halved k-mesh
  `kmesh_get` already produces looks like it should reproduce the
  dedicated real-arithmetic `dis_extract_gamma`/`wann_main_gamma` routines
  (the halving/doubling trick is exactly what those routines rely on), but
  empirically does not -- checked against
  `test-suite/tests/testw90_na_chain_gamma`, it gives a wildly different,
  non-converged spread. Those routines constrain the unitary rotations to
  be real (time-reversal symmetry at Gamma), a genuinely different
  optimization problem, not a faster arithmetic path through the same one.
  Porting it needs a dedicated real-arithmetic engine (fresh
  `dis_extract_gamma`/`internal_find_u_gamma`/`wann_main_gamma`/
  `wann_omega_gamma` ports), comparable in scope to everything above
  combined. See the comment in `_engine.wannier_run` and
  `tests/test_engine_gamma_only.py`.
- `devel_flag=kmesh_degen` (b-vectors read from a file) -- inherently at
  odds with this backend's no-file-I/O design, likely to stay unported.
- `auto_projections`, `random`/partial-random projections.

#### Performance

Not yet JIT-compiled (plain NumPy) -- benchmarked at ~2x the Fortran
extension's wall time on GaAs (includes subprocess overhead on the Fortran
side), i.e. practical as-is for small systems; revisit with Numba if
profiling on a larger system shows it's needed.

### Why `setup()`/`run()` aren't a thin pass-through (read before touching `api.py` or `wannier90.pyf`)

Three non-obvious constraints, discovered empirically (none are documented
upstream), shape almost everything in `api.py`:

1. **Fatal Fortran errors kill the process, not raise an exception.**
   `io_error` (`src/io.F90`) calls `STOP`/`exit(1)` on any internal error.
   Linked in-process, that takes the whole Python interpreter down with it.
   `api.py` therefore runs each calculation in a `spawn`-ed subprocess by
   default (`in_process=True` opts out) and turns a dead worker into a
   normal `WannierError`.

2. **`wannier_run` silently requires `wannier_setup` to have run in the
   *same process* immediately before it** -- despite `wannier_run`'s own
   arguments looking self-sufficient, it depends on a module-global flag
   (`library_param_read_first_pass` in `src/parameters.F90`) that only
   `wannier_setup` initializes correctly. Skipping this reliably crashes
   with `SIGFPE` mid-disentanglement. This is why `run()` *requires* the
   `SetupResult` from its matching `setup()` call: in subprocess mode, it
   replays the exact original `wannier_setup` call (via
   `SetupResult._setup_args`) inside the same worker before calling
   `wannier_run`. If you add new arguments to `setup()`, make sure they
   still end up captured in `_setup_args`.

3. **`multiprocessing.Queue` deadlocks if you `join()` before draining
   it.** `wannier_run`'s output (U matrices etc.) can exceed the pipe
   buffer; `_call`'s polling `queue.get(timeout=...)` loop (checking
   `proc.is_alive()` between attempts) exists specifically to avoid both
   that deadlock and hanging forever when a worker dies without producing
   output. Don't "simplify" this back to `join()` then `get()`.

### f2py gotcha (relevant if you touch `wannier90.pyf` or the build)

`f2py -c`'s meson backend silently drops a bare `.a` path passed
positionally -- it's not a source type it recognizes, and it doesn't end up
in the meson.build it generates. Linking `libwannier.a` requires
`LIBRARY_PATH` (so gcc's own early `-L` set picks it up) plus `-lwannier`
on the command line, not a path argument -- `scripts/build_fortran_extension.py`'s
`_build_extension` depends on this; don't refactor it to pass the `.a` path
directly.

`wannier90.pyf` is hand-written rather than auto-cracked from
`wannier_lib.F90` because some `intent(out)` arrays are dimensioned by
`num_nnmax`, a `parameter` in `w90_parameters` rather than a dummy
argument -- `crackfortran` can't resolve that across module boundaries. Its
value (12) is hard-coded in the `.pyf`; if the vendored Wannier90 version
ever changes, re-check `src/parameters.F90`'s `num_nnmax` definition.

### Library mode only reads a subset of `.win`

In library mode, wannier90 reads and then *ignores* `mp_grid`, `num_bands`,
and the `unit_cell_cart`/`atoms_frac`/`kpoints` blocks in `.win` -- those
come from `setup()`'s own arguments instead (confirmed from
`src/parameters.F90`'s "Ignoring `<mp_grid>` in input file" branches, and
covered by `tests/test_gaas.py::test_gaas_fully_in_memory`). Only
`num_wann` is unconditionally required. `write_win`/`setup()`'s
`win_keywords` intentionally don't support those redundant blocks.
