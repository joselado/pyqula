---
name: new-feature
description: The completeness checklist for adding a user-facing feature to pyqula - where the implementation goes, what kind of test it needs, and the five documentation surfaces that must move with it. Use when adding a new method, formalism, observable, or Hamiltonian term, or when asked whether a feature is finished.
---

# Adding a feature to pyqula

A feature here is not done when the code runs. It is done when a stranger can
find it, a test pins what it promises, and nothing in the docs contradicts it.
This is the checklist for that. CLAUDE.md points here and no longer states these
rules itself, so this file is their only home.

## Before writing code

- **Is it a new formalism or an extension of one already here?** For anything
  genuinely new -- a method the codebase does not already implement -- check
  arXiv for a paper covering it and use that reference to guide the
  implementation, rather than general knowledge alone. Fetch the e-print TeX
  source, not the HTML: the HTML mangles prefactors and drops index
  conventions. Re-derive every prefactor yourself.
- **Is there a tested open-source implementation** under a compatible license?
  Benchmark against it, or mirror its structure, rather than writing the
  algorithm from scratch. (`wanniertk/wannierpy/` is the precedent: a bundled
  pure-Python port rather than a reimplementation.)
- **Grep `examples/` first.** There may already be a script doing most of it.
- **Check `future_development/README.md`** -- if a roadmap covers the area, it
  probably records a measurement or a dead end you would otherwise re-derive.

## Where the code goes

1. **Implementation in a `*tk/` subpackage**, or in the top-level module that
   composes one (`topology.py` over `topologytk/`, `scf.py` over `scftk/`).
   Non-trivial functionality does not live in `hamiltonians.py`.
2. **A one-line delegator on `Hamiltonian` or `Geometry`** if it is meant to be
   called as `h.get_something(...)`. The class is deliberately thin; the
   delegator does nothing but call into the module.
3. **Follow the mutate-and-return convention.** Methods that add terms modify
   in place *and* return `self`; callers `.copy()` before mutating. Do not
   quietly make a new method purely functional if its siblings are not.
4. **Guard arguments with a real message.** `ValueError` for a bad value or a
   Hamiltonian in the wrong Hilbert space, `NotImplementedError` for a
   combination not built yet, `TypeError` for a wrong type -- each naming the
   offending value. A string-selected option (`mode=`, `solver=`, `channel=`)
   must list the accepted values in the error. Never a bare `raise` outside the
   jump-to-except idiom.
5. **Parallelism:** prefer numba `@jit(parallel=True)`/`prange` over
   `parallel.pcall`'s process pool. The pool measured *slower* than serial on
   the KPM moment loop. Reach for `pcall` only when the work is not
   numba-jittable.

## The test

`tests/<topic>/test_*.py`, asserting a **physical or numerical invariant**, not
a recorded number. The suite's standard shapes:

- a result must not depend on something it physically cannot depend on (the
  random seed of an SCF initial guess, the choice of unit cell);
- two independent code paths computing the same quantity agree to tolerance;
- a symmetry or sum rule holds (a Goldstone mode for a spin response, a
  quantized invariant, a conserved current).

Traps that make a correct calculation look broken, or a broken one look
correct:

- **`sum(bands) == 0` and friends are vacuous** for a symmetric spectrum -- the
  assertion passes whatever the code does. Assert something that can fail.
- **Check nk-convergence before asserting a nonzero value.** A hung or wrong
  mean-field result is usually the k-mesh, not the mixing.
- **Test a generic direction**, never one axis. `full_dm` is the *transpose* of
  rho, and contracting it flips the sign of `sy`, valley and current operators
  -- an x-only test cannot see it.
- SCF failure is a `None` return, not an exception, and `maxite` defaults to
  `None` so a non-converging loop never returns.

## The five documentation surfaces

All of these, for a user-facing feature:

1. `documentation/user_guide.md` -- a prose section with the physics and
   motivation plus a runnable snippet, in the existing style.
2. `documentation/user_guide.md`, `# Main functions and methods` -- an entry,
   for anything with a method on `Hamiltonian` or `Geometry`.
3. `README.md`, the `# FUNCTIONALITIES #` list -- a bullet where relevant.
4. `examples/<dimensionality>/<name>/main.py` -- a runnable script. These
   double as usage documentation and are where the next person will grep.
5. `jupyter-notebooks/functionalities/` -- a notebook if the FUNCTIONALITIES
   bullet should link to one. The README's tutorial section states how many
   bullets currently do; if you add a notebook, that count moves with it.

Then run `python -m pytest tests/documentation` -- it statically checks that
every method the guide names actually exists -- and rebuild the PDF. The
`refresh-docs` skill does that sweep.

## If you deliberately leave something unbuilt

Write it up in `future_development/`, with what was measured and what the next
decision point is, and add it to that directory's `README.md` index. The point
is that picking the work up again does not mean re-deriving a conclusion
someone already reached.

## Never

Put cluster details -- hostnames, scratch paths, partitions, job IDs, queue
measurements -- into anything tracked. Performance *conclusions* belong in the
roadmaps; the machine that produced them does not.
