"""Subprocess entry point used to isolate calls into the Fortran library.

wannier90's internal error handler (``io_error`` in src/io.F90) calls
Fortran ``STOP``/``exit(1)`` on any fatal problem -- linked in-process that
kills the whole Python interpreter with no exception raised. wannier_setup/
wannier_run also mutate module-global Fortran state and are not designed to
be called more than once per process. Running each call in a fresh
``spawn``-context subprocess sidesteps both problems: a Fortran abort only
kills the worker, and each worker is used for exactly one call.
"""
from __future__ import annotations

import os


def run_in_worker(queue, func_name: str, args: tuple, cwd: str, setup_args=None) -> None:
    """Target function for the worker process. Must stay importable at
    module level (not a closure/lambda) so ``multiprocessing`` can pickle
    it under the ``spawn`` start method.

    If ``setup_args`` is given, ``wannier_setup(*setup_args)`` is called
    first and its output discarded -- ``wannier_run`` depends on module
    state that only ``wannier_setup`` initializes, even though its own
    arguments are self-sufficient on paper (see api.py's module docstring).
    """
    os.chdir(cwd)
    from . import _wannier90

    try:
        if setup_args is not None:
            _wannier90.wannier_setup(*setup_args)
        result = getattr(_wannier90, func_name)(*args)
    except Exception as exc:  # noqa: BLE001 - relayed to the parent as-is
        queue.put(("error", repr(exc)))
        return
    queue.put(("ok", result))
