"""Vendored upstream test suite for `pyqula.qutecipytk`.

`src/pyqula/qutecipytk/` is a verbatim vendor of https://github.com/joselado/qutecipy
at commit 6df39e7 (MIT). These are that project's own 16 test files, copied here
so the port stays validated: if someone edits the vendored source, or re-vendors
from a newer upstream, one command says whether it still behaves.

The only change from upstream is the import rewrite
`qutecipy.` -> `pyqula.qutecipytk.`, applied mechanically. Keep it that way --
do not hand-edit assertions or add pyqula-specific cases here, or the next
re-vendor has to be reconciled by hand. pyqula's own tests for how it *uses*
the port (topology.chern(integration="qtci"), get_dm_qtci) live in
tests/topology/ and tests/scf/ instead.

Everything here is marked `slow` from this file rather than by decorating the
16 files individually, which would be 16 more diffs against upstream. The full
directory takes ~6.5 min, dominated by the tensor-cross-interpolation
convergence tests.

    python -m pytest tests/qutecipy          # run them
    python -m pytest tests -m "not slow"     # skip them (and the rest of the slow suite)
"""
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent


def pytest_collection_modifyitems(items):
    """Mark every test in this directory `slow`. The hook receives the items
    of the whole session, not only the ones of this directory, so the test
    path is checked: marking them all made `pytest tests -m "not slow"`
    deselect the whole suite."""
    for item in items:
        if HERE in pathlib.Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.slow)
