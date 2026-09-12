import os
import subprocess
import sys

import pytest

import pyqula


def _run(statement):
    """Import `statement` in a fresh interpreter, so that nothing else of
    pyqula has been imported first. The parent process has already imported
    superconductivity, which is exactly what hides the cycle."""
    env = dict(os.environ)
    src = os.path.dirname(os.path.dirname(os.path.abspath(pyqula.__file__)))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", statement],
                          capture_output=True, text=True, env=env)


@pytest.mark.parametrize("statement", [
    "from pyqula.sctk.extract import extract_triplet_pairing",
    "from pyqula.sctk.dvector import matrix2dvector",
    "from pyqula.sctk.pairing import pairing_generator",
    "from pyqula.sctk.superfluidweight import superfluid_weight",
    "import pyqula.superconductivity; "
    "from pyqula.sctk.extract import extract_triplet_pairing",
    ])
def test_sctk_modules_can_be_the_first_pyqula_import(statement):
    """Any module of the package must import on its own, in any order. The
    sctk helpers used to work only if pyqula.superconductivity happened to
    be imported first: superconductivity.py imports names from sctk.extract
    and sctk.dvector, while those two imported superconductivity at module
    level, so whichever side was reached first left the other executing
    against a half-initialised module."""
    r = _run(statement)
    assert r.returncode == 0, r.stderr
