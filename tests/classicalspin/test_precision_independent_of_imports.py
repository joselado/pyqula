import os
import subprocess
import sys
from pathlib import Path

import numpy as np


_SRC = str(Path(__file__).resolve().parents[2]/"src")

_SCRIPT = """
import sys
if sys.argv[1]=="1": from pyqula.scftk import densitydensity_jax
import numpy as np
import jax
from pyqula import geometry, classicalspin
g = geometry.triangular_lattice().get_supercell(3)
sm = classicalspin.SpinModel(g); sm.add_heisenberg(Jij=[1.0])
np.random.seed(1); sm.minimize_energy(tries=3,silent=True)
print(int(jax.config.jax_enable_x64), repr(sm.energy()),
      *[repr(float(x)) for x in sm.magnetization[0]])
"""


def _run(first_import):
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC+os.pathsep+env.get("PYTHONPATH","")
    out = subprocess.run([sys.executable,"-c",_SCRIPT,first_import],
            env=env,capture_output=True,text=True,check=True)
    return np.array([float(x) for x in out.stdout.split()[-5:]])


def test_spin_minimizer_does_not_depend_on_what_was_imported_first():
    """jax double precision used to be switched on as a side effect of
    importing one of eleven jax modules, so the classical spin minimizer ran
    in float32 on its own and in float64 after one of them, and landed on a
    different texture of the degenerate 120-degree state. It is now switched
    on by pyqula/gpu.py, which every jax module imports"""
    alone = _run("0")
    after = _run("1")
    assert alone[0] == 1 and after[0] == 1 # double precision both times
    assert np.array_equal(alone, after)
    assert abs(alone[1]/9 + 3.) < 1e-9 # -3 J per site, in double precision
