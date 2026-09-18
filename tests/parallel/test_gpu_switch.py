"""The package-wide CPU/GPU switch, pyqula.gpu.

It replaced the per-call kpm_cpugpu/chi_cpugpu arguments: one setting now
decides where every GPU-capable routine runs, and the jax modules with no
backend branch of their own follow it through jax's default device.
"""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from pyqula import gpu


def _fresh_interpreter(code):
    """Run code in a new interpreter, so that the state under test is the
    one a script gets on startup rather than whatever this session left"""
    import pyqula
    src = os.path.dirname(os.path.dirname(os.path.abspath(pyqula.__file__)))
    env = dict(os.environ, PYTHONPATH=src)
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_the_package_starts_on_the_cpu():
    """Even where a device exists: a result should not depend on which
    machine the script landed on"""
    assert _fresh_interpreter(
        "from pyqula import gpu; print(gpu.get_gpu())") == "False"


def test_setting_the_switch_moves_jax_and_comes_back():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    was = gpu.get_gpu()
    try:
        assert gpu.set_gpu(False) is False
        assert "Cpu" in str(jnp.ones(2).devices())
        gpu.set_gpu(True)
        assert gpu.get_gpu() is True
        expected = "Cuda" if gpu.is_gpu_available() else "Cpu"
        assert expected in str(jnp.ones(2).devices())
    finally:
        gpu.set_gpu(was)


def test_a_machine_without_a_gpu_warns_instead_of_failing():
    """A script written for a GPU machine must still run anywhere, but
    silently running on the CPU is the failure this switch exists to
    prevent, so it says so"""
    was, available = gpu.get_gpu(), gpu._available
    try:
        gpu._available = False  # pretend there is no device
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gpu.set_gpu(True)
        assert len(caught) == 1 and "CPU" in str(caught[0].message)
        assert gpu.get_gpu() is True  # the jax code paths still run
    finally:
        gpu._available = available
        gpu.set_gpu(was)


def test_a_backend_that_is_not_a_backend_is_refused():
    with pytest.raises(ValueError, match="True or False"):
        gpu.set_gpu("GPU")
    with pytest.raises(ValueError, match="'CPU' or 'GPU'"):
        gpu.get_device("cuda")


def test_the_removed_per_call_arguments_raise():
    """kpm_cpugpu/chi_cpugpu used to select the backend per call. They are
    swallowed by **kwargs, so without a guard a script passing them would
    quietly run on whichever backend the switch happened to hold"""
    from pyqula.kpmtk.kpmnumba import kpm_moments_batch
    m = np.diag([0.1, -0.2, 0.3])
    vs = np.eye(3)
    with pytest.raises(ValueError, match="set_gpu"):
        kpm_moments_batch(vs, m, n=4, kpm_cpugpu="GPU")
