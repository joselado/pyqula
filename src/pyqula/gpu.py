"""Global CPU/GPU switch for the whole package.

pyqula runs on the CPU unless it is asked not to:

    from pyqula import gpu
    gpu.set_gpu(True)     # every GPU-capable routine now uses the device
    ...
    gpu.set_gpu(False)    # back to the CPU

This is the single control. Routines with a GPU path (the KPM moments of
kpmtk/kpmjax.py, the RPA Lindhard kernel of chitk/chijax.py, the batched
dense diagonalization of htk/eigenvectors.py) consult get_gpu() and pick
their backend from it, and the jax modules with no CPU/GPU branch of their
own (scftk/densitydensity_jax.py, scftk/vjinteraction_jax.py,
graphenetk/relax.py, transporttk/kappa_jax.py, keldyshtk/current_jax.py,
fermisurfacetk/swarmfs.py, classicalspin.py, symmetrytk/localsymmetry.py)
follow it too, because set_gpu points jax's default device at the chosen
backend for the whole process.

set_gpu(True) on a machine with no GPU warns and runs the same jax code
paths on jax's CPU backend, so a script written for a GPU machine still
runs anywhere.

The default is the CPU even where a CUDA-built jax and a device are
present. That is deliberate: it keeps a result from depending on which
machine a script lands on, it keeps the CPU reference for every GPU path
one call away, and on a consumer card several of those paths are no faster
on the device anyway (see documentation/gpu_porting_plan.md).

Precision is a separate axis and stays a per-call argument (kpm_prec,
chi_prec), since it changes the numbers rather than where they are
computed.
"""

import warnings

_enabled = False # the CPU until asked otherwise
_available = None # cached, since probing imports and initializes jax


def is_gpu_available():
    """Whether jax can see a GPU on this machine"""
    global _available
    if _available is None:
        import jax
        try:
            jax.devices("gpu")
            _available = True
        except Exception:
            _available = False
    return _available


def get_gpu():
    """Whether the package is currently set to use the GPU"""
    return _enabled


def set_gpu(value=True):
    """Run every GPU-capable routine on the GPU (value=True) or on the CPU
    (value=False), and return what was set.

    With no GPU on the machine this warns and keeps the jax code paths,
    which jax then runs on its own CPU backend: the same numbers, none of
    the speedup. It warns rather than failing so that a script written for
    a GPU machine still runs anywhere, and rather than staying silent
    because a run that was meant to use the device and quietly did not is
    the failure this switch exists to prevent"""
    global _enabled
    if value is not True and value is not False:
        raise ValueError("set_gpu takes True or False, got "+str(value))
    if value and not is_gpu_available():
        warnings.warn("pyqula.gpu.set_gpu(True): jax sees no GPU on this "
                "machine, so the GPU code paths will run on jax's CPU "
                "backend instead")
    _enabled = value
    apply()
    return _enabled


def get_device(where=None):
    """The jax device for where ("CPU"/"GPU", default the current switch).
    Asking for the GPU where there is none gives the CPU device, matching
    set_gpu's fallback"""
    import jax
    if where is None: where = "GPU" if _enabled else "CPU"
    if where not in ("CPU","GPU"):
        raise ValueError("the backend must be 'CPU' or 'GPU', got "+str(where))
    if where=="GPU" and not is_gpu_available(): where = "CPU"
    return jax.devices(where.lower())[0]


def apply():
    """Point jax's default device at the selected backend, which is what
    carries the switch into the jax modules that have no backend argument
    of their own. Called by set_gpu, and at import time by those modules,
    so that importing one does not leave placement to jax's own default"""
    import jax
    jax.config.update("jax_default_device",get_device())


def check_removed_arguments(kwargs,name):
    """Raise for the per-call backend arguments this switch replaced, so
    that code passing them fails instead of silently running on the CPU"""
    if name in kwargs:
        raise ValueError(name+" was removed; the backend is now a global "
                "switch, use pyqula.gpu.set_gpu(True) instead")
