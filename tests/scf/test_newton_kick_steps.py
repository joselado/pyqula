import numpy as np
import pytest

pytest.importorskip("jax")

from pyqula import geometry
from pyqula.scftk import densitydensity_jax
from pyqula.scftk.densitydensity import Vinteraction
from pyqula.scftk.spinspin import VJinteraction

# The Newton solvers of the use_jax=True engine leave a stationary point of
# the merit with bursts of linear-mixing steps, whose length of 60 was
# tuned on one system. kick_steps exposes it: it has to reach the solver
# from both public routes, keep 60 as its default, and be refused by the
# numpy engine, which has no kicks.


def _spy(monkeypatch, name):
    """Record the kick_steps each call of densitydensity_jax.<name> gets"""
    seen = []
    original = getattr(densitydensity_jax, name)
    def spy(*args, **kwargs):
        seen.append(kwargs.get("kick_steps"))
        return original(*args, **kwargs)
    monkeypatch.setattr(densitydensity_jax, name, spy)
    return seen


def _chain():
    return geometry.chain().get_hamiltonian()


@pytest.mark.parametrize("solver,target", [("newton", "newton_solve"),
        ("newton_krylov", "newton_krylov_solve"), ("fsolve", "fsolve_solve")])
def test_vjinteraction_forwards_kick_steps(monkeypatch, solver, target):
    seen = _spy(monkeypatch, target)
    VJinteraction(_chain(), U=2.0, mu=0.0, nk=4, T=1e-2, use_jax=True,
            solver=solver, kick_steps=17)
    assert seen == [17]


def test_vinteraction_forwards_kick_steps(monkeypatch):
    seen = _spy(monkeypatch, "newton_solve")
    h = geometry.chain().get_hamiltonian(has_spin=False)
    Vinteraction(h, V1=1.0, mu=0.0, nk=4, T=1e-2, use_jax=True,
            solver="newton", kick_steps=23, verbose=0)
    assert seen == [23]


def test_the_default_is_still_sixty(monkeypatch):
    seen = _spy(monkeypatch, "newton_solve")
    VJinteraction(_chain(), U=2.0, mu=0.0, nk=4, T=1e-2, use_jax=True,
            solver="newton")
    assert seen == [60]


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_a_kick_steps_that_is_not_a_positive_integer_is_refused(bad):
    with pytest.raises(ValueError, match="kick_steps must be a positive"):
        VJinteraction(_chain(), U=2.0, mu=0.0, nk=4, T=1e-2, use_jax=True,
                solver="newton", kick_steps=bad)


def test_the_numpy_engine_refuses_kick_steps():
    with pytest.raises(NotImplementedError, match="kick_steps"):
        VJinteraction(_chain(), U=2.0, mu=0.0, nk=4, kick_steps=30)
