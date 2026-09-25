import numpy as np
import pytest

from pyqula import geometry
from pyqula.scftk.densitydensity import Vinteraction
from pyqula.scftk.densitydensity_kpm import Vinteraction_kpm

# Vinteraction_kpm used to swallow every keyword its exact sibling
# Vinteraction refuses, and ran the plain half-filled calculation instead:
# a misspelled filling, the keywords of the old selfconsistency interface
# (which ran with no interaction at all), and a local U on a spinless
# Hamiltonian. Each of these must now raise the same exception as
# Vinteraction on the identical call, and a solver that only Vinteraction
# has must be refused by the KPM loop, which only mixes.

CASES = [
    (dict(V1=1.0, filing=0.25), TypeError, "unexpected keyword"),
    (dict(V1=1.0, kernel="lorentz"), TypeError, "unexpected keyword"),
    (dict(g=1.0, mode="V"), TypeError, "old scftypes.selfconsistency"),
    (dict(V1=1.0, U=2.0), ValueError, "requires the spin degree of freedom"),
]


def _spinless_chain():
    return geometry.chain().get_supercell(2).get_hamiltonian(has_spin=False)


@pytest.mark.parametrize("kwargs,error,match", CASES)
def test_vinteraction_kpm_refuses_what_vinteraction_refuses(kwargs, error,
        match, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path) # a run that wrongly goes through saves MF.pkl
    h = _spinless_chain()
    common = dict(mf={(0,0,0): np.diag([0.3,-0.3]).astype(complex)}, nk=6,
            maxite=2, verbose=0, load_mf=False)
    for fun in (Vinteraction, Vinteraction_kpm):
        with pytest.raises(error, match=match):
            fun(h, **common, **kwargs)


@pytest.mark.parametrize("kwargs", [dict(solver="broyden_mixing"),
        dict(use_jax=True)])
def test_vinteraction_kpm_refuses_the_solvers_it_does_not_have(kwargs,
        monkeypatch, tmp_path):
    """Vinteraction has these solvers, the KPM loop only plain mixing"""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError, match="unexpected keyword"):
        Vinteraction_kpm(_spinless_chain(), V1=1.0, nk=6, maxite=2,
                verbose=0, load_mf=False, **kwargs)


def test_the_spinless_kpm_route_refuses_a_misspelled_keyword(monkeypatch,
        tmp_path):
    """a spinless get_mean_field_hamiltonian(integration="kpm") goes to
    Vinteraction_kpm, and used to return None here"""
    monkeypatch.chdir(tmp_path)
    h = _spinless_chain()
    with pytest.raises(TypeError, match="unexpected keyword"):
        h.get_mean_field_hamiltonian(integration="kpm", V1=1.0, filing=0.25,
                nk=6, maxite=2, verbose=0, load_mf=False)
