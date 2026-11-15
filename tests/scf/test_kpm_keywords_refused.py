"""The keywords of the Chebyshev expansion (scale, npol, ne, cores,
kpm_prec, kpm_radius) tune integration="kpm" only. Exact diagonalization
and the states of an ultracell used to take them and ignore them, so
h.get_mean_field_hamiltonian(U=2., npol=400) ran the exact loop and nothing
told the caller that npol had no effect. They now raise a TypeError, the
way the jax engine refuses them, and a call that leaves them at None, their
default, is not refused."""
import pytest

from pyqula import geometry
from pyqula.scftk.densitydensity_kpm import KPM_KEYWORDS

VALUES = dict(scale=5.0, npol=400, ne=100, cores=1, kpm_prec="double",
        kpm_radius=3)


def _match(name):
    return r"keyword\(s\) %s tune" % name


@pytest.mark.parametrize("name", KPM_KEYWORDS)
def test_exact_diagonalization_refuses_the_kpm_keywords(name):
    h = geometry.chain().get_hamiltonian() # spinful, VJinteraction
    with pytest.raises(TypeError, match=_match(name)):
        h.get_mean_field_hamiltonian(U=2.0, filling=0.5, nk=4, maxite=2,
                **{name: VALUES[name]})


@pytest.mark.parametrize("name", KPM_KEYWORDS)
@pytest.mark.parametrize("spinful", [True, False])
def test_the_ultracell_refuses_the_kpm_keywords(name, spinful, monkeypatch,
        tmp_path):
    """A spinful supercell goes through VJinteraction, a spinless one
    through the sparse loop of densitydensity_kpm with the ultracell
    backend"""
    monkeypatch.chdir(tmp_path)
    h = geometry.chain().get_hamiltonian(has_spin=spinful)
    u = h.get_ultracell(h.get_supercell(4))
    interaction = dict(U=2.0) if spinful else dict(V1=1.5)
    with pytest.raises(TypeError, match=_match(name)):
        u.get_mean_field_hamiltonian(filling=0.5, nk=2, maxite=2,
                write=False, **interaction, **{name: VALUES[name]})


def test_the_kpm_keywords_at_their_default_are_not_refused():
    from pyqula.scftk.spinspin import VJinteraction
    h = geometry.chain().get_hamiltonian()
    scf = VJinteraction(h, U=2.0, filling=0.5, nk=4, maxite=2,
            **{name: None for name in KPM_KEYWORDS})
    assert scf.hamiltonian.has_spin
