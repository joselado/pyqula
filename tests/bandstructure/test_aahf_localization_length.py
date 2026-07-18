import numpy as np

from pyqula import geometry
from pyqula import potentials


def test_aahf_localization_length_matches_reference(tmp_path, monkeypatch):
    """Regression check for the AAH-potential localization length vs.
    potential strength on a finite chain, at a smaller size (chain(60)
    instead of 400) and coarser sweep (6 values instead of 30): the summed
    mean localization lengths must match the value recorded from a
    known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain(60)
    g.dimensionality = 0
    vs = np.linspace(0.0, 4.0, 6)

    def discard(w):
        w2 = np.abs(w) * np.abs(w)
        n = len(w)
        if np.sum(w2[0:n // 10]) > 0.5 or np.sum(w2[9 * n // 10:n]) > 0.5:
            return True
        else:
            return False

    lm = []
    for v in vs:
        h = g.get_hamiltonian(has_spin=False)
        fun = potentials.aahf1d(v=v, beta=0.0)
        h.add_onsite(fun)
        (es, ls) = h.get_tails(discard=discard)
        lm.append(np.mean(ls))

    assert np.isclose(np.sum(lm), 1.8667903744738448, atol=1e-4)
