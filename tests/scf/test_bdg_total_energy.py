import numpy as np

from pyqula import geometry, meanfield
from pyqula.superconductivity import get_eh_sector


def test_nambu_band_energy_matches_the_normal_state():
    """A Nambu (BdG) description of a state with NO pairing describes the
    same state as the normal-state Hamiltonian it was built from, so it
    must have the same total energy.

    Summing the BdG spectrum below the Fermi level is not that energy:
    H = (1/2) Psi^dag H_BdG Psi + (1/2) Tr h, so sum_{E<0} E_BdG =
    2*E_normal - Tr h. The invariant asserted here is the equality of the
    two descriptions, not either side of that identity."""
    for gf in [geometry.chain, geometry.honeycomb_lattice]:
        for mu in (0.0, 0.7):
            h = gf().get_hamiltonian()
            h.shift_fermi(-mu)
            hn = h.copy()
            hn.setup_nambu_spinor() # the same state, in the Nambu basis
            assert abs(h.get_total_energy(nk=20)
                       - hn.get_total_energy(nk=20)) < 1e-8


def test_nambu_scf_total_energy_matches_the_normal_state(tmp_path,
                                                         monkeypatch):
    """Same invariant, one level up: a repulsive intersite interaction on
    a Nambu Hamiltonian converges to EXACTLY zero pairing, so the whole
    mean-field chain (band energy + the mu*N un-shift + the double-counting
    energy) must reproduce the normal-state answer. It is the chain that
    has to be consistent: the band energy and the mu*N term used to be on
    a doubled (Nambu) scale while the double-counting term was not."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain()
    out = dict()
    for nambu in (False, True):
        h = g.get_hamiltonian()
        if nambu: h.setup_nambu_spinor()
        h0 = h.copy()
        s = meanfield.Vinteraction(h, V1=1.0, filling=0.5, nk=20,
                mf="ferroZ", mix=0.3, maxerror=1e-8, maxite=1000,
                load_mf=False, verbose=0)
        if nambu: # check the premise: the converged state has no pairing
            mfm = np.array(s.hamiltonian.intra) - np.array(h0.intra)
            assert np.max(np.abs(get_eh_sector(mfm, i=0, j=1))) < 1e-8
        out[nambu] = s.total_energy
    assert abs(out[True]-out[False]) < 1e-6
