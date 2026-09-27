"""A normal tip on a superconductor, deep inside the gap.

Inside the gap a normal tip can only pass current into a superconductor
by Andreev reflection, which transfers two electrons and so goes as the
fourth power of the tip-sample hopping T, against T^2 for single-particle
tunneling: the decay rate kappa=(dlogG/dlogT)/(dlogG_N/dlogT) is 2 at weak
coupling. The only normal (electron) conductance left in the gap is the
one the broadening delta of the sample's Green's function gives it, a
Dynes density of states of relative size delta/Delta, so that it equals
G_N*delta/Delta: it goes as T^2 and vanishes with delta.

The scattering matrix of a LocalProbe used to carry an extra
i*delta_smatrix on both sites of its central region (the tip's last cell
and the probed site), a sink coupled to no lead. It removed a
T-independent 8*delta_smatrix = 8e-12 from the reflected current, which
didv_BdG reported as normal conductance, so that the S-matrix was unitary
only to 4e-12, the in-gap electron conductance at T=1e-3 was 18 times its
Dynes value, and kappa(E=0) at T=1e-3 came out 0.80, a value no mixture
of T^2 and T^4 processes can give. Every test below fails on that code.
"""
import numpy as np

from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe
from pyqula.transporttk.smatrix import get_smatrix

GAP = 0.1


def _sample(pairing=True, onsite=-1.0):
    """Spinful chain with its Fermi level inside the band"""
    h = geometry.chain().get_hamiltonian()
    h.add_onsite(onsite)
    if pairing: h.add_swave(GAP)
    else: h.setup_nambu_spinor() # same Nambu basis, no pairing
    return h


def _probe(delta, T, **kwargs):
    lp = LocalProbe(_sample(**kwargs), delta=delta)
    lp.T = T
    return lp


def test_smatrix_of_a_local_probe_is_unitary():
    """With no broadening in the central region, the only anti-Hermitian
    parts are the two selfenergies, and Fisher-Lee is exactly unitary.
    The old code was unitary to 4e-12."""
    lp = _probe(1e-8, 1e-3)
    for T in [1e-3, 0.3, 1.0]:
        lp.T = T
        for e in [0.0, 0.05, 0.2]: # in the gap, and above it
            s = get_smatrix(lp, energy=e, check=False)
            S = np.block([[s[0][0], s[0][1]], [s[1][0], s[1][1]]])
            err = np.max(np.abs(S@S.conj().T - np.eye(S.shape[0])))
            assert err < 1e-13, (T, e, err)


def test_ingap_normal_conductance_is_the_dynes_value():
    """In the gap, the electron conductance is G_N*delta/Delta, with G_N
    the conductance of the same probe on the sample without pairing: it
    goes as T^2 and is proportional to delta. The old code added a
    T-independent 8e-12 to it (18 times the Dynes value at T=1e-3)."""
    delta = 1e-8
    for T in [1e-3, 1e-2]:
        ge = _probe(delta, T).didv(energy=0.0, component="electron")
        gn = _probe(delta, T, pairing=False).didv(energy=0.0)
        ratio = ge/(gn*delta/GAP)
        assert abs(ratio - 1.0) < 2e-2, (T, ge, gn, ratio)


def test_ingap_conductance_goes_as_the_fourth_power_of_the_coupling():
    """With a small broadening the in-gap conductance is Andreev
    reflection alone, so doubling the coupling multiplies it by 16. The
    old code gave a slope of 2.8 between T=1e-3 and 2e-3."""
    lp = _probe(1e-10, 1e-3)
    G1 = lp.didv(energy=0.0, T=1e-3)
    G2 = lp.didv(energy=0.0, T=2e-3)
    slope = np.log(G2/G1)/np.log(2.)
    assert abs(slope - 4.0) < 1e-2, slope
    GA1 = lp.didv(energy=0.0, T=1e-3, component="Andreev")
    GA2 = lp.didv(energy=0.0, T=1e-2, component="Andreev")
    assert abs((GA2/GA1)/1e4 - 1.0) < 1e-3, (GA1, GA2)


def test_decay_rate_is_two_at_weak_coupling():
    """kappa = 2 in the gap and 1 above it, at the weak coupling where
    the old code gave 0.80 in the middle of the gap."""
    lp = _probe(1e-10, 1e-3, onsite=-1.0)
    for e in [0.0, 0.05]:
        k = lp.get_kappa(energy=e, T=1e-3)
        assert abs(k - 2.0) < 1e-2, (e, k)
    k = lp.get_kappa(energy=0.2, T=1e-3)
    assert abs(k - 1.0) < 1e-2, k


def test_tunneling_conductance_is_proportional_to_the_sample_dos():
    """In the tunneling limit the single-particle conductance is
    4 pi^2 T^2 rho_tip rho_sample(E) (in e^2/h, rho_tip=1/pi at the end of
    the default chain tip), with rho_sample the DOS of the probed site
    without the tip (get_dos at T=0), in the gap as well as out of it.
    get_dos used to extract the sample's selfenergy with a broadening of
    1e-5 and add it back with 1e-12, which left a gain on the probed site:
    its in-gap DOS came out 2.4 times too small with the Fermi level
    inside the band and negative with it at the band edge."""
    T = 3e-4 # small enough that the tip does not broaden the site
    for onsite in [-1.0, 2.0]:
        lp = _probe(1e-6, 0.0, onsite=onsite)
        for e in [0.0, 0.05, 0.15]:
            lp.T = 0.0
            dos = lp.get_dos(energy=e) # DOS of the sample itself
            assert dos > 0., (onsite, e, dos)
            lp.T = T
            ge = lp.didv(energy=e, component="electron")
            ratio = ge/(4*np.pi*T**2*dos)
            assert abs(ratio - 1.0) < 1e-2, (onsite, e, ratio)
