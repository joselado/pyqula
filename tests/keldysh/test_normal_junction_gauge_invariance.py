import numpy as np
import pytest
from scipy.integrate import quad

from pyqula import algebra
from pyqula import geometry
from pyqula import heterostructures
from pyqula.operators import get_electron, get_hole


def _tauz(h):
    return np.array((get_electron(h) - get_hole(h)).todense())


def _static_bias_current(h1, h2, transparency, voltage, delta, central=None):
    """Independent, non-Floquet reference: apply the bias directly as a
    static +-voltage/2 shift (via the Nambu tauz operator) on each lead's
    onsite term, and integrate the (now manifestly time-independent)
    zero-temperature Landauer current over the resulting chemical-potential
    window [-voltage/2, voltage/2]. This is the same physics the
    Floquet-Keldysh gauge transform describes, computed a completely
    different way (no Floquet sidebands at all).

    Any explicit `central` Hamiltonians are shifted with the LEFT lead,
    because that is the electrostatic model the Floquet path implements: it
    puts the AC-carrying bond on the junction's rightmost bond, so the whole
    central region sits in the left lead's gauge region at the left lead's
    potential (see keldyshtk.current._dense_floquet_integrand). Shifting the
    central region differently is a different physical model, not a
    different gauge, and would legitimately give a different current."""
    tauz = _tauz(h1)
    h1b = h1.copy()
    h1b.intra = h1b.intra + (voltage/2)*tauz
    h2b = h2.copy()
    h2b.intra = h2b.intra - (voltage/2)*tauz
    kwargs = {}
    if central is not None:
        shifted = []
        for hc in central:
            hcb = hc.copy()
            hcb.intra = hcb.intra + (voltage/2)*tauz
            shifted.append(hcb)
        kwargs["central"] = shifted
    HTb = heterostructures.build(h1b, h2b, **kwargs)
    HTb.set_coupling(transparency)
    HTb.delta = delta
    f = lambda e: HTb.didv(energy=e)
    val, _ = quad(f, -abs(voltage)/2, abs(voltage)/2, limit=100, epsrel=1e-5)
    return val*np.sign(voltage)


def _nambu_chain():
    h = geometry.chain().get_hamiltonian()
    h.turn_nambu()
    return h


@pytest.mark.parametrize("transparency", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("voltage", [0.3, 0.6, 1.0, -0.3, -0.6])
def test_normal_junction_matches_static_bias_reference(transparency, voltage):
    """A two-terminal junction between two plain (non-superconducting)
    leads, promoted to trivial (zero-pairing) Nambu form so the
    Floquet-Keldysh machinery applies, must reduce to ordinary
    (non-Floquet) biased Landauer transport: the DC current computed via
    the gauge-transformed Floquet-Keldysh formalism
    (Heterostructure.get_dc_current) must match the current obtained by
    directly biasing each lead's chemical potential and integrating the
    resulting (static) transmission over the bias window. These are two
    physically equivalent but numerically unrelated ways of describing the
    same bias, related only by an exact gauge transform -- any bug in the
    Floquet-sideband bookkeeping would break this equivalence. Negative
    voltages are included so that a current-reversal (I(-V) = -I(V)) sign
    bug cannot slip through undetected."""
    h0 = geometry.chain().get_hamiltonian()
    h1 = h0.copy()
    h1.turn_nambu()
    h2 = h1.copy()

    HT = heterostructures.build(h1.copy(), h2.copy())
    HT.set_coupling(transparency)
    HT.delta = 1e-4

    Icalc = HT.get_dc_current(voltage, nmax=8, nmax_max=30, tol=1e-4)
    Iref = _static_bias_current(h1, h2, transparency, voltage, HT.delta)

    assert abs(Icalc-Iref) < 2e-2*max(abs(Iref), 1e-8)


def _detuned_dot(h1, eps):
    """A central site detuned from the leads by `eps`, as a valid BdG
    Hamiltonian: the shift carries the Nambu grading (+eps on electrons,
    -eps on holes). `h.shift_fermi(eps)` does the same thing through
    pyqula's own API -- see
    test_central_site_detuning_conventions_agree."""
    hc = h1.copy()
    hc.intra = hc.intra + eps*_tauz(h1)
    return hc


@pytest.mark.parametrize("transparency", [0.3, 1.0])
@pytest.mark.parametrize("voltage", [0.3, -0.3])
@pytest.mark.parametrize("eps", [0.0, 0.5])
def test_single_central_site_matches_static_bias_reference(transparency,
                                                            voltage, eps):
    """The same gauge-invariance check as above, for a junction carrying one
    explicit central site (`heterostructures.build(h1,h2,central=[hc])`,
    always `block_diagonal=False` -- a dense `central_intra` -- in pyqula),
    including one detuned from the leads (`eps=0.5`, a quantum dot rather
    than a piece of lead).

    This case used to raise NotImplementedError: a "confirmed, reproducible
    2-8% systematic error, growing at low transparency" had been reported
    against this very reference for a detuned central site. It does not
    reproduce -- see the module comment in keldyshtk/current.py for the
    measurement and for what does reproduce it (a central site detuned
    WITHOUT the Nambu grading, which is not a BdG Hamiltonian at all, so the
    scattering-matrix reference formula does not apply to it)."""
    h1 = _nambu_chain()
    h2 = h1.copy()
    hc = _detuned_dot(h1, eps)

    HT = heterostructures.build(h1.copy(), h2.copy(), central=[hc.copy()])
    assert not HT.block_diagonal  # pyqula's representation for one site
    HT.set_coupling(transparency)
    HT.delta = 1e-4

    Icalc = HT.get_dc_current(voltage, nmax=4, nmax_max=16, tol=1e-3)
    Iref = _static_bias_current(h1, h2, transparency, voltage, HT.delta,
                                central=[hc])
    assert abs(Icalc-Iref) < 2e-2*max(abs(Iref), 1e-8)


def test_block_diagonal_central_list_matches_static_bias_reference():
    """Same check for the other representation of an explicit central
    region: several central sites, which pyqula keeps as a block list
    (`block_diagonal=True`) rather than one dense matrix -- a different code
    path into the same Floquet solve (`enlarge_hlist` rather than
    `_dense_hlist`)."""
    h1 = _nambu_chain()
    h2 = h1.copy()
    hc = _detuned_dot(h1, 0.4)

    HT = heterostructures.build(h1.copy(), h2.copy(),
                                central=[hc.copy(), hc.copy()])
    assert HT.block_diagonal
    HT.set_coupling(0.6)
    HT.delta = 1e-4

    Icalc = HT.get_dc_current(0.3, nmax=4, nmax_max=16, tol=1e-3)
    Iref = _static_bias_current(h1, h2, 0.6, 0.3, HT.delta,
                                central=[hc, hc])
    assert abs(Icalc-Iref) < 3e-2*max(abs(Iref), 1e-8)


def test_central_region_agreement_improves_with_smaller_broadening():
    """The residual Floquet-vs-reference difference for a central-region
    junction is an O(`HT.delta`) regularization artifact, not a systematic
    error: the Floquet solve adds `i*delta` to every sideband of the whole
    scattering region, which the static scattering-matrix reference does not
    do the same way. Pinning this down is what closed out the "2-8%
    systematic error" report that had this case rejected, so it is asserted
    rather than left as a comment: the disagreement must fall by at least
    ~5x for a 10x smaller broadening (measured: 9.4e-2 -> 9.5e-3 -> 9.5e-4
    for delta = 1e-3, 1e-4, 1e-5 at transparency 0.1).

    Deliberately run at low transparency, where the current is small and the
    same absolute artifact is largest in relative terms -- the regime the
    original report described as the error "growing"."""
    h1 = _nambu_chain()
    h2 = h1.copy()
    hc = _detuned_dot(h1, 0.5)
    rels = []
    for delta in (1e-3, 1e-4):
        HT = heterostructures.build(h1.copy(), h2.copy(), central=[hc.copy()])
        HT.set_coupling(0.1)
        HT.delta = delta
        Icalc = HT.get_dc_current(0.3, nmax=4, nmax_max=16, tol=1e-3)
        Iref = _static_bias_current(h1, h2, 0.1, 0.3, delta, central=[hc])
        rels.append(abs(Icalc-Iref)/abs(Iref))
    assert rels[1] < rels[0]/5.


def test_central_region_with_non_hermitian_lead_coupling():
    """A multi-orbital lead (a chain with a two-site unit cell), whose
    coupling to the central region is NOT Hermitian -- the only shape that
    can tell apart the two ways of storing a bond in the block chain
    `_dense_hlist` builds.

    Every single-orbital case is blind to this: there `right_coupling` is
    real and diagonal, so storing `hlist[2][1]` as `right_coupling` or as
    `dagger(right_coupling)` builds the identical Hamiltonian. The
    pre-restriction `_dense_hlist` used the opposite convention to
    `enlarge_hlist` (the one the validated scattering-matrix path uses) and
    was measured here to be 97.7% wrong on this junction while the shipped
    convention agrees with the reference to 2.2e-3 -- a real latent bug for
    multi-orbital or spin-orbit-coupled leads, invisible to every
    single-orbital test."""
    g = geometry.chain(2)  # two sites per unit cell -> non-Hermitian inter
    h1 = g.get_hamiltonian()
    h1.turn_nambu()
    h2 = h1.copy()
    hc = _detuned_dot(h1, 0.5)

    HT = heterostructures.build(h1.copy(), h2.copy(), central=[hc.copy()])
    HT.set_coupling(0.6)
    HT.delta = 1e-4
    rc = np.asarray(algebra.todense(HT.right_coupling))
    assert not np.allclose(rc, rc.conj().T)  # the property being tested

    Icalc = HT.get_dc_current(0.3, nmax=4, nmax_max=8, tol=1e-2)
    Iref = _static_bias_current(h1, h2, 0.6, 0.3, HT.delta, central=[hc])
    assert abs(Icalc-Iref) < 5e-2*max(abs(Iref), 1e-8)


def test_central_region_finite_temperature_reduces_to_zero_temperature():
    """The dense central-region path has its own `temperature` branch (the
    Fermi factor in `lesser_from_retarded`, applied per sideband). The
    agreement checks above are all at `temperature=0.`, so exercise the
    finite-temperature branch too and require it to be continuous in the
    temperature: at a temperature far below every scale in the problem
    (bandwidth, detuning, bias) it must reproduce the zero-temperature
    current, and a physically-sized temperature must change it without
    blowing up."""
    h1 = _nambu_chain()
    h2 = h1.copy()
    hc = _detuned_dot(h1, 0.5)

    def current(temperature):
        HT = heterostructures.build(h1.copy(), h2.copy(), central=[hc.copy()])
        HT.set_coupling(0.6)
        HT.delta = 1e-4
        return HT.get_dc_current(0.3, nmax=4, nmax_max=16, tol=1e-3,
                                  temperature=temperature)

    I0 = current(0.)
    assert abs(current(1e-5)-I0) < 1e-3*abs(I0)
    Iwarm = current(0.05)
    assert np.isfinite(Iwarm) and abs(Iwarm-I0) < 0.5*abs(I0)


def test_central_site_detuning_conventions_agree():
    """`hc.shift_fermi(eps)` and an explicit `eps*tauz` shift are the same
    physical detuning of the central site, and must give the same current --
    the check that separates "how the test case was built" from "what the
    solver does", which is exactly what the earlier 2-8% report turned on.
    (Adding `eps*identity` instead is NOT particle-hole symmetric and so is
    not a BdG Hamiltonian; it is not covered here because the
    scattering-matrix reference it would be compared against does not apply
    to it.)"""
    h1 = _nambu_chain()
    h2 = h1.copy()
    hc_tauz = _detuned_dot(h1, 0.5)
    hc_shift = h1.copy()
    hc_shift.shift_fermi(0.5)

    out = []
    for hc in (hc_tauz, hc_shift):
        HT = heterostructures.build(h1.copy(), h2.copy(), central=[hc.copy()])
        HT.set_coupling(0.6)
        HT.delta = 1e-4
        out.append(HT.get_dc_current(0.3, nmax=4, nmax_max=16, tol=1e-3))
    assert abs(out[0]-out[1]) < 1e-6*max(abs(out[0]), 1e-8)
