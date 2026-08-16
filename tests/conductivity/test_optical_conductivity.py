"""Physical invariants of the Kubo-Greenwood optical conductivity
(src/pyqula/conductivity.py).

The absolute normalization is pinned by two analytically known results --
the f-sum-rule weight 2|t|/pi of a half-filled nearest-neighbour chain and
the universal optical conductivity pi*e^2/(4h) of (spinless) graphene --
and the antisymmetric part is pinned against pyqula's own, independent
Fukui-Hatsugai-Suzuki Chern number.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import conductivity
from pyqula import topology
from pyqula.conductivitytk import kubo


def _chain():
    """Half-filled spinless nearest-neighbour chain, E(K) = 2 t cos(K)
    with t = 1 (pyqula's default hopping)"""
    return geometry.chain().get_hamiltonian(has_spin=False)


def _haldane(t2=0.2, has_spin=False):
    """Spinless Haldane model, a Chern insulator. As in
    tests/topology/test_quantum_geometric_tensor.py, the direct gap for
    t2=0.2 is [-0.9,0.9] at every k and shift_fermi(0.3) leaves a safe
    margin to either band edge everywhere in the Brillouin zone."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_haldane(t2)
    h.shift_fermi(0.3)
    return h


def test_chain_weights_match_the_analytic_value():
    """The two weights that fix the absolute normalization of the module.
    For a half-filled nearest-neighbour chain, E(K) = 2 t cos(K) with
    K = 2 pi k the Cartesian crystal momentum, the diamagnetic weight is

      W_xx = (1/2pi) int dK f(E) (-2 t cos K) = 2|t|/pi

    and, since a single band has no interband transitions, the whole f-sum
    rule weight sits in the Drude term, so D_xx = W_xx = 2|t|/pi as well.
    A missing (or spurious) factor of 2*pi in the k-derivative, or the use
    of a normalized rather than a true lattice vector, would show up here
    immediately.

    Both are evaluated at T=0.05, which is a compromise: D is a Fermi
    surface integral of -df/dE, so the temperature has to stay well above
    the k-mesh level spacing (2|v|*2pi/nk = 0.06 here at nk=200) for the
    mesh to resolve it, while the analytic value is the T -> 0 one, from
    which finite T pulls it away by a Sommerfeld correction of order
    T^2."""
    h = _chain()
    W = conductivity.sum_rule_weight(h, nk=200, T=0.05)
    D = conductivity.drude_weight(h, nk=200, T=0.05)
    analytic = 2./np.pi
    assert np.isclose(W[0,0], analytic, atol=2e-3)
    assert np.isclose(D[0,0], analytic, atol=2e-3)
    # the chain lies along x, so every other component vanishes
    assert np.max(np.abs(W[1:,:])) < 1e-12
    assert np.max(np.abs(D[1:,:])) < 1e-12


def test_chain_f_sum_rule():
    """int_{-inf}^{inf} Re sigma_xx(omega) domega = pi * W_xx, with W the
    diamagnetic weight computed independently from the second
    k-derivative of the Hamiltonian. This ties the spectrum (Drude peak
    included) to the normalization checked above. The window is finite, so
    the Lorentzian tails cost a fraction of a percent."""
    h = _chain()
    ws = np.linspace(-40., 40., 4001)
    ws, sigma = conductivity.optical_conductivity(h, energies=ws, nk=200,
            T=0.05, delta=0.2)
    weight = np.trapezoid(sigma[:,0,0].real, ws)/np.pi
    W = conductivity.sum_rule_weight(h, nk=200, T=0.05)[0,0]
    assert np.isclose(weight, W, rtol=2e-2)
    assert np.isclose(weight, 2./np.pi, rtol=2e-2)


def test_band_velocity_matches_the_derivative_of_the_bands():
    """The diagonal elements of the velocity operator must be the band
    velocities dE_n/dK_alpha. Checked by finite differences of the Bloch
    eigenvalues along the Cartesian direction alpha, whose reduced-
    coordinate displacement is exactly the Jacobian row jac[:,alpha]
    (see kubo._velocities). This validates the reduced-to-Cartesian
    conversion and the 2*pi normalization of the k-derivative on a
    multiorbital, non-orthogonal lattice."""
    h = _haldane()
    hm, orders, hkgen, jac, dr, cellvol, scale = kubo._setup(h)
    from pyqula import algebra
    eps = 1e-5
    for k in [np.array([0.13,0.27,0.]), np.array([0.4,-0.1,0.])]:
        hk = kubo._hk(hkgen, k)
        (es, ws) = algebra.eigh(hk)
        v = kubo._velocities(hm, orders, jac, dr, hk, k)
        for alpha in range(2):
            dk = np.zeros(3)
            dk[:len(orders)] = jac[:,alpha] # Cartesian step, in reduced coords
            ep = algebra.eigh(kubo._hk(hkgen, k+eps*dk))[0]
            em = algebra.eigh(kubo._hk(hkgen, k-eps*dk))[0]
            fd = (ep-em)/(2.*eps)
            diag = np.array([np.real(np.conjugate(ws[:,n])@v[alpha]@ws[:,n])
                             for n in range(len(es))])
            assert np.allclose(diag, fd, atol=1e-5)


def test_graphene_universal_optical_conductivity():
    """Pristine graphene has a universal interband conductivity
    sigma = pi e^2/(2 h) once the two spins are counted, i.e.
    pi e^2/(4 h) = 1/8 in the e^2/hbar units used here for the spinless
    model, flat over the frequency window where the dispersion is Dirac
    like. This is a purely interband, absolute-scale check in 2D -- it is
    what catches a velocity operator that misses the intracell bonds."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    ws, sigma = conductivity.optical_conductivity(h,
            energies=[0.2,0.4], nk=120, T=0.01, delta=0.04)
    universal = np.pi/(4.*2.*np.pi) # pi e^2/(4 h), in e^2/hbar
    assert np.allclose(sigma[:,0,0].real, universal, rtol=0.1)
    assert np.allclose(sigma[:,1,1].real, universal, rtol=0.1)


def test_honeycomb_response_is_isotropic():
    """The honeycomb lattice is C3 symmetric, which forces the symmetric
    part of any rank-2 response tensor to be isotropic: sigma_xx =
    sigma_yy at every frequency, with no xy symmetric part. Only the
    antisymmetric (Hall) part survives, sigma_xy = -sigma_yx. The lattice
    is non-orthogonal, so this fails outright if the Cartesian velocity is
    built with the wrong lattice vectors or without the intracell bond
    vectors."""
    h = _haldane()
    ws, sigma = conductivity.optical_conductivity(h,
            energies=np.linspace(0., 4., 21), nk=18, T=0.02, delta=0.1)
    scale = np.max(np.abs(sigma[:,0,0]))
    assert np.max(np.abs(sigma[:,0,0]-sigma[:,1,1])) < 1e-10*scale
    assert np.max(np.abs(sigma[:,0,1]+sigma[:,1,0])) < 1e-10*scale
    # the lattice is in the xy plane, so nothing couples to z
    assert np.max(np.abs(sigma[:,2,:])) < 1e-14
    assert np.max(np.abs(sigma[:,:,2])) < 1e-14


def test_hall_conductivity_of_a_chern_insulator(tmp_path, monkeypatch):
    """The DC limit of the antisymmetric part of a Chern insulator is
    quantized, sigma_xy(omega->0) = -C e^2/h = -C/(2 pi) in the units of
    this module (see conductivity.py for that sign convention). C is taken
    from pyqula's independent Fukui-Hatsugai-Suzuki implementation, and
    both signs of the Haldane flux are checked so that the test really
    constrains the sign and not just the magnitude. sigma_xx must vanish
    at the same time -- a Chern insulator is gapped."""
    monkeypatch.chdir(tmp_path) # topology.chern writes *.OUT files to cwd
    for t2 in [0.2, -0.2]:
        h = _haldane(t2=t2)
        c = topology.chern(h, nk=20)
        assert abs(round(c)-c) < 1e-6 and round(c) != 0
        ws, sigma = conductivity.optical_conductivity(h, energies=[0.],
                nk=30, T=0.01, delta=1e-3)
        assert np.isclose(2.*np.pi*sigma[0,0,1].real, -round(c), atol=1e-3)
        assert abs(sigma[0,0,1].imag) < 1e-6
        assert abs(sigma[0,0,0]) < 1e-3


def test_time_reversal_symmetric_model_has_no_hall_response():
    """A time-reversal symmetric insulator (here a honeycomb lattice with
    a sublattice imbalance) has no Hall response at any frequency, while
    its longitudinal absorption is finite."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.4)
    ws, sigma = conductivity.optical_conductivity(h,
            energies=np.linspace(0., 4., 21), nk=24, T=0.02, delta=0.05)
    assert np.max(np.abs(sigma[:,0,1])) < 1e-12
    assert np.max(sigma[:,0,0].real) > 0.1


def test_gapped_insulator_absorption_edge():
    """Below the direct gap an insulator does not absorb: the only
    sub-gap Re sigma_xx is the tail of the Lorentzian broadening, so the
    test is relative to the peak height and stays a few broadenings away
    from the edge. The absorption then switches on at the direct gap
    reported by the (independent) h.get_gap()."""
    h = _haldane()
    gap = h.get_gap()
    ws = np.linspace(0., 4., 81)
    ws, sigma = conductivity.optical_conductivity(h, energies=ws, nk=24,
            T=0.02, delta=0.05)
    re = sigma[:,0,0].real
    peak = np.max(re)
    assert np.max(re[ws<gap-4*0.05]) < 0.05*peak # no sub-gap absorption
    onset = ws[np.argmax(re>0.3*peak)] # steep edge, well above the tail
    assert abs(onset-gap) < 4*0.05 # onset at the direct gap
    assert np.min(re) > -1e-12 # Re sigma_xx is dissipative, hence positive


def test_supercell_gives_the_same_conductivity():
    """The conductivity is an intensive quantity: describing the same
    crystal with a 2x2 supercell (four times the bands, a folded
    Brillouin zone, four times the cell area) must give the same
    sigma_ab(omega). This also exercises the numpy.matrix hoppings that
    get_supercell produces."""
    g = geometry.honeycomb_lattice()
    hs = []
    for gi, nk in [(g,24), (g.get_supercell(2),12)]:
        h = gi.get_hamiltonian(has_spin=False)
        h.add_haldane(0.15)
        h.shift_fermi(0.2)
        hs.append(conductivity.optical_conductivity(h,
            energies=np.linspace(0.5,3.,6), nk=nk, T=0.02, delta=0.1)[1])
    assert np.allclose(hs[0], hs[1], atol=1e-8)


def test_drude_weight_separates_metals_from_insulators():
    """The Drude weight vanishes in a gapped system and is positive in a
    metal, and switching the intraband channel off removes exactly that
    contribution from sigma(omega->0)."""
    metal = _chain()
    insulator = _haldane()
    assert conductivity.drude_weight(metal, nk=200, T=0.05)[0,0] > 0.5
    assert abs(conductivity.drude_weight(insulator, nk=24, T=0.02)[0,0]) < 1e-6
    # the metal's zero-frequency conductivity is the Drude peak, D/delta
    delta = 0.05
    ws, sigma = conductivity.optical_conductivity(metal, energies=[0.],
            nk=200, T=0.05, delta=delta)
    D = conductivity.drude_weight(metal, nk=200, T=0.05)[0,0]
    assert np.isclose(sigma[0,0,0].real, D/delta, rtol=1e-2)
    ws, inter = conductivity.optical_conductivity(metal, energies=[0.],
            nk=200, T=0.05, delta=delta, intraband=False)
    assert abs(inter[0,0,0]) < 1e-10 # a single band has nothing interband


def test_conductivity_is_converged_in_nk():
    """The quantized Hall response of a gapped model must not depend on
    the k-mesh once it is dense enough."""
    h = _haldane()
    out = [conductivity.optical_conductivity(h, energies=[0.], nk=nk,
             T=0.01, delta=1e-3)[1][0,0,1].real for nk in [18,36]]
    assert abs(out[0]-out[1]) < 1e-4


def test_component_selection_and_channel_switches():
    """component= returns the same numbers as the full tensor, and the
    intraband and interband channels add up to the full response."""
    h = _haldane()
    ws = np.linspace(0., 3., 11)
    kw = dict(energies=ws, nk=12, T=0.05, delta=0.1)
    full = conductivity.optical_conductivity(h, **kw)[1]
    xy = conductivity.optical_conductivity(h, component="xy", **kw)[1]
    assert np.allclose(xy, full[:,0,1])
    intra = conductivity.optical_conductivity(h, interband=False, **kw)[1]
    inter = conductivity.optical_conductivity(h, intraband=False, **kw)[1]
    assert np.allclose(intra+inter, full)


def test_unsupported_hamiltonians_raise():
    """Zero-dimensional and three-dimensional Hamiltonians, and Nambu
    (superconducting) ones, are rejected rather than silently wrong."""
    h3 = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError):
        conductivity.optical_conductivity(h3, energies=[1.], nk=4)
    hsc = geometry.chain().get_hamiltonian(has_spin=True)
    hsc.add_swave(0.2)
    with pytest.raises(NotImplementedError):
        conductivity.optical_conductivity(hsc, energies=[1.], nk=4)
