"""The finite-temperature absorptive part of the Chebyshev (KPM) Kubo
conductivity, conductivitytk/kpmkubo.py, with T below the spacing of the
energy nodes.

The thermal occupation factor [f(E) - f(E+w)]/w is integrated over the
Chebyshev interpolant of the rest of the integrand near the two Fermi
edges, with its exact cosine moments, and by the midpoint rule on the nodes
away from them. So T -> 0 is the box of T=0, a T of several node spacings
is the plain midpoint rule, and in between the result does not depend on
the number of nodes. The occupation factor used to be sampled on the nodes
at every T > 0, which is wrong once the Fermi step falls between two of
them.
"""
import numpy as np

from pyqula import geometry
from pyqula import conductivity
from pyqula.conductivitytk import kpmkubo


def _metallic_cell():
    """The disordered gapped honeycomb cell of test_kpm_conductivity.py,
    18 orbitals, with the Fermi level moved into the valence band, and its
    exact moments"""
    rng = np.random.default_rng(1)
    g = geometry.honeycomb_lattice().get_supercell(3)
    h = g.get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.5)
    h.intra = h.intra + np.diag(0.3*(rng.random(len(g.r))-0.5))
    h.shift_fermi(1.2)
    st = kpmkubo._Setup(h, nk=2, npol=60)
    return st, kpmkubo.exact_moments(st)


def _nodes(st, ne=None):
    """The energy nodes of absorptive(), the one nearest to E=0.05, and the
    spacing there"""
    ne = 4*st.M if ne is None else ne
    E = st.a*np.cos(np.pi*(np.arange(ne)+0.5)/ne) + st.b
    j = int(np.argmin(np.abs(E-0.05)))
    return E, j, abs(E[j+1]-E[j])


def _fermi_f(x, T):
    return 1./(1.+np.exp(np.clip(x/T, -500., 500.)))


def _ratio(E, w, fermi, T):
    if w == 0.:
        f = _fermi_f(E-fermi, T)
        return f*(1.-f)/T
    return (_fermi_f(E-fermi, T)-_fermi_f(E+w-fermi, T))/w


def _rho(st, mu, E, w):
    """rho_ab(E,E+w) = sum_nm mu_nm phi_n(E+w) phi_m(E), the double series
    of the module docstring, at the energies E"""
    def phi(E):
        x = (E-st.b)/st.a
        return kpmkubo._phi(x, st.M)/(np.pi*st.a*np.sqrt(1.-x**2))
    nd, M = mu.shape[0], st.M
    Y = (mu.reshape((nd*nd*M, M))@phi(E)).reshape((nd, nd, M, len(E)))
    return np.sum(Y*phi(E+w)[None, None], axis=2)


def _thermal_reference(st, mu, w, fermi, T, spacing):
    """pi int dE r(E) rho(E,E+w), r = [f(E) - f(E+w)]/w, by Gauss-Legendre
    in energy on panels half the node spacing wide, refined geometrically
    down to T/8 around the two Fermi edges, so no node of absorptive() is
    used. The double series varies on the scale of the line width, seven
    node spacings. Energies within 1e-3 of the edges of the rescaled
    spectrum, where rho is negligible, are left out"""
    lo, hi = fermi-max(w, 0.), fermi-min(w, 0.)
    L = 44.*T
    bps = list(np.arange(lo-L, hi+L, spacing/2.))+[hi+L]
    for e in (lo, hi):
        bps.append(e)
        for c in 2.**np.arange(-3, 7):
            bps += [e+c*T, e-c*T]
    bps = np.unique(np.clip(np.array(bps), lo-L, hi+L))
    xg, wg = np.polynomial.legendre.leggauss(16)
    c, hw = (bps[:-1]+bps[1:])/2., (bps[1:]-bps[:-1])/2.
    E = (c[:, None]+hw[:, None]*xg[None, :]).ravel()
    wq = (hw[:, None]*wg[None, :]).ravel()
    ok = (np.abs((E-st.b)/st.a) < 0.999) & (np.abs((E+w-st.b)/st.a) < 0.999)
    E, wq = E[ok], wq[ok]
    return np.pi*_rho(st, mu, E, w)@(_ratio(E, w, fermi, T)*wq)


def test_vanishing_temperature_is_the_box_of_zero_temperature():
    """T = 1e-9 node spacings gives the T=0 result, where the occupation
    box is integrated in closed form, for boxes far narrower than the node
    spacing, a few spacings wide, much wider, of the other sign, and w=0,
    the top of the Drude peak"""
    st, mu = _metallic_cell()
    E, j, spacing = _nodes(st)
    fermi = E[j] + 0.37*(E[j+1]-E[j]) # between two nodes
    ws = np.array([spacing/20., spacing/3., 2.5*spacing, 0.4, -spacing/7.,
            0.])
    s0 = kpmkubo.absorptive(st, mu, ws, fermi=fermi, T=0.)
    s1 = kpmkubo.absorptive(st, mu, ws, fermi=fermi, T=1e-9*spacing)
    scale = np.max(np.abs(s0))
    assert scale > 1e-3 # a real signal
    assert np.max(np.abs(s1-s0)) < 1e-9*scale


def test_temperature_above_the_node_spacing_is_the_midpoint_rule():
    """With T several node spacings, as the default T = delta, the
    occupation factor is smooth on the scale of the nodes, its cosine
    moments beyond the number of nodes are below rounding, and the result
    is the midpoint rule on the nodes, r(E_j) pi/ne, the rule used at
    every T > 0 before. The reference applies that rule to the double
    series directly, with the same energy window"""
    st, mu = _metallic_cell()
    E, j, spacing = _nodes(st)
    T = 7.*spacing
    fermi = E[j] + 0.37*(E[j+1]-E[j])
    ws = np.array([spacing/3., 2.5*spacing, 0.4, -0.3, 0.])
    out = kpmkubo.absorptive(st, mu, ws, fermi=fermi, T=T)
    ne = 4*st.M
    th = np.pi*(np.arange(ne)+0.5)/ne
    pad = 10.*st.delta
    d = st.dirs
    for iw, w in enumerate(ws):
        keep = (E > st.emin-pad) & (E < st.emax+pad) & \
                (E+w > st.emin-pad) & (E+w < st.emax+pad) & \
                (np.abs(E+w-st.b) < st.a*(1.-1e-9))
        Ek = E[keep]
        G = st.a*np.sin(th[keep])*_rho(st, mu, Ek, w) # dE = a sin dtheta
        ref = np.pi*G@(_ratio(Ek, w, fermi, T)*np.pi/ne)
        scale = np.max(np.abs(ref))
        assert np.max(np.abs(out[iw][np.ix_(d, d)]-ref)) < 1e-12*scale, w


def test_absorption_below_the_node_spacing_is_the_energy_integral():
    """Across the crossover, T from a third of the node spacing to three
    spacings, the absorptive part is pi int dE r(E) rho(E,E+w) done by
    Gauss-Legendre in energy on panels finer than T, with the Fermi level
    on a node and between two"""
    st, mu = _metallic_cell()
    E, j, spacing = _nodes(st)
    d = st.dirs
    ws = np.array([spacing/3., 2.5*spacing, 0.])
    for fermi, T in [(E[j], spacing/3.), (E[j]+0.37*(E[j+1]-E[j]), spacing),
            (E[j]+0.37*(E[j+1]-E[j]), 3.*spacing)]:
        out = kpmkubo.absorptive(st, mu, ws, fermi=fermi, T=T)
        for iw, w in enumerate(ws):
            ref = _thermal_reference(st, mu, w, fermi, T, spacing)
            scale = np.max(np.abs(ref))
            assert np.max(np.abs(out[iw][np.ix_(d, d)]-ref)) < 1e-10*scale, \
                    (fermi, T, w)


def test_absorption_with_a_tail_at_one_end_of_the_spectrum_is_the_energy_integral():
    """A Fermi level in the upper and in the lower part of the band, at a T
    whose thermal tail reaches one end of the rescaled spectrum and not the
    other, so that the grid of the edge starts at theta = 0 or at pi; and a
    box wider than the two tails, so that each edge has its own grid. The
    absorptive part is still the energy integral"""
    st, mu = _metallic_cell()
    E, j, spacing = _nodes(st)
    d = st.dirs
    for x, T, ws in [(0.6, 3.*spacing, [spacing/3., 0.]),
            (-0.6, 3.*spacing, [spacing/3., 0.]),
            (0.1, spacing/3., [0.8, -0.8])]:
        fermi = st.b+x*st.a
        Y = kpmkubo._SUPPORT*T/st.a # the tail, in the rescaled energy
        if x != 0.1: assert (x+Y >= 1.) != (x-Y <= -1.) # one end only
        else: assert 0.8 > 2.*kpmkubo._SUPPORT*T # separate tails
        out = kpmkubo.absorptive(st, mu, np.array(ws), fermi=fermi, T=T)
        for iw, w in enumerate(ws):
            ref = _thermal_reference(st, mu, w, fermi, T, spacing)
            scale = np.max(np.abs(ref))
            assert scale > 1e-4 # a real signal
            assert np.max(np.abs(out[iw][np.ix_(d, d)]-ref)) < 1e-10*scale, \
                    (x, T, w)


def test_a_temperature_near_the_smallest_float_stays_finite():
    """At T = 1e-300 and 1e-305, with boxes from 2e-9 to a third of the node
    spacing, the weights stay finite and give the box of T=0, to the
    digits that a box so narrow keeps"""
    st, mu = _metallic_cell()
    E, j, spacing = _nodes(st)
    fermi = E[j] + 0.37*(E[j+1]-E[j])
    ws = np.array([2e-9, 1e-6, spacing/3.])
    s0 = kpmkubo.absorptive(st, mu, ws, fermi=fermi, T=0.)
    for T in [1e-300, 1e-305]:
        s1 = kpmkubo.absorptive(st, mu, ws, fermi=fermi, T=T)
        assert np.all(np.isfinite(s1))
        for iw in range(len(ws)):
            scale = np.max(np.abs(s0[iw]))
            assert np.max(np.abs(s1[iw]-s0[iw])) < 1e-6*scale, (T, ws[iw])


def test_absorption_below_the_node_spacing_does_not_depend_on_ne():
    """At T a fifth of the node spacing the same moments give the same
    absorptive part with 4 and 6 nodes per moment, for frequencies below,
    at and above the spacing. Sampled on the nodes, the Fermi step fell
    between two of them, and the two grids differed by tens of percent"""
    st, mu = _metallic_cell()
    E, j, spacing = _nodes(st)
    fermi = E[j] + 0.37*(E[j+1]-E[j])
    T = spacing/5.
    ws = np.array([spacing/20., spacing/3., 2.5*spacing, 0.4, 0.])
    s4 = kpmkubo.absorptive(st, mu, ws, fermi=fermi, T=T, ne=4*st.M)
    s6 = kpmkubo.absorptive(st, mu, ws, fermi=fermi, T=T, ne=6*st.M)
    for iw in range(len(ws)):
        scale = np.max(np.abs(s4[iw]))
        assert scale > 1e-3
        assert np.max(np.abs(s6[iw]-s4[iw])) < 1e-8*scale, ws[iw]


def test_drude_peak_below_the_node_spacing_matches_the_bands():
    """A clean ring of 400 sites with the Fermi level at 0.1 in the band,
    at a temperature a fifth of the spacing of the energy nodes there. The
    KPM Drude peak is smooth down to w = 0, its top, and follows the band
    calculation at the same temperature (integration="grid" on 8000
    k-points, the chain itself, fine enough for -df/dE) to the few percent
    of the Lorentz kernel, as at T=0 in test_kpm_conductivity.py, and the
    peak is the same with 6 instead of 4 energy nodes per moment. Sampled
    on the nodes, the Fermi derivative of width T between two nodes made
    the top of the peak about 30% too high or too low, depending on where
    the Fermi level fell"""
    fermi = 0.1
    g = geometry.chain().get_supercell(400)
    h = g.get_hamiltonian(has_spin=False, is_sparse=True)
    st = kpmkubo._Setup(h, delta=0.2, kernel="lorentz")
    delta = st.delta # the line width of npol polynomials
    ne = 4*st.M
    E = st.a*np.cos(np.pi*(np.arange(ne)+0.5)/ne) + st.b
    j = int(np.argmin(np.abs(E-fermi)))
    T = abs(E[j+1]-E[j])/5.
    ws = delta*np.array([0., 1./64, 1./32, 1./16, 1./8, 1./2, 1., 2.])
    kw = dict(integration="kpm", T=T, npol=st.npol, fermi=fermi,
            kernel="lorentz", trace="exact", reactive=False, component="xx")
    ws, s = conductivity.optical_conductivity(h, ws, **kw)
    ws, s6 = conductivity.optical_conductivity(h, ws, ne=6*st.M, **kw)
    assert np.max(np.abs(s6-s)) < 1e-8*abs(s[0])
    s = s.real
    h0 = geometry.chain().get_hamiltonian(has_spin=False)
    h0.shift_fermi(-fermi) # the band path occupies the states below zero
    ws, s0 = conductivity.optical_conductivity(h0, ws, nk=8000, T=T,
            delta=delta, component="xx")
    s0 = s0.real
    D = np.sqrt(4.-fermi**2)/np.pi
    assert abs(s0[0]-D/delta) < 0.02*D/delta # the reference is the peak
    assert abs(s[1]-s[0]) < 1e-3*s[0] # w = delta/64 against w = 0
    assert np.all(np.diff(s) < 0.) # the peak falls monotonically
    assert np.max(np.abs(s-s0)) < 0.05*s0[0]
