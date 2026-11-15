import numpy as np
import pytest

from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.scftk.kondolattice import hs_constant

# A Kondo coupling between a moment and the conduction electrons of other
# sites and other cells: each bond b = (c, m, R, J_b) couples the moment m of
# the cell 0 to the conduction site c of the cell R and has its own field
# V_b = -(J_b/2) sum_s <c^dag_{c,R,s} f_{m,0,s}>. The tests use three oracles
# that do not go through the bond code: a coupling to the conduction site
# of another cell only is the onsite model with the conduction electrons
# relabeled, so every observable must agree; the Bloch matrix of the chain
# written by hand from the converged fields; and the same lattice in a
# doubled cell, where half of the bonds become intracell.

J0, FILLING, NK, T = 1.5, 0.15, 60, 0.02


def displaced(R, J=J0):
    """Coupling J between the moment at rm and the conduction site at
    rm + R only"""
    R = np.array(R, dtype=float)
    def f(rc, rm):
        return J if np.linalg.norm(rc - rm - R) < 1e-6 else 0.0
    return f


def by_distance(values):
    """Coupling that depends on |rc - rm| only, values[n] at distance n"""
    def f(rc, rm):
        d = np.linalg.norm(rc - rm)
        n = int(round(d))
        if abs(d - n) > 1e-6 or n >= len(values): return 0.0
        return values[n]
    return f


def chain_kondo(n=1):
    g = geometry.chain()
    if n > 1: g = g.get_supercell(n)
    return KondoLatticeHamiltonian(g.get_hamiltonian(has_spin=True))


def converge(h, J, nk=NK, seed=0.3, **kwargs):
    kw = dict(J=J, filling=FILLING, nk=nk, T=T, mf=(seed, 0.0),
            maxerror=1e-10, maxite=4000, return_total_energy=True)
    kw.update(kwargs)
    h2, e = h.get_mean_field_hamiltonian(**kw)
    assert h2 is not None, "the loop did not converge"
    return h2, e


@pytest.fixture(scope="module")
def onsite():
    return converge(chain_kondo(), J0)


def test_onsite_function_is_the_onsite_number(onsite):
    """A function that is J0 at zero distance and zero elsewhere is the
    onsite coupling J0, bond for bond"""
    h0, e0 = onsite
    h1, e1 = converge(chain_kondo(), by_distance([J0]))
    assert h1.kondo_bonds == [(0, 0, (0, 0, 0), J0)]
    assert np.max(np.abs(h1.hybridization - h0.hybridization)) < 1e-12
    assert np.max(np.abs(h1.constraint_lambda - h0.constraint_lambda)) < 1e-12
    assert abs(e1 - e0) < 1e-12


@pytest.mark.parametrize("R", [(1, 0, 0), (-1, 0, 0), (2, 0, 0)])
def test_bond_to_another_cell_is_the_onsite_model_relabeled(onsite, R):
    """f_j coupled only to c_{j+R} is the onsite model for the conduction
    electrons c'_j = c_{j+R}, which have the same bands. The field of the
    bond, the multiplier, the occupation and the energy must be the onsite
    ones, and so must every observable that the relabeling leaves alone:
    the spectral functions A_cc and A_FF at each k (c_k only picks up a
    phase) and the susceptibility of the moments, also with the
    hybridization rescaled. A bond read from dm[R] instead of dm[-R], or
    a scale that misses the hoppings between cells, changes them"""
    h0, e0 = onsite
    h1, e1 = converge(chain_kondo(), displaced(R))
    assert h1.kondo_bonds == [(0, 0, R, J0)]
    assert abs(h0.hybridization[0]) > 0.3 # a hybridized state
    assert np.max(np.abs(h1.hybridization - h0.hybridization)) < 1e-8
    assert np.max(np.abs(h1.constraint_lambda - h0.constraint_lambda)) < 1e-8
    assert np.max(np.abs(h1.local_occupation - 1.)) < 1e-8
    assert abs(e1 - e0) < 1e-8
    kw = dict(kpath=[[0.13, 0., 0.], [0.31, 0., 0.]],
            energies=np.linspace(-3., 3., 41), delta=0.05)
    for scale in (1.0, 1.5):
        k, e, a0 = h0.get_composite_spectral_function(
                hybridization_scale=scale, **kw)
        k, e, a1 = h1.get_composite_spectral_function(
                hybridization_scale=scale, **kw)
        for key in ("cc", "FF", "gF"):
            assert np.max(np.abs(a1[key] - a0[key])) < 1e-7, (key, scale)
        assert np.max(a0["FF"]) > 0.1 # not vacuous
        kw2 = dict(q=[0.37, 0., 0.], energies=np.linspace(0.2, 3., 8),
                nk=40, delta=0.05, U=-0.5, hybridization_scale=scale)
        e, c0 = h0.get_moment_spinchi(**kw2)
        e, c1 = h1.get_moment_spinchi(**kw2)
        assert np.max(np.abs(c0)) > 1e-2
        assert np.max(np.abs(c1 - c0)) < 1e-7, scale


ASYMMETRIC = {0: 0.8, 1: 1.4, -1: 0.5} # J of the bond to c_{j+R}, by R


def asymmetric(rc, rm):
    d = rc[0] - rm[0]
    n = int(round(d))
    if abs(d - n) > 1e-6 or abs(rc[1]-rm[1]) + abs(rc[2]-rm[2]) > 1e-6:
        return 0.0
    return ASYMMETRIC.get(n, 0.0)


def bloch_by_hand(h, V, Rs, lam, mu, k):
    """Bloch matrix of one spin of the chain mean field, in the basis
    (c, f), written from the definition: sum_j V f^dag_j c_{j+R} is
    sum_k e^{2 pi i k R} V f^dag_k c_k"""
    hc = chain_kondo().get_hk_gen()([k, 0., 0.]) # bare bands of c and f
    eps = np.array(hc)[0, 0].real
    hfc = np.sum(V*np.exp(2j*np.pi*k*np.array(Rs)))
    return np.array([[eps - mu, np.conj(hfc)], [hfc, lam - mu]])


def neighbors(rc, rm):
    """Coupling 2 to the conduction electrons of the two neighbors only"""
    return 2.0 if abs(np.linalg.norm(rc - rm) - 1.0) < 1e-6 else 0.0


# two chains with bonds to other cells: three different couplings at
# filling 0.15, where every field is real, and the half-filled chain
# coupled to its neighbors only, where the two fields are imaginary and
# opposite, V(k) ~ sin k, the state of the guide. The second one is what
# tells V from conj(V) at R != 0, and gives the imaginary step of the
# stationarity test something to test
MODELS = {"asymmetric": (asymmetric, FILLING, 0.2),
        "neighbors": (neighbors, 0.5, [0.4j, -0.4j])}
_converged = dict()


def model(name):
    if name not in _converged:
        J, filling, seed = MODELS[name]
        h = chain_kondo()
        h2, e = converge(h, J, seed=seed, filling=filling)
        mu = h.get_fermi4filling(filling, nk=NK) # the loop fixes it this way
        _converged[name] = (h2, e, mu)
    return _converged[name]


@pytest.fixture(scope="module")
def asym():
    return model("asymmetric")


def test_the_two_models_are_what_they_should_be():
    """Three different real fields, and two opposite imaginary ones"""
    h2, e, mu = model("asymmetric")
    Rs = [b[2][0] for b in h2.kondo_bonds]
    assert Rs == [0, -1, 1] or Rs == [0, 1, -1]
    assert np.allclose([b[3] for b in h2.kondo_bonds],
            [ASYMMETRIC[R] for R in Rs])
    V = h2.hybridization
    assert np.min(np.abs(V)) > 1e-3 # every bond is hybridized
    assert np.max(np.abs(np.abs(V[1]) - np.abs(V[2]))) > 0.02 # different
    h2, e, mu = model("neighbors")
    assert [b[2][0] for b in h2.kondo_bonds] == [-1, 1]
    V = h2.hybridization
    assert np.max(np.abs(V.real)) < 1e-8, V
    assert abs(V[0] + V[1]) < 1e-8, V
    assert abs(V[0].imag) > 0.4, V # 0.498 at nk=60


@pytest.mark.parametrize("name", list(MODELS))
def test_bond_fields_are_the_order_parameters(name):
    """The fields must be -(J_b/2) sum_s <c^dag_{j+R} f_j> of the Bloch
    matrix written by hand from the definitions, and n_f must be one. A
    bond placed at -R instead of R, or read with the indices exchanged,
    gives a different number: in the first model the three couplings are
    different, in the second one the fields are complex"""
    h2, e, mu = model(name)
    bonds = h2.kondo_bonds
    Rs = [b[2][0] for b in bonds]
    Js = np.array([b[3] for b in bonds])
    V, lam = h2.hybridization, h2.constraint_lambda[0]
    A, nf = np.zeros(len(bonds), dtype=np.complex128), 0.
    ks = h2.geometry.get_kmesh(nk=NK)
    for k in ks:
        es, us = np.linalg.eigh(bloch_by_hand(h2, V, Rs, lam, mu, k[0]))
        occ = 1./(np.exp(es/T) + 1.)
        cf = np.sum(occ*np.conj(us[0])*us[1]) # <c^dag_k f_k>, one spin
        A += 2*np.exp(-2j*np.pi*k[0]*np.array(Rs))*cf/len(ks)
        nf += 2*np.sum(occ*np.abs(us[1])**2)/len(ks)
    assert abs(nf - 1.) < 1e-6, nf
    assert np.max(np.abs(V + Js*A/2.)) < 1e-6, (V, -Js*A/2.)


@pytest.mark.parametrize("name", list(MODELS))
def test_grand_potential_is_stationary_in_every_bond(name):
    """Hellmann-Feynman bond by bond: the grand potential of the Bloch
    matrix written by hand, with the constant 2|V_b|^2/J_b of each bond,
    is stationary in the real and the imaginary part of every V_b at the
    converged fields. It fixes the constant of each bond independently"""
    h2, e, mu = model(name)
    bonds = h2.kondo_bonds
    Rs = [b[2][0] for b in bonds]
    Js = np.array([b[3] for b in bonds])
    V0, lam = h2.hybridization.copy(), h2.constraint_lambda[0]
    ks = h2.geometry.get_kmesh(nk=NK)

    def omega(V):
        band = 0.
        for k in ks:
            es = np.linalg.eigvalsh(bloch_by_hand(h2, V, Rs, lam, mu, k[0]))
            band += -2*T*np.sum(np.logaddexp(0., -es/T))/len(ks)
        return band + hs_constant(V, Js) - lam

    d = 1e-3
    for b in range(len(bonds)):
        for step in (d, 1j*d):
            dv = np.zeros(len(bonds), dtype=np.complex128)
            dv[b] = step
            op, o0, om = omega(V0+dv), omega(V0), omega(V0-dv)
            first, second = (op-om)/(2*d), (op-2*o0+om)/d**2
            assert abs(second) > 1e-2, (b, step, second)
            assert abs(first) < 1e-4, (b, step, first)


def test_a_bond_that_decays_does_not_stop_the_loop():
    """The half-filled chain with an onsite and a first-neighbor coupling:
    with particle-hole symmetry and the onsite field, <c^dag_{j+1} f_j>
    vanishes, so the neighbor fields decay from the seed to zero while the
    onsite one converges. Relative to its own size a field at roundoff
    never converges, so a field counts as converged once it moves by less
    than a precision of 1e-10 times the largest field, and the loop
    stops, with the onsite field of the onsite model and the neighbor
    fields below the tolerance"""
    def J(rc, rm):
        d = np.linalg.norm(rc - rm)
        if d < 1e-6: return 4.0
        return 0.5 if abs(d - 1.0) < 1e-6 else 0.0
    kw = dict(filling=0.5, nk=100, mf=(1.0, 0.0), maxerror=1e-5,
            maxite=3000)
    h1 = chain_kondo().get_mean_field_hamiltonian(J=J, **kw)
    h0 = chain_kondo().get_mean_field_hamiltonian(J=4.0, **kw)
    V0 = h0.hybridization[0]
    assert abs(V0) > 1.8
    assert abs(h1.hybridization[0] - V0) < 1e-4
    assert np.max(np.abs(h1.hybridization[1:])) < 1e-5*abs(V0)


J_SMALL = 1e-4 # a weak coupling to the next cell, see below


def weak_neighbor(rc, rm):
    """Onsite J0, and J_SMALL to the conduction electron of the next cell
    only"""
    d = rc - rm
    if np.linalg.norm(d) < 1e-6: return J0
    if np.linalg.norm(d - np.array([1., 0., 0.])) < 1e-6: return J_SMALL
    return 0.0


def test_each_field_is_measured_against_its_size_with_a_floor():
    """A field converges relative to its own size, as the field of a
    single moment does, but a field smaller than a tenth of the largest
    one, or zero by symmetry, is measured against a tenth of the largest
    one: it counts as converged once it moves by less than maxerror/10
    times the largest field"""
    from pyqula.scftk.kondolattice import field_residual, FIELD_FLOOR
    assert FIELD_FLOOR == 0.1
    d = 1e-6
    # a field of half the largest one: relative to itself
    assert np.isclose(field_residual(np.array([1., 0.5]),
            np.array([1., 0.5+d])), d/(0.5+d))
    # a field of a hundredth of the largest one: against a tenth of it
    assert np.isclose(field_residual(np.array([1., 0.01]),
            np.array([1., 0.01+d])), d/0.1)
    # a field zero by symmetry, at roundoff
    assert np.isclose(field_residual(np.array([1., 1e-16]),
            np.array([1., -1e-16])), 2e-16/0.1)
    # every field decaying together does not converge
    assert np.isclose(field_residual(np.array([1e-3, 1e-4]),
            np.array([0.5e-3, 0.5e-4])), 0.5)


def test_a_small_field_converges_to_a_tenth_of_maxerror_of_the_largest():
    """The chain with the onsite coupling J0 and a weak coupling J_SMALL
    to the next cell, whose field is small but genuine, of order 1e-5 of
    the onsite one. The loop starts from the converged onsite field and
    multiplier and from zero on the weak bond, so the weak field is the
    one that has to converge, and it must come out within maxerror/10 of
    the onsite field, times a margin for the rate of the loop"""
    h = chain_kondo()
    href, e = converge(h, weak_neighbor) # maxerror=1e-10
    assert href.kondo_bonds == [(0, 0, (0, 0, 0), J0),
            (0, 0, (1, 0, 0), J_SMALL)]
    Vref = href.hybridization
    ratio = abs(Vref[1])/abs(Vref[0])
    assert abs(Vref[0]) > 0.3 and ratio < 1e-1, Vref # a small field
    maxerror = 1e-4
    seed = (np.array([Vref[0], 0.0]), href.constraint_lambda)
    h2 = h.get_mean_field_hamiltonian(J=weak_neighbor, filling=FILLING,
            nk=NK, T=T, mf=seed, maxerror=maxerror, maxite=4000)
    err = abs(h2.hybridization[1] - Vref[1])/abs(Vref[0])
    assert err < 20*maxerror/10, (err, h2.hybridization, Vref)


def unfolded(h, n):
    """The bonds of a cell of n sites of the chain, as a dictionary from
    (moment, displacement of the conduction site from the moment) to the
    field"""
    return {(m, c + n*R[0] - m): v for ((c, m, R, J), v)
            in zip(h.kondo_bonds, h.hybridization)}


def test_doubled_cell_gives_the_same_bonds():
    """The chain with an onsite and a first-neighbor coupling, in its
    primitive cell and in a cell of two sites, where one of the two
    neighbor bonds of each moment is inside the cell and the other one
    crosses to the next cell. Every bond field, the multiplier and the
    energy per site must be the same, and so must the local spectral
    function, folded, and the susceptibility of the moments at q=0 and at
    the zone boundary of the primitive cell"""
    J = by_distance([1.2, 0.4])
    h1, e1 = converge(chain_kondo(1), J, nk=60)
    h2, e2 = converge(chain_kondo(2), J, nk=30)
    assert len(h1.kondo_bonds) == 3 and len(h2.kondo_bonds) == 6
    b1, b2 = unfolded(h1, 1), unfolded(h2, 2)
    assert sorted(b1) == [(0, -1), (0, 0), (0, 1)]
    assert sorted(b2) == [(m, x) for m in (0, 1) for x in (-1, 0, 1)]
    assert abs(b1[(0, 1)]) > 0.02 and abs(b1[(0, 0)]) > 0.2
    for (m, x), v in b2.items():
        assert abs(v - b1[(0, x)]) < 1e-7, (m, x, v, b1[(0, x)])
    assert np.max(np.abs(h2.constraint_lambda - h1.constraint_lambda[0])) \
            < 1e-7
    assert abs(e2 - 2*e1) < 1e-7
    # A_cc averaged over the pairs of the doubled cell at K is the average
    # of the primitive one at K/2 and K/2 + 1/2
    es = np.linspace(-3., 3., 31)
    kw = dict(energies=es, delta=0.05)
    K = 0.26
    k, e, a2 = h2.get_composite_spectral_function(kpath=[[K, 0., 0.]], **kw)
    k, e, a1 = h1.get_composite_spectral_function(
            kpath=[[K/2, 0., 0.], [K/2 + 0.5, 0., 0.]], **kw)
    for key in ("cc", "FF"):
        assert np.max(np.abs(a2[key][0] - np.mean(a1[key], axis=0))) < 1e-7
    # sum over the moments of the doubled cell, with and without the sign
    # of the zone boundary, against the primitive cell at q=0 and q=1/2
    kw = dict(q=[0., 0., 0.], energies=np.linspace(0.3, 3., 6), delta=0.05,
            U=-0.4)
    e, c2 = h2.get_moment_spinchi(nk=30, **kw)
    s = np.array([1., -1.])
    for sign, q in ((np.ones(2), 0.), (s, 0.5)):
        kw["q"] = [q, 0., 0.]
        e, c1 = h1.get_moment_spinchi(nk=60, **kw)
        fold = np.einsum("a,wab,b->w", sign, c2, sign)/2.
        assert np.max(np.abs(c1[:, 0, 0])) > 1e-2
        assert np.max(np.abs(fold - c1[:, 0, 0])) < 1e-7, q


def test_bonds_inside_a_finite_cluster():
    """A finite chain of six sites with an onsite and a first-neighbor
    coupling: every bond is inside the one cell. The fields and the
    constraint must be those of the eigenstates of the whole cluster"""
    g = geometry.chain(6)
    g.dimensionality = 0
    hc = g.get_hamiltonian(has_spin=True)
    h = KondoLatticeHamiltonian(hc)
    J = by_distance([1.6, 0.5])
    h2 = h.get_mean_field_hamiltonian(J=J, filling=0.15, T=T,
            mf=(0.3, 0.0), maxerror=1e-10, maxite=6000)
    assert h2 is not None
    bonds = h2.kondo_bonds
    assert len(bonds) == 6 + 2*5 # onsite, and the neighbors of the chain
    assert all(b[2] == (0, 0, 0) for b in bonds)
    es, us = np.linalg.eigh(np.array(h2.intra))
    occ = 1./(np.exp(es/T) + 1.)
    rho = (us*occ[None, :])@us.conj().T # rho[i,j] = <c^dag_j c_i>
    pairs = h2._kondo_pairs
    for (c, m, R, Jb), V in zip(bonds, h2.hybridization):
        f = pairs[m][1]
        A = rho[2*f, 2*c] + rho[2*f+1, 2*c+1] # sum_s <c^dag_c f_f>
        assert abs(V + Jb*A/2.) < 1e-6, (c, m, V, -Jb*A/2.)
    for (c, f) in pairs:
        assert abs((rho[2*f, 2*f] + rho[2*f+1, 2*f+1]).real - 1.) < 1e-6


def test_amplitudes_take_the_onsite_bond(asym):
    """With J= the composite fermion and the cotunneling amplitude are
    those of the site, 4 conj(V)/(3J) and 2 tf conj(V)/J with the onsite
    bond, also when the moment is coupled to its neighbors; a coupling to
    other sites only has no such amplitude, and says so"""
    from pyqula.kondotk import cotunneling, compositespectrum
    h2, e, mu = asym
    pairs = h2._kondo_pairs
    V0 = h2.hybridization[0] # the onsite bond comes first
    assert h2.kondo_bonds[0][2] == (0, 0, 0)
    z = compositespectrum.composite_amplitude(h2, pairs, asymmetric)
    assert abs(z[0] - 4*np.conj(V0)/(3*ASYMMETRIC[0])) < 1e-12
    amp = cotunneling._tf_per_site(0.7, asymmetric, h2, [0])
    assert abs(amp[0] - 2*0.7*np.conj(V0)/ASYMMETRIC[0]) < 1e-12
    h1, e1 = converge(chain_kondo(), displaced((1, 0, 0)))
    with pytest.raises(ValueError, match="onsite Kondo coupling"):
        h1.get_composite_spectral_function(J=displaced((1, 0, 0)), nk=4)
    with pytest.raises(ValueError, match="onsite Kondo coupling"):
        h1.get_cotunneling_didv(J=displaced((1, 0, 0)), nk=4)


def test_amplitudes_need_the_onsite_bond_of_the_selected_moment_only():
    """A cell of two moments and a function J that is J0 at the onsite bond
    of the first moment and zero at the second one. For the first moment
    the composite fermion and the cotunneling spectrum must be those of
    the number J0, since only its own onsite bond enters, and the second
    moment must not stop them. Asking for the second moment must name it,
    moment 1, and not its place in the list of the selected pairs"""
    h2, e2 = converge(chain_kondo(2), J0, nk=30)
    pairs = h2._kondo_pairs
    r0 = np.array(h2.geometry.r[pairs[0][0]]) # where the first moment sits
    def J(rc, rm): # onsite J0 at the first moment only
        if np.linalg.norm(rc - rm) < 1e-6 and np.linalg.norm(rm - r0) < 1e-6:
            return J0
        return 0.0
    kw = dict(kpath=[[0.13, 0., 0.]], energies=np.linspace(-3., 3., 61),
            delta=0.05, site=0)
    k, e, a0 = h2.get_composite_spectral_function(J=J0, **kw)
    k, e, a1 = h2.get_composite_spectral_function(J=J, **kw)
    assert np.max(np.abs(a0["FF"])) > 1e-2 # not vacuous
    for key in ("cc", "FF", "cF", "gF"):
        assert np.max(np.abs(a1[key] - a0[key])) < 1e-12, key
    e, d0 = h2.get_cotunneling_didv(J=J0, site=0, nk=4)
    e, d1 = h2.get_cotunneling_didv(J=J, site=0, nk=4)
    assert np.max(np.abs(d0)) > 1e-2 # not vacuous
    assert np.max(np.abs(d1 - d0)) < 1e-12
    for f in (h2.get_composite_spectral_function, h2.get_cotunneling_didv):
        with pytest.raises(ValueError, match="at the moment 1"):
            f(J=J, site=1, nk=4)
        with pytest.raises(ValueError, match="at the moment 1"):
            f(J=J, site=[1], nk=4)


def test_bad_inputs_say_what_is_wrong():
    h = chain_kondo()
    with pytest.raises(ValueError, match="rcut"):
        h.get_mean_field_hamiltonian(J=1.0, rcut=2.0, nk=4)
    with pytest.raises(ValueError, match="one V per bond, 3"):
        h.get_mean_field_hamiltonian(J=by_distance([1.0, 0.5]), nk=4,
                mf=([0.3, 0.3], [0.0]))
    with pytest.raises(ValueError, match="real and finite"):
        h.get_mean_field_hamiltonian(J=lambda rc, rm: 1j, nk=4)
    with pytest.raises(TypeError, match="J must be a number or a function"):
        from pyqula.scftk.kondolattice import function_bonds
        function_bonds([(0, 1)], [[0., 0., 0.]], [], 1.0)
