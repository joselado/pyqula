"""A distance-dependent interaction Vr or Jr must give one number per pair
of sites, Vr(r_i, r_j+R) = Vr(r_j, r_i-R), since n_i n_j and S_i.S_j are
symmetric.

A function that gives two numbers used to go through with no error, and
what read it did not agree: the Hartree-Fock mean field averaged the two
ends of a pair, and the Bethe-Salpeter kernel and the RPA vertex V(q) took
one end each, which is not a Hermitian V(q). The usual way to write such a
function is a modulation that is not periodic in the unit cell, which is
symmetric in its two arguments and so looks right. Every builder raises
now, the dense one, the sparse one and the one of a finite system, with
the keyword the function was given with in the message.

The coupling must also be real and finite. A complex one is not the
coupling of n_i n_j or S_i.S_j, and the comparison of the two ends takes
no conjugate, so it would let an imaginary one through. An infinite value
at zero distance was the scale of the relative tolerance and switched the
check off."""
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyqula import geometry, meanfield, specialhopping
from pyqula.bsetk.interaction import density_interaction
from pyqula.chitk.densitychi import _density_v
from pyqula.scftk import sparsemeanfield


def chain4():
    return geometry.chain().get_supercell(4)


X0 = np.mean(chain4().r[:, 0]) # the middle of the cell


def _first_neighbors(r1, r2):
    return abs(np.linalg.norm(np.array(r1) - np.array(r2)) - 1.) < 1e-3


def ramp(r1, r2):
    """Symmetric in its two arguments and not periodic in the cell of four
    sites: the pair across the cell is 1.2 from site 3 and 0.8 from site 0"""
    if not _first_neighbors(r1, r2): return 0.
    return 1. + 0.1*((r1[0] + r2[0])/2. - X0)


def one_ended(r1, r2):
    """Periodic in the cell of four sites and not symmetric in its two
    arguments: it is modulated with the position of the first site alone"""
    if not _first_neighbors(r1, r2): return 0.
    return 1. + 0.3*np.cos(np.pi*(r1[0] - X0 - 0.5)/2.)


def periodic(r1, r2):
    """A modulation of the pair with the period of the cell of four sites,
    symmetric in its two arguments"""
    if not _first_neighbors(r1, r2): return 0.
    return 1. + 0.3*np.cos(np.pi*(r1[0] + r2[0] - 2*X0)/4.)


def test_reciprocity_check_compares_the_two_ends_of_every_pair():
    """The check compares v[d][i,j] with v[-d][j,i]. A direction whose
    opposite is absent is a pair seen from one end only, so it is a
    mismatch, and so is a cell matrix that is not symmetric. The tolerance
    is relative to the largest entry. Dense and sparse matrices give the
    same answer."""
    check = specialhopping.require_reciprocal_interaction
    m = np.array([[0., 1.], [0.5, 0.]])
    off = np.array([[0., 1.], [0., 0.]])
    for form in (np.array, csr_matrix):
        def make(v): return {d: form(x) for d, x in v.items()}
        check(make({})) # no interaction at all
        check(make({(0, 0, 0): m + m.T, (1, 0, 0): m, (-1, 0, 0): m.T}))
        for v in ({(1, 0, 0): m}, # the opposite direction is absent
                  {(0, 0, 0): m}, # not symmetric inside the cell
                  {(1, 0, 0): m, (-1, 0, 0): m}): # not transposed
            with pytest.raises(ValueError, match="not reciprocal"):
                check(make(v))
        # 1e-4 on an entry of 1e6 is roundoff, 1e-10 on one of 1e-6 is not
        check(make({(1, 0, 0): 1e6*m, (-1, 0, 0): 1e6*m.T + 1e-4*off}))
        with pytest.raises(ValueError, match="not reciprocal"):
            check(make({(1, 0, 0): 1e-6*m, (-1, 0, 0): 1e-6*m.T + 1e-10*off}))
    # the message names the pair, its two values and the keyword
    with pytest.raises(ValueError) as err:
        check({(1, 0, 0): m, (-1, 0, 0): m}, name="Jr")
    text = str(err.value)
    assert "Jr = 1 from site 0 to site 1 of the cell (1,0,0)" in text, text
    assert "Jr = 0.5 for the same pair" in text, text
    assert "mismatch of 0.5" in text, text


def _builders(g, **extra):
    """Every route that turns a function of two positions into an
    interaction, as (label, keyword in the message, call, whether the call
    goes on to a self-consistent loop). The two sparse mean fields reach
    the sparse builder through the call of the engine itself."""
    spinless = g.get_hamiltonian(has_spin=False)
    spinful = g.get_hamiltonian(has_spin=True)
    sparse = g.get_hamiltonian(has_spin=True, is_sparse=True)
    kw = dict(rcut=1.5, filling=0.5, nk=4, mf="random", **extra)
    return [
        ("builder", "Vr", lambda f: specialhopping.distance_cut_interaction(
            g, f, rcut=1.5), False),
        ("mean field Vr", "Vr", lambda f: meanfield.VJinteraction(
            spinful, Vr=f, **kw), True),
        ("mean field Jr", "Jr", lambda f: meanfield.VJinteraction(
            spinful, Jr=f, **kw), True),
        ("public mean field", "Vr", lambda f:
            spinless.get_mean_field_hamiltonian(Vr=f, **kw), True),
        ("Bethe-Salpeter kernel", "Vr", lambda f: density_interaction(
            spinful, Vr=f, rcut=1.5), False),
        ("density response", "Vr", lambda f: _density_v(
            spinless, Vr=f, rcut=1.5), False),
        ("sparse builder Vr", "Vr", lambda f: sparsemeanfield.interaction(
            sparse, Vr=f, rcut=1.5), False),
        ("sparse mean field Vr", "Vr", lambda f: meanfield.VJinteraction(
            sparse, Vr=f, integration="kpm", **kw), True),
        ("sparse mean field Jr", "Jr", lambda f: meanfield.VJinteraction(
            sparse, Jr=f, integration="kpm", **kw), True),
    ]


LABELS = [b[0] for b in _builders(chain4())]


def _builder(label, **extra):
    """(keyword in the message, call) of one route"""
    (out,) = [b[1:3] for b in _builders(chain4(), **extra) if b[0] == label]
    return out


@pytest.mark.parametrize("label", LABELS)
def test_interaction_that_is_not_reciprocal_raises(label):
    """Both ways of writing two numbers for one pair raise in every
    builder, before any loop starts, and the hopping builder underneath
    does not add its own warning, whose advice is the one of a hopping."""
    name, call = _builder(label)
    for f, mismatch in ((ramp, "0.4"), (one_ended, "0.3")):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="the hopping function")
            with pytest.raises(ValueError, match="not reciprocal") as err:
                call(f)
        text = str(err.value)
        assert text.startswith(name + " is not reciprocal"), text
        assert "mismatch of " + mismatch in text, text
    # the pair across the cell, from its two ends
    with pytest.raises(ValueError) as err:
        call(ramp)
    assert (name + " = 1.2") in str(err.value), str(err.value)
    assert (name + " = 0.8") in str(err.value), str(err.value)


@pytest.mark.parametrize("label", LABELS)
def test_periodic_modulation_is_accepted(label):
    """The same modulation written with the period of the cell is one
    number per pair, and every route takes it. The routes with a
    self-consistent loop go into the loop, which is all that is asked of
    them here: two iterations, converged or not."""
    name, call = _builder(label, maxite=2)
    try: call(periodic)
    except meanfield.MeanFieldConvergenceError: pass


def imaginary(r1, r2):
    """The same imaginary number from the two ends of a pair: symmetric,
    so the comparison of the two ends alone would take it"""
    return 0.3j if _first_neighbors(r1, r2) else 0.


def hermitian(r1, r2):
    """A complex number and its conjugate from the two ends of a pair, as
    a hopping would have"""
    if not _first_neighbors(r1, r2): return 0.
    return 0.3*np.exp(0.5j*np.sign(r2[0] - r1[0]))


@pytest.mark.parametrize("label", LABELS)
def test_complex_interaction_raises(label):
    """The coupling of n_i n_j or of S_i.S_j is a real number. A complex
    one raises with a message that says so, also when it is Hermitian, and
    not with the message of a pair with two values."""
    name, call = _builder(label)
    for f in (imaginary, hermitian):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="the hopping function")
            with pytest.raises(ValueError) as err:
                call(f)
        text = str(err.value)
        assert text.startswith(name + " must be real"), text
        assert "from site" in text and "of the cell" in text, text


def diverging(modulation):
    """1/d times 1 + modulation*x of the first site, infinite at zero
    distance: one number per pair without the modulation, two with it"""
    def f(r1, r2):
        d = np.linalg.norm(np.array(r1) - np.array(r2))
        if d < 1e-9: return np.inf
        return (1. + modulation*(r1[0] - X0))/d
    return f


@pytest.mark.parametrize("label", LABELS)
def test_infinite_value_does_not_switch_the_check_off(label):
    """An infinite value at zero distance used to be the scale of the
    relative tolerance, so that no mismatch was ever above it. Where the
    pair of a site with itself is used (an onsite U in the mean field and
    in the density response) the infinite value itself raises. The
    Bethe-Salpeter kernel drops that pair, so it takes the tail and still
    sees the two values of the modulated one."""
    name, call = _builder(label)
    if label == "Bethe-Salpeter kernel":
        with pytest.raises(ValueError, match="Vr is not reciprocal"):
            call(diverging(0.3))
        with warnings.catch_warnings(): # and numpy has nothing to say
            warnings.simplefilter("error", RuntimeWarning)
            W = call(diverging(0.)) # a Coulomb tail without a soft core
        assert all(np.all(np.isfinite(m)) for m in W.values())
        assert np.max(np.abs(np.diag(W[(0, 0, 0)]))) == 0.
        # first neighbors at distance 1, in the cell and across it
        assert all(abs(np.max(np.abs(W[d])) - 1.) < 1e-14
                   for d in [(0, 0, 0), (1, 0, 0), (-1, 0, 0)])
        return
    for modulation in (0., 0.3):
        with pytest.raises(ValueError) as err:
            call(diverging(modulation))
        text = str(err.value)
        assert text.startswith(name + " is not finite: "
                + name + " = inf from site"), text


@pytest.mark.parametrize("label", LABELS)
def test_nan_between_two_sites_raises(label):
    """A NaN between two different sites is not a coupling, in every
    route, the Bethe-Salpeter kernel too"""
    name, call = _builder(label)
    def f(r1, r2):
        if not _first_neighbors(r1, r2): return 0.
        return np.nan if abs((r1[0] + r2[0])/2. - X0) < 0.1 else 1.
    with pytest.raises(ValueError) as err:
        call(f)
    assert str(err.value).startswith(name + " is not finite"), str(err.value)


def test_finite_system_raises_too():
    """An island has no lattice vector, so only the symmetry in the two
    arguments is left, with and without rcut. A ramp is one number per
    pair there, and the builder returns it as it is."""
    g = chain4()
    g.dimensionality = 0
    for rcut in (None, 1.5):
        with pytest.raises(ValueError, match="Vr is not reciprocal"):
            specialhopping.distance_cut_interaction(g, one_ended, rcut=rcut)
        v = specialhopping.distance_cut_interaction(g, ramp, rcut=rcut)
        assert list(v) == [(0, 0, 0)]
        for i in range(3):
            assert abs(v[(0, 0, 0)][i, i+1] - ramp(g.r[i], g.r[i+1])) < 1e-14
            assert abs(v[(0, 0, 0)][i+1, i] - ramp(g.r[i], g.r[i+1])) < 1e-14


def test_builder_output_is_reciprocal_for_a_coulomb_tail():
    """A function of the distance alone passes on a lattice with two sites
    per cell and a long range, and what the builder returns has
    v[d] = v[-d]^T exactly"""
    g = geometry.honeycomb_lattice()
    def coulomb(r1, r2):
        dr = np.array(r1) - np.array(r2)
        return 0.8/np.sqrt(dr.dot(dr) + 0.25)
    v = specialhopping.distance_cut_interaction(g, coulomb, rcut=4.0)
    assert len(v) > 20
    for d, m in v.items():
        back = v[tuple(-x for x in d)]
        assert np.max(np.abs(m - back.T)) < 1e-14
