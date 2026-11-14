import numpy as np
import pytest

from pyqula import algebra
from pyqula import geometry
from pyqula import embedding
from pyqula.increase_hilbert import full2profile

# Embedding.get_gf declared an `operator` argument and never referenced it,
# so eb.get_gf(operator="sz") returned the plain Green's function while
# eb.get_ldos(operator="sz") -- the sibling method of the same class, built
# on top of that very Green's function -- did honour it. The first invariant
# used here is that relation between the two: whatever get_gf returns for a
# given operator has to reproduce get_ldos for the same one.
#
# Both used to weight the Green's function on one side, -Im diag(A G)/pi.
# Summed over the orbitals of a site that is the local matrix element only
# for an operator local to each site (a spin component): for a bond
# operator, a hopping or a random Hermitian matrix, it moved weight between
# the two ends of a bond, by 3e-2 on a honeycomb site, while h.get_ldos in
# every mode returns -Im diag(G A + A G)/(2 pi). The second invariant is
# therefore the pristine embedding against h.get_ldos(mode="green") of the
# same crystal, which is the same k-sum when the embedding uses
# gf_mode="full".

ARGS = dict(energy=0.2, delta=0.1, nk=6)


def _defect(h):
    """The Hamiltonian with a strong onsite potential on its first site.
    The sites of the honeycomb lattice are at x=-0.5 and x=0.5, so a
    potential selected by x=0 touches no site and is no defect at all"""
    r0 = h.geometry.r[0]
    hv = h.copy()
    hv.add_onsite(lambda r: 3.0 if np.sum(np.abs(r-r0)) < 1e-6 else 0.0)
    return hv


def _embedding():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_exchange([0., 0., 0.5])
    return embedding.Embedding(h, m=_defect(h).intra)


def _bond(h, seed=3):
    """A random Hermitian matrix of the cell, which couples the sites"""
    n = h.intra.shape[0]
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    return (a + a.conj().T)/2.


def test_the_operator_reaches_the_greens_function():
    eb = _embedding()
    plain = eb.get_ldos(write=False, **ARGS)[2]
    assert abs(plain[0]-plain[1]) > 1e-2 # the defect is on one site only
    for op in ["sz", "sx", _bond(eb.H)]: # the sx map is zero here
        g = np.array(eb.get_gf(operator=op, **ARGS))
        d = full2profile(eb.H, -np.diag(g).imag/np.pi, check=False)
        ref = eb.get_ldos(operator=op, write=False, **ARGS)[2]
        if not isinstance(op, str) or op == "sz": # not a vacuous map
            assert np.max(np.abs(ref)) > 1e-2
        assert np.max(np.abs(np.array(d)-np.array(ref))) < 1e-10


def test_an_operator_actually_changes_the_greens_function():
    """The exchange field polarizes the lattice, so the sz-resolved
    Green's function cannot coincide with the charge one."""
    eb = _embedding()
    plain = np.array(eb.get_gf(**ARGS))
    sz = np.array(eb.get_gf(operator="sz", **ARGS))
    assert np.max(np.abs(plain-sz)) > 1e-3


E, DELTA, NK = 0.4, 0.1, 12


def _pristine():
    """A crystal whose states are not eigenstates of any spin component,
    embedded without a defect"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_rashba(0.3)
    h.add_exchange([0.1, 0.2, 0.3])
    eb = embedding.Embedding(h)
    eb.gf_mode = "full" # the k-sum of h.get_ldos(mode="green", nk=)
    return h, eb


@pytest.mark.parametrize("which", ["sz", "sx", "bond", "hopping"])
def test_the_embedding_ldos_is_the_local_matrix_element(which):
    h, eb = _pristine()
    op = {"sz": "sz", "sx": "sx", "bond": _bond(h),
          "hopping": algebra.todense(h.intra)}[which]
    got = eb.get_ldos(energy=E, delta=DELTA, nk=NK, operator=op,
                      write=False)[2]
    ref = h.get_ldos(e=E, delta=DELTA, nk=NK, mode="green", operator=op,
                     nrep=1, write=False)[2]
    assert np.max(np.abs(np.array(ref))) > 1e-2 # not a vacuous map
    assert np.max(np.abs(np.array(got)-np.array(ref))) < 1e-10
    # the map sums over the sites to the operator-resolved DOS
    g = algebra.todense(eb.get_gf(energy=E, delta=DELTA, nk=NK))
    A = algebra.todense(h.get_operator(op).get_matrix())
    assert abs(np.sum(got) + np.trace(g@A).imag/np.pi) < 1e-10
    # the one-sided weight agrees for a spin component and not for a bond,
    # so the bond cases are the ones that pin the convention
    left = full2profile(h, -np.diag(A@g).imag/np.pi, check=False)
    if which in ["sz", "sx"]:
        assert np.max(np.abs(left-np.array(ref))) < 1e-10
    else:
        assert np.max(np.abs(left-np.array(ref))) > 1e-3


@pytest.mark.parametrize("op", ["sz", "sx"])
def test_a_named_operator_is_built_on_the_supercell(op):
    """With nsuper=2 the Green's function lives on the 2x2 supercell, and
    a named operator used to be built on the primitive cell and fail with
    a dimension mismatch. In the pristine crystal each of the four cells
    carries the map of nsuper=1"""
    h, eb = _pristine()
    one = eb.get_ldos(energy=E, delta=DELTA, nk=NK, operator=op,
                      write=False)[2]
    four = eb.get_ldos(energy=E, delta=DELTA, nk=NK, nsuper=2, operator=op,
                       write=False)[2]
    assert len(four) == 4*len(one)
    assert np.max(np.abs(np.array(four)-np.tile(one, 4))) < 1e-10
    g = np.array(eb.get_gf(energy=E, delta=DELTA, nk=NK, nsuper=2,
                           operator=op))
    d = full2profile(h, -np.diag(g).imag/np.pi, check=False)
    assert np.max(np.abs(np.array(d)-np.array(four))) < 1e-10


def test_an_operator_without_a_matrix_is_refused():
    h, eb = _pristine()
    with pytest.raises(NotImplementedError):
        eb.get_ldos(energy=E, delta=DELTA, nk=NK, operator="valley",
                    write=False)
    with pytest.raises(ValueError):
        eb.get_ldos(energy=E, delta=DELTA, nk=NK, nsuper=2,
                    operator=algebra.todense(h.intra), write=False)


def test_a_potential_is_built_on_the_supercell():
    """A potential f(r) is local to each site, so its map is f(r_i) times
    the plain map, at the positions that get_ldos returns. With a defect
    and nsuper=2 the four cells are not equivalent and f is different in
    each of them, so a potential built on the primitive cell and repeated
    cannot pass"""
    h = _pristine()[0]
    eb = embedding.Embedding(h, m=_defect(h).intra)
    kw = dict(energy=E, delta=DELTA, nk=NK, nsuper=2, write=False)
    x, y, plain = eb.get_ldos(**kw)
    x, y, got = eb.get_ldos(operator=lambda r: 1.0 + 0.3*r[0] + 0.17*r[1],
                            **kw)
    f = 1.0 + 0.3*np.array(x) + 0.17*np.array(y)
    assert np.std(plain) > 1e-2 # the defect is seen
    assert np.max(np.abs(f-np.tile(f[:2], 4))) > 0.1 # f is not periodic
    assert np.max(np.abs(np.array(got)-f*np.array(plain))) < 1e-10


@pytest.mark.parametrize("which", ["bond", "hopping"])
def test_a_matrix_of_the_supercell_matches_the_bulk_supercell(which):
    """nsuper=2 with nk k-points is the same k-sum as the Hamiltonian of
    the 2x2 supercell with nk/2, so a bond operator given as a matrix of
    the supercell pins the order of the cells in the Green's function of
    the embedding against the order of h.supercell"""
    h, eb = _pristine()
    hs = h.supercell(2)
    op = {"bond": _bond(hs), "hopping": algebra.todense(hs.intra)}[which]
    kw = dict(energy=E, delta=DELTA, nk=NK, nsuper=2)
    got = eb.get_ldos(operator=op, write=False, **kw)[2]
    ref = hs.get_ldos(e=E, delta=DELTA, nk=NK//2, mode="green", operator=op,
                      nrep=1, write=False)[2]
    assert np.max(np.abs(np.array(ref))) > 1e-2 # not a vacuous map
    assert np.max(np.abs(np.array(got)-np.array(ref))) < 1e-10
    g = algebra.todense(eb.get_gf(**kw))
    left = full2profile(h, -np.diag(op@g).imag/np.pi, check=False)
    assert np.max(np.abs(left-np.array(ref))) > 1e-3 # one-sided is off


@pytest.mark.parametrize("which", ["sz", "potential", "bond"])
def test_a_defect_given_in_a_supercell_takes_the_operator_of_the_supercell(
        which):
    """Embedding(h, m=<matrix of the 2x2 supercell>, nsuper=2) is the
    embedding of h.supercell(2) itself, so the two give the same map with
    any operator: a name and a potential are built on the supercell, and a
    matrix is the one of the supercell"""
    h, eb = _pristine()
    hs = h.supercell(2)
    m = _defect(hs).intra
    eb1 = embedding.Embedding(h, m=m, nsuper=2)
    eb2 = embedding.Embedding(hs, m=m)
    op = {"sz": "sz", "potential": lambda r: 1.0 + 0.3*r[0] + 0.17*r[1],
          "bond": _bond(hs)}[which]
    for nsuper in [1, 2]:
        if which == "bond" and nsuper == 2: continue # a matrix of 2x2 only
        kw = dict(energy=E, delta=DELTA, nk=6, nsuper=nsuper, write=False)
        plain = eb1.get_ldos(**kw)[2]
        one = eb1.get_ldos(operator=op, **kw)[2]
        two = eb2.get_ldos(operator=op, **kw)[2]
        assert len(one) == 8*nsuper**2
        assert np.std(plain) > 1e-2 # the defect is seen
        assert np.max(np.abs(np.array(one)-np.array(plain))) > 1e-3
        assert np.max(np.abs(np.array(one)-np.array(two))) < 1e-12
