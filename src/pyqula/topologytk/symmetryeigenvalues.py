"""Eigenvalues of a point-group operation on the occupied states, and the
invariants they give

At a momentum k that the operation maps onto itself up to a reciprocal
lattice vector, k' = k + G, the Bloch Hamiltonian of pyqula has no
intracell phases, so H(k') = H(k) exactly, and the matrix P(k) of
symmetrytk.pointgroup commutes with H(k). Its eigenvalues on the occupied
states are then the symmetry labels of the bands at k, which is what the
invariants below are built from.

C. Fang, M. J. Gilbert and B. A. Bernevig, Bulk topological invariants in
noninteracting point group symmetric insulators, Phys. Rev. B 86, 115112
(2012), arXiv:1207.5767 (the Chern number modulo n, read from the TeX
source: eqs. ChernC2, ChernC4, ChernC3 and ChernC6)
L. Fu and C. L. Kane, Topological insulators with inversion symmetry,
Phys. Rev. B 76, 045302 (2007), cond-mat/0611341 (the parity Z2)

The rotations are about the z axis, counterclockwise for n>0, and with this
orientation the Chern number modulo n has the sign of h.get_chern().
"""
import numpy as np

from ..symmetrytk.pointgroup import (SymmetryOperation, CompiledSymmetry,
                                     compile_symmetry)

_FRACTIONS = (0., 1/2, 1/3, 2/3, 1/4, 3/4, 1/6, 5/6)


def rotation_operation(n, center=(0., 0., 0.)):
    """The counterclockwise rotation by 2 pi/n about the z axis through
    center, as a SymmetryOperation"""
    a = 2*np.pi/n
    R = np.array([[np.cos(a), -np.sin(a), 0.],
                  [np.sin(a), np.cos(a), 0.],
                  [0., 0., 1.]])
    return SymmetryOperation(R, center=center, name="C"+str(n))


def inversion_operation(center=(0., 0., 0.)):
    """The inversion r -> 2 center - r, as a SymmetryOperation"""
    return SymmetryOperation(-np.eye(3), center=center, name="I")


def compile_operation(h, op):
    """A CompiledSymmetry of h for op, which is a SymmetryOperation or an
    already compiled one; raises if op is not a symmetry of h"""
    if isinstance(op, CompiledSymmetry): return op
    if not isinstance(op, SymmetryOperation):
        raise TypeError("the operation must be a SymmetryOperation or a "
          +"CompiledSymmetry from symmetrytk.pointgroup, got "
          +type(op).__name__)
    c = compile_symmetry(h, op)
    if c is None:
        raise ValueError("the operation "+op.name+" about the center "
          +str(op.center.tolist())+" is not a symmetry of this "
          +"Hamiltonian; check the center, or that no term breaks it")
    return c


def invariant_momenta(h, op):
    """The momenta, in fractional coordinates, that op maps onto themselves
    up to a reciprocal lattice vector, among the fractions 0, 1/2, 1/3, 2/3,
    1/4, 3/4, 1/6 and 5/6 of each reciprocal lattice vector. They are found
    from the operation itself rather than from a table, so they follow the
    lattice vectors of the geometry"""
    c = compile_operation(h, op)
    dim = h.dimensionality
    out = []
    for f in np.ndindex(*([len(_FRACTIONS)]*dim)):
        k = np.zeros(3)
        k[:dim] = [_FRACTIONS[i] for i in f]
        d = c.k_image(k) - k
        if np.max(np.abs(d - np.round(d))) < 1e-8: out.append(k)
    return out


def symmetry_eigenvalues(h, op, k, nocc=None, tol=1e-6):
    """Eigenvalues of the operation op on the occupied states at the
    momentum k, the states below zero energy or the lowest nocc

    op: a SymmetryOperation (see rotation_operation, inversion_operation)
        or a CompiledSymmetry, verified against h
    k: fractional momentum, which op has to map onto itself up to a
        reciprocal lattice vector (see invariant_momenta)

    Returns a complex array sorted by phase. For a spinful Hamiltonian the
    rotation acts on the spin too, so a C_n has eigenvalues with
    lambda^n=-1. Raises if the occupied states do not span a representation
    of the operation, which happens when the gap closes at k or when nocc
    cuts through a degenerate multiplet."""
    c = compile_operation(h, op)
    dim = h.dimensionality
    kv = np.zeros(3)
    kv[:dim] = np.array(k, dtype=float)[:dim]
    d = c.k_image(kv) - kv
    if np.max(np.abs(d - np.round(d))) > 1e-8:
        raise ValueError("the operation "+c.op.name+" maps the momentum "
          +str(kv[:dim].tolist())+" onto "+str(c.k_image(kv)[:dim].tolist())
          +", which is not the same momentum, so it has no eigenvalues "
          +"there; use one of invariant_momenta(h,op)")
    hk = h.get_hk_gen()(kv)
    if hasattr(hk, "todense"): hk = hk.todense()
    hk = np.array(hk, dtype=complex)
    es, vs = np.linalg.eigh(hk)
    if nocc is None: U = vs[:, es < 0.]
    else:
        if not 0 < nocc <= len(es):
            raise ValueError("nocc must be between 1 and the "+str(len(es))
              +" states of the Hamiltonian, got "+str(nocc))
        U = vs[:, :nocc]
    if U.shape[1] == 0:
        raise ValueError("there are no occupied states at the momentum "
          +str(kv[:dim].tolist())+"; shift the Fermi level to zero or pass nocc")
    P, kp = c.orbital_operator(kv)
    A = U.conj().T@P@U # the operation in the occupied states
    err = np.max(np.abs(P@U - U@A))
    if err > tol:
        raise ValueError("the occupied states at the momentum "
          +str(kv[:dim].tolist())+" are not mapped onto themselves by "
          +c.op.name+" (error "+str(err)+"), so the gap closes there or "
          +"nocc cuts a degenerate multiplet")
    ev = np.linalg.eigvals(A)
    return ev[np.argsort(np.angle(ev))]


def chern_mod_n(h, n=None, center=(0., 0., 0.), nocc=None):
    """Chern number modulo n of a two-dimensional insulator with a C_n
    rotation about the z axis, from the rotation eigenvalues of the
    occupied states at the rotation-invariant momenta (Fang, Gilbert and
    Bernevig, arXiv:1207.5767):

        (-1)^C     = prod zeta(Gamma) zeta(X) zeta(Y) zeta(M)   (C2)
        i^C        = prod (-1)^F xi(Gamma) xi(M) zeta(Y)        (C4)
        e^{2 pi i C/3} = prod (-1)^F theta(Gamma) theta(K) theta(K')  (C3)
        e^{i pi C/3}   = prod (-1)^F eta(Gamma) theta(K) zeta(M)      (C6)

    with the product over the occupied bands, zeta, theta, xi and eta the
    eigenvalues of C2, C3, C4 and C6, and F=1 for a spinful Hamiltonian.

    n: 2, 3, 4 or 6; by default the largest of them that is a symmetry
    center: a point of the rotation axis, in Cartesian coordinates

    Returns an integer between 0 and n-1. It has the sign convention of
    h.get_chern(), so a Chern number -1 gives n-1."""
    if h.dimensionality != 2:
        raise ValueError("the Chern number modulo n needs a two-dimensional "
          +"Hamiltonian, and this one has dimensionality "
          +str(h.dimensionality))
    if n is None:
        for m in (6, 4, 3, 2):
            if compile_symmetry(h, rotation_operation(m, center)) is not None:
                n = m
                break
        else:
            raise ValueError("this Hamiltonian has no C2, C3, C4 or C6 "
              +"rotation about the z axis through "+str(list(center))
              +"; pass the center of the rotation")
    if n not in (2, 3, 4, 6):
        raise ValueError("n must be one of 2, 3, 4 or 6, got "+str(n))
    compile_operation(h, rotation_operation(n, center)) # raise if absent
    def fixed(m): # momenta fixed by C_m, Gamma first
        return invariant_momenta(h, rotation_operation(m, center))
    gamma = np.zeros(3)
    def is_in(k, ks): return any(np.allclose(k, q) for q in ks)
    if n == 2:
        points = [(2, k) for k in fixed(2)]
        expected = 4
    elif n == 3:
        points = [(3, k) for k in fixed(3)]
        expected = 3
    elif n == 4:
        k4, k2 = fixed(4), fixed(2)
        m = [k for k in k4 if not np.allclose(k, gamma)]
        y = [k for k in k2 if not is_in(k, k4)]
        points = [(4, gamma)] + [(4, k) for k in m] + [(2, y[0])]
        expected = 3 if (len(k4), len(k2)) == (2, 4) else -1
    else:
        k3, k2 = fixed(3), fixed(2)
        kk = [k for k in k3 if not np.allclose(k, gamma)]
        mm = [k for k in k2 if not np.allclose(k, gamma)]
        points = [(6, gamma), (3, kk[0]), (2, mm[0])]
        expected = 3 if (len(fixed(6)), len(k3), len(k2)) == (1, 3, 4) else -1
    if len(points) != expected:
        raise ValueError("found "+str(len(points))+" momenta invariant "
          +"under the rotations of C"+str(n)+" instead of the expected "
          +"ones; the rotation does not act on the lattice as a C"+str(n))
    val = 1.0 + 0j
    counts = {}
    for (m, k) in points:
        ev = symmetry_eigenvalues(h, rotation_operation(m, center), k,
                                  nocc=nocc)
        counts[tuple(np.round(k[:2], 4))] = len(ev)
        val *= np.prod(ev)
    _check_filling(counts)
    if h.has_spin and n != 2: # (-1)^F once per occupied band
        val *= (-1)**len(ev)
    x = np.angle(val)/(2*np.pi)*n # C modulo n, up to roundoff
    if abs(x - np.round(x)) > 1e-4:
        raise ValueError("the product of the rotation eigenvalues is "
          +str(val)+", not an n-th root of unity, so the occupied states "
          +"are not an insulator with this rotation")
    return int(np.round(x)) % n


def _check_filling(counts):
    """Raise unless the number of occupied states is the same at every
    momentum, which a metal breaks"""
    if len(set(counts.values())) != 1:
        raise ValueError("the number of occupied states changes between "
          +"the invariant momenta ("+", ".join(str(list(k))+": "+str(n)
          for (k, n) in counts.items())+"), so the Fermi level crosses a "
          +"band and this is not an insulator")


def inversion_centers(h):
    """Candidate inversion centers: an inversion maps the first site onto
    some site j in the cell R, so its center is the midpoint of r_0 and
    r_j + R, with R in -1, 0, 1 along each lattice vector"""
    g = h.geometry
    dim = h.dimensionality
    vecs = [np.array(v) for v in (g.a1, g.a2, g.a3)[:dim]]
    out = []
    for rj in g.r:
        for R in np.ndindex(*([3]*dim)):
            shift = sum([(R[i]-1)*vecs[i] for i in range(dim)], np.zeros(3))
            out.append((np.array(g.r[0]) + np.array(rj) + shift)/2.)
    return out


def parity_z2(h, center=None, nocc=None):
    """Z2 invariant of a time-reversal-symmetric insulator with inversion,
    from the parities of the occupied states at the time-reversal-invariant
    momenta (Fu and Kane, cond-mat/0611341): with n_- the number of
    occupied states of parity -1 at a momentum Gamma_i, even because of the
    Kramers pairs, delta_i = (-1)^(n_-/2), and (-1)^nu = prod_i delta_i.

    center: the inversion center, in Cartesian coordinates. By default it is
        searched among the midpoints of the first site and its images; the
        result does not depend on which inversion center is used, since
        moving the center by half a lattice vector flips the parities at an
        even number of the momenta in each product

    Returns nu (0 or 1) in two dimensions, and the strong and weak indices
    nu0, (nu1, nu2, nu3) in three, in the format of z2_invariant_3d."""
    from ..check import require_spin
    require_spin(h, "the parity Z2 invariant")
    dim = h.dimensionality
    if dim not in (2, 3):
        raise ValueError("the parity Z2 invariant needs a two- or "
          +"three-dimensional Hamiltonian, and this one has dimensionality "
          +str(dim))
    if not h.has_time_reversal_symmetry():
        raise ValueError("the parity Z2 invariant needs time-reversal "
          +"symmetry, which this Hamiltonian breaks; use chern_mod_n or "
          +"the Chern number instead")
    if center is None:
        for cen in inversion_centers(h):
            c = compile_symmetry(h, inversion_operation(cen))
            if c is not None: break
        else:
            raise ValueError("no inversion center was found among the "
              +"midpoints of the first site and its images, so this "
              +"Hamiltonian has no inversion symmetry")
    else: c = compile_operation(h, inversion_operation(center))
    delta = {}
    counts = {}
    for n in np.ndindex(*([2]*dim)):
        k = np.array(n)/2.
        xi = symmetry_eigenvalues(h, c, k, nocc=nocc)
        counts[tuple(k[:dim])] = len(xi)
        if np.max(np.abs(xi - np.round(xi.real))) > 1e-4:
            raise ValueError("the inversion eigenvalues at "+str(k.tolist())
              +" are not +1 or -1: "+str(xi))
        nm = int(np.sum(xi.real < 0.))
        if nm % 2 != 0:
            raise ValueError("the number of occupied states of parity -1 "
              +"at "+str(k.tolist())+" is odd, so the occupied states are "
              +"not Kramers pairs")
        delta[n] = (-1)**(nm//2)
    _check_filling(counts)
    nu = int(np.prod(list(delta.values())) < 0)
    if dim == 2: return nu
    weak = tuple(int(np.prod([d for (n, d) in delta.items() if n[i] == 1]) < 0)
                 for i in range(3))
    return (nu, weak)
