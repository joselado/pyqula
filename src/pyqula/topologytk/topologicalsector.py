import numpy as np
from .. import algebra

# Return topological objects in a sector of an operator #

def get_berry_curvature_operator_sector(H,operator=None,sector=1.0,
        nocc=None,
        **kwargs):
    """Return the Berry curvature in one sector"""
    from .occstates import occ_states_sector_generator
    H = H.copy() # make a copy
    H.os_gen = occ_states_sector_generator(H,operator=operator,
            sector=sector,nocc=nocc)
    return H.get_berry_curvature(**kwargs)


def get_chern_operator_sector(H,operator=None,sector=1.0,
        nocc=None,
        **kwargs):
    """Return the Chern number in one operator sector"""
    from .occstates import occ_states_sector_generator
    H = H.copy() # make a copy
    H.os_gen = occ_states_sector_generator(H,operator=operator,
            sector=sector,nocc=nocc)
    from ..topology import mesh_chern
    return mesh_chern(H,**kwargs)




def _hermitian_operator(H,operator):
    """The operator as a callable (v,k) -> O(k)@v, from a name, a matrix, an
    Operator or such a callable"""
    from ..topology import get_operator
    op = get_operator(H,operator)
    if op is None:
        raise ValueError("splitting the occupied states needs an operator")
    return op


def get_chern_operator_sign_sector(H,operator=None,sign=1,nk=40,**kwargs):
    """Chern number of the occupied states on which the operator is positive
    (sign=1) or negative (sign=-1)

    At each k-point the operator is diagonalized inside the occupied
    states, P O P, and the states with an eigenvalue of the chosen sign
    are kept (Prodan, arXiv:0904.1894). The operator need not commute with
    the Hamiltonian, only the spectrum of P O P has to stay away from zero,
    see operator_gap."""
    from .occstates import filter_state,states_generator
    from ..topology import mesh_chern
    if sign not in (1,-1):
        raise ValueError("sign must be 1 or -1, got "+str(sign))
    op = _hermitian_operator(H,operator)
    H = H.copy() # make a copy
    focc = filter_state(H.get_operator("energy"),accept=lambda e: e<0.)
    fop = filter_state(op,accept=lambda e: sign*e>0.)
    H.os_gen = states_generator(H,filt=fop*focc)
    return mesh_chern(H,nk=nk,**kwargs)


def operator_gap(H,operator=None,nk=40):
    """Smallest absolute eigenvalue of P O P over a k-mesh, the gap that
    separates the two sectors of get_chern_operator_sign_sector"""
    from .occstates import occupied_states
    from .. import klist
    op = _hermitian_operator(H,operator)
    hkgen = H.get_hk_gen()
    n = H.intra.shape[0]
    gap = np.inf
    sizes = set() # number of occupied states at each k-point
    for k in klist.kmesh(H.dimensionality,nk=nk):
        w = occupied_states(hkgen,k) # rows are the conjugated states
        sizes.add(len(w))
        if len(sizes)>1:
            raise ValueError("the number of occupied states changes over the "
              +"Brillouin zone (between "+str(min(sizes))+" and "
              +str(max(sizes))+"), so the occupied manifold is not "
              +"separated by a gap and its Chern numbers are not defined")
        if len(w)==0: continue
        o = np.array(op(np.identity(n,dtype=complex),k=k))
        m = np.conjugate(w)@o@w.T # P O P in the occupied states
        gap = min(gap,np.min(np.abs(np.linalg.eigvalsh((m+m.conj().T)/2.))))
    return gap


def _band_extremum(hkgen,d,ks,es,ib,sign,nstart=3):
    """Largest (sign=1) or smallest (sign=-1) energy of band ib over the
    continuous Brillouin zone, as (energy,k), polished with a local
    optimization started from the nstart best points of the mesh"""
    from scipy.optimize import minimize
    def f(k): # the band, sign-flipped so that its extremum is a minimum
        kv = np.zeros(3) ; kv[:d] = k
        return -sign*np.sort(algebra.eigvalsh(hkgen(kv)))[ib]
    best = (np.inf,None)
    for i in np.argsort(-sign*es[:,ib])[:nstart]: # best points of the mesh
        r = minimize(f,np.array(ks[i])[:d],method="Nelder-Mead",
                     options={"xatol":1e-8,"fatol":1e-10})
        if r.fun<best[0]: best = (r.fun,r.x)
    return -sign*best[0],best[1]


def _check_insulator(H,nk=40):
    """Raise if the occupied states, the ones below zero energy, are not
    separated from the empty ones over the whole Brillouin zone.

    A constant number of occupied states on the k-mesh is not enough: a
    Fermi pocket that falls between the points of a coarse mesh leaves it
    unchanged, and the split Chern numbers then came out as non-integers,
    or as a wrong integer. So the top of the highest occupied band and the
    bottom of the lowest empty one are followed over the continuous k,
    starting from the extrema of the mesh"""
    from .. import klist
    hkgen = H.get_hk_gen()
    d = H.dimensionality
    ks = klist.kmesh(d,nk=nk)
    es = np.array([np.sort(algebra.eigvalsh(hkgen(k))) for k in ks])
    nocc = set(np.sum(es<0.,axis=1)) # occupied states at each k-point
    if len(nocc)>1: return # operator_gap reports it
    nocc = nocc.pop()
    edges = [] # (band, energy, k) that are on the wrong side of zero
    if nocc>0: # top of the highest occupied band
        e,k = _band_extremum(hkgen,d,ks,es,nocc-1,1)
        if e>=0.: edges.append((nocc-1,e,k))
    if nocc<es.shape[1]: # bottom of the lowest empty band
        e,k = _band_extremum(hkgen,d,ks,es,nocc,-1)
        if e<=0.: edges.append((nocc,e,k))
    if edges:
        (ib,e,k) = edges[0]
        raise ValueError("the number of occupied states changes over the "
          +"Brillouin zone between the points of the k-mesh (nk="+str(nk)
          +"): band "+str(ib)+" reaches E="+str(e)+" at k="
          +str(np.round(k,4))+", on the other side of the Fermi energy at "
          +"zero, so this is a metal and its Chern numbers are not defined")


def split_chern(H,operator=None,nk=40,tol=1e-4):
    """(C_+ - C_-)/2 for the occupied states split by the sign of P O P,
    raising if P O P closes its gap on the mesh, or if the Hamiltonian is a
    metal, with a Fermi pocket between the points of the mesh"""
    gap = operator_gap(H,operator=operator,nk=nk)
    _check_insulator(H,nk=nk)
    if gap<tol:
        raise ValueError("the occupied states cannot be split by the sign "
          +"of the operator, since its projection on them has an eigenvalue "
          +"of "+str(gap)+" on the k-mesh; the split invariant is not "
          +"defined when that gap closes")
    cp = get_chern_operator_sign_sector(H,operator,sign=1,nk=nk)
    cm = get_chern_operator_sign_sector(H,operator,sign=-1,nk=nk)
    return (cp-cm)/2.


def spin_chern(H,operator="sz",nk=40,tol=1e-4):
    """Spin Chern number C_s = (C_+ - C_-)/2 of a two-dimensional insulator,
    with the occupied states split by the sign of their spin, P s_z P
    (Sheng-Weng-Sheng-Haldane, cond-mat/0603054; Prodan, arXiv:0904.1894)

    It is the difference of the Chern numbers of the two spin sectors when
    s_z is conserved, and stays quantized when it is not, as long as the
    spectrum of P s_z P keeps a gap around zero, which is checked; with time
    reversal, C_s modulo 2 is the Z2 invariant.

    operator: the spin component, a name, a matrix or an Operator
    nk: k-points per direction of the mesh"""
    from ..check import require_spin
    require_spin(H,"the spin Chern number")
    if H.dimensionality!=2:
        raise ValueError("the spin Chern number needs a two-dimensional "
          +"Hamiltonian, and this one has dimensionality "
          +str(H.dimensionality))
    return split_chern(H,operator=operator,nk=nk,tol=tol)


def mirror_operator(H):
    """The mirror z -> -z about the middle plane of the geometry, as a
    Hermitian operator: M for a spinless Hamiltonian, where M^2=1, and iM
    for a spinful one, where M^2=-1. Its eigenvalue +1 is M=+1 spinless and
    M=-i spinful. Raises if the Hamiltonian has no such mirror"""
    from ..symmetrytk.pointgroup import SymmetryOperation,compile_symmetry
    from ..operators import Operator
    z = np.array(H.geometry.r)[:,2]
    center = (0.,0.,(np.max(z)+np.min(z))/2.)
    M = compile_symmetry(H,SymmetryOperation(np.diag([1.,1.,-1.]),
                             center=center))
    if M is None:
        raise ValueError("this Hamiltonian has no mirror symmetry z -> -z "
          +"about the plane z="+str(center[2])+", so it has no mirror Chern "
          +"number; a Rashba coupling or an in-plane field breaks it")
    fac = 1j if H.has_spin else 1.
    def f(v,k=None):
        kv = np.zeros(3)
        if k is not None: kv[:len(k)] = np.array(k)[:3]
        return fac*M.orbital_operator(kv)[0]@v
    return Operator(f)


def mirror_chern(H,nk=40):
    """Mirror Chern number C_M = (C_{+i} - C_{-i})/2 of a two-dimensional
    insulator with the mirror z -> -z, which maps every momentum of the
    plane onto itself, so that the occupied states split into the two
    mirror sectors at every k (Teo-Fu-Kane, arXiv:0804.2664); for a
    spinless Hamiltonian the sectors are M=+1 and M=-1.

    The mirror is found and verified by symmetrytk.pointgroup. For a single
    layer it is -i sigma_z, so C_M is minus the spin Chern number. It is
    defined without time reversal, so it stays quantized under an
    out-of-plane exchange field that keeps the mirror."""
    if H.dimensionality!=2:
        raise ValueError("the mirror Chern number needs a two-dimensional "
          +"Hamiltonian, and this one has dimensionality "
          +str(H.dimensionality))
    op = mirror_operator(H)
    c = split_chern(H,operator=op,nk=nk)
    return -c if H.has_spin else c # iM=+1 is M=-i
