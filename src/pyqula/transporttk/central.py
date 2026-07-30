import numpy as np
from copy import deepcopy
from ..algebra import dagger


def get_central_heterostructure(h,i=0,j=None,left=None,right=None,**kwargs):
    """Build a two-terminal Heterostructure using a finite Hamiltonian `h`
    as the central scattering region, with two semi-infinite 1D chain
    leads attached at sites `i` and `j` (0-indexed, following the same
    site convention as htk.extract.local_hamiltonian). `left`/`right`
    default to a plain spinless chain (geometry.chain().get_hamiltonian())
    promoted to match whatever has_spin/has_eh combination is needed; pass
    a Hamiltonian with nonzero pairing (e.g. via add_swave) as one of them
    (or make `h` itself superconducting) for a normal-superconductor
    junction -- at most one of {h, left, right} may carry pairing, see
    _num_superconducting below. Returns a Heterostructure, so all its
    existing methods apply unmodified: .landauer() for normal transport,
    .didv() (auto-dispatches to the BdG/smatrix formula when has_eh is
    True), .get_dos(), .get_kappa(), etc.

    Known limitation: .get_dos(operator=...)/device_dos's "central" mode
    projects with `ht.Hr.get_operator(operator)`, sized to the *lead's*
    unit cell -- correct for heterostructures.build's own constructors,
    where the central region is stacked out of lead-sized cells, but not
    generally the right shape here, where the central region (`h`) can be
    any size. Plain `.get_dos()` (no operator) is unaffected, since it
    only traces the central Green's function block.

    Dispatches on h.dimensionality; only 0d (finite) central regions are
    supported so far -- the dispatch exists so this stays the single
    public entry point when 1d/2d central regions are added later."""
    if h.dimensionality==0:
        return _central_heterostructure_0d(h,i=i,j=j,left=left,right=right,**kwargs)
    raise NotImplementedError("get_central_heterostructure is only "
            "implemented for 0d (finite) Hamiltonians so far, got "
            "dimensionality="+str(h.dimensionality))


class _Flags:
    """Stand-in object exposing just the has_spin/has_eh attributes that
    htk.mode.make_compatible reads off its second argument -- lets three
    objects (the central Hamiltonian and the two leads) be promoted to a
    common basis in one shot, instead of make_compatible's usual pairwise,
    order-dependent chaining (see heterostructures.build's
    h1=make_compatible(h1,h2); h2=make_compatible(h2,h1), which only works
    because it has exactly two objects to reconcile)."""
    def __init__(self,has_spin,has_eh):
        self.has_spin,self.has_eh = has_spin,has_eh


def _promote_all(h,left,right,maxiter=4):
    """Promote h, left and right to a common has_spin/has_eh basis.

    A single OR-and-promote pass is not enough: make_compatible's
    turn_nambu() has no way to produce a purely spinless Nambu object from
    a plain spinless one (its "spinless" branch always calls
    turn_spinful() first, same as heterostructures.build's own pairwise
    make_compatible calls end up doing), so promoting a spinless normal
    lead against a spinless_nambu central region can silently pull
    has_spin=True in on that one pass, changing what the OR target should
    have been. Iterate to a fixed point instead, recomputing the target
    from the actual post-promotion flags each round -- mirrors what
    build()'s own two-object h1=make_compatible(h1,h2);
    h2=make_compatible(h2,h1) sequence relies on to converge, generalized
    to three objects."""
    from ..htk.mode import make_compatible
    h2,left2,right2 = h,left,right
    for _ in range(maxiter):
        target = _Flags(
            has_spin = h2.has_spin or left2.has_spin or right2.has_spin,
            has_eh = h2.has_eh or left2.has_eh or right2.has_eh,
        )
        h3 = make_compatible(h2,target)
        left3 = make_compatible(left2,target)
        right3 = make_compatible(right2,target)
        converged = ((h3.has_spin,h3.has_eh)==(h2.has_spin,h2.has_eh) and
                     (left3.has_spin,left3.has_eh)==(left2.has_spin,left2.has_eh) and
                     (right3.has_spin,right3.has_eh)==(right2.has_spin,right2.has_eh))
        h2,left2,right2 = h3,left3,right3
        if converged: break
    else:
        raise RuntimeError("could not reconcile has_spin/has_eh between "
                "the central Hamiltonian and the two leads")
    # turn_spinful()/turn_nambu() can hand back sparse matrices even when
    # every input was dense (see e.g. increase_hilbert.spinful) -- force
    # dense again since this whole builder is dense-only (v1)
    return (h2.get_dense(),
            left2.get_no_multicell().get_dense(),
            right2.get_no_multicell().get_dense())


def _default_lead():
    from ..geometry import chain
    return chain().get_hamiltonian(has_spin=False)


def _num_superconducting(*hs):
    """How many of the given (already has_eh-promoted) Hamiltonians carry
    an actual nonzero pairing amplitude, as opposed to merely being
    written in the Nambu basis with zero pairing."""
    n = 0
    for hi in hs:
        if not hi.has_eh: continue
        if not hi.get_anomalous_hamiltonian().is_zero(): n += 1
    return n


def _embed_coupling(h,site_index,block):
    """Nc x n_lead matrix, zero except the dof-rows of `site_index`,
    filled with `block` (whose row count must equal that site's dof)."""
    from ..htk.extract import site_slice
    Nc = h.intra.shape[0]
    n_lead = block.shape[1]
    out = np.zeros((Nc,n_lead),dtype=np.complex128)
    out[site_slice(h,site_index),:] = block
    return out


def _central_heterostructure_0d(h,i=0,j=None,left=None,right=None):
    h = h.get_dense()
    if left is None: left = _default_lead()
    else: left = left.get_dense()
    if right is None: right = _default_lead()
    else: right = right.get_dense()
    from ..htk.extract import site_dof
    nsites = h.intra.shape[0]//site_dof(h)
    if j is None: j = nsites-1
    if not (0<=i<nsites):
        raise ValueError("site index i="+str(i)+" out of range for "
                          +str(nsites)+" sites")
    if not (0<=j<nsites):
        raise ValueError("site index j="+str(j)+" out of range for "
                          +str(nsites)+" sites")
    h2,left2,right2 = _promote_all(h,left,right)
    if _num_superconducting(h2,left2,right2)>1:
        raise ValueError("get_central_heterostructure supports at most "
                "one superconducting source among the central Hamiltonian "
                "and the two leads")
    from ..heterostructures import Heterostructure
    ht = Heterostructure() # bare object, populate by hand
    ht.right_intra,ht.right_inter = right2.intra.copy(),right2.inter.copy()
    ht.left_intra,ht.left_inter = left2.intra.copy(),dagger(left2.inter).copy()
    ht.central_intra = h2.intra.copy()
    ht.block_diagonal = False
    ht.has_spin,ht.has_eh = h2.has_spin,h2.has_eh
    ht.get_eh_sector = h2.get_eh_sector
    ht.central_geometry = deepcopy(h2.geometry)
    ht.left_coupling = _embed_coupling(h2,i,dagger(left2.inter))
    ht.right_coupling = _embed_coupling(h2,j,right2.inter)
    ht.Hl = left2.copy() # lead Hamiltonians -- surface_dos/get_dos/get_kappa/
    ht.Hr = right2.copy() # didv's SC-detection all read these directly
    return ht
