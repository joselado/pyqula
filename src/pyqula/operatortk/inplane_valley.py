import numpy as np
from .. import algebra
from .sharpen import get_sharpen

# The in-plane valley pseudospin (tau_x, tau_y) is built from a chiral
# Kekule coupling (kekule.chiral_kekule), the real-space analogue of the
# second-neighbor modified-Haldane coupling used for the out-of-plane
# valley operator (tau_z, see operatortk/valley.py): a first-neighbor bond
# texture that mixes the K and K' Dirac points.
#
# Unlike the Haldane term, a bare chiral Kekule coupling with a single
# (t1,t2) amplitude pair is never a pure tau_a operator: since it is built
# from first-neighbor (A-B) hoppings, its projection onto the low-energy
# valley x sublattice subspace is necessarily off-diagonal in sublattice
# (a combination of tau_x/tau_y dressed by sigma_x/sigma_y), never
# proportional to the sublattice identity. However tau_x and tau_y only
# need to share the *same* sublattice dressing to satisfy the expected
# pseudospin algebra ({tau_x,tau_y}=0, tau_x^2=tau_y^2=1, and both
# anticommute with tau_z regardless of dressing) - so the two independent
# complex amplitudes (t1,t2) of chiral_kekule give exactly enough freedom
# to solve for a pure tau_x (call it angle=0) and pure tau_y (angle=pi/2)
# operator, both dressed by the same sigma matrix.
#
# The (t1,t2) values are found by projecting the chiral Kekule Bloch
# matrix, evaluated at Gamma of a Kekule-commensurate (3x3) supercell of
# plain graphene, onto its 4-dimensional zero-energy subspace (the folded
# K,K' Dirac points), and solving the resulting (real-)linear map from
# (t1,t2) to the 4 allowed Pauli components (tau_x/y x sigma_x/y) for the
# combinations giving a pure tau_x and a pure tau_y. This calibration only
# depends on the honeycomb lattice/bond-direction conventions used by
# kekule.chiral_kekule, not on the geometry of the Hamiltonian the
# operator is eventually applied to, so it is computed once and cached.

_calibration_cache = {}


def _calibrate_taus():
    """Return the (t1,t2) chiral-Kekule amplitude pairs that give a pure
    tau_x and a pure tau_y valley pseudospin operator"""
    if "txty" in _calibration_cache: return _calibration_cache["txty"]
    from .. import geometry as geometrymod
    g3 = geometrymod.honeycomb_lattice().supercell(3) # Kekule-commensurate cell
    h0 = g3.get_hamiltonian(has_spin=False)
    h0.turn_multicell()
    hk0 = h0.get_hk_gen()([0.,0.,0.])
    (es,vs) = algebra.eigh(hk0)
    inds = np.where(np.abs(es)<1e-6)[0] # the folded K,K' zero modes
    V = vs[:,inds]
    hz = h0.copy() ; hz.clean() ; hz.add_modified_haldane(1.0/4.5)
    hkz = hz.get_hk_gen()([0.,0.,0.])
    Mz = V.conj().T@hkz@V
    (ez,uz) = np.linalg.eigh(Mz) # sort into the two valleys
    W = V@uz
    sub = np.diag(g3.sublattice.astype(np.complex128))
    def sublattice_sort(Wsub):
        Ms = Wsub.conj().T@sub@Wsub
        (es_s,us_s) = np.linalg.eigh(Ms) # sort into A,B sublattice
        return Wsub@us_s
    Wfull = np.concatenate([sublattice_sort(W[:,:2]),sublattice_sort(W[:,2:])],
            axis=1) # basis ordered as (valley,sublattice)
    sx = np.array([[0,1],[1,0]],dtype=np.complex128)
    sy = np.array([[0,-1j],[1j,0]],dtype=np.complex128)
    basis4 = [np.kron(sx,sx),np.kron(sy,sx),np.kron(sx,sy),np.kron(sy,sy)]
    def kekule_bloch(t1,t2):
        hk = h0.copy() ; hk.clean() ; hk.add_chiral_kekule(t1=t1,t2=t2)
        return hk.get_hk_gen()([0.,0.,0.])
    def coeffs(t1,t2):
        M = Wfull.conj().T@kekule_bloch(t1,t2)@Wfull
        return np.array([np.trace(M@b.conj().T)/4.0 for b in basis4])
    basis_inputs = [(1.0,0.0),(1j,0.0),(0.0,1.0),(0.0,1j)]
    Rmat = np.array([coeffs(t1,t2) for (t1,t2) in basis_inputs]).T
    Rinv = np.linalg.inv(Rmat)
    cx = Rinv@np.array([1.,0.,0.,0.]) # solve for pure tau_x sigma_x
    cy = Rinv@np.array([0.,1.,0.,0.]) # solve for pure tau_y sigma_x
    t1x,t2x = cx[0]+1j*cx[1], cx[2]+1j*cx[3]
    t1y,t2y = cy[0]+1j*cy[1], cy[2]+1j*cy[3]
    out = (t1x,t2x,t1y,t2y)
    _calibration_cache["txty"] = out
    return out


def add_valley_exchange(h,v):
    """Add a valley-space exchange term v=(vx,vy,vz).(tau_x,tau_y,tau_z)
    to the Hamiltonian, the valley-pseudospin analogue of add_exchange
    for real spin. vz is added as a modified-Haldane (second-neighbor)
    coupling and vx,vy as a chiral Kekule (first-neighbor) coupling,
    both calibrated so that they match get_operator("valley"/"valley_x"/
    "valley_y"): e.g. add_valley_exchange([0,0,d]) opens the same gap as
    get_operator("valley") would measure as +-d.

    Just like the valley operators themselves, the Kekule (vx,vy) part is
    only periodic on a Kekule-commensurate geometry (e.g. h.geometry
    already a 3x3, or other multiple-of-3, supercell of the primitive
    honeycomb cell) for dimensionality>0 Hamiltonians; for 0-dimensional
    (finite flake) Hamiltonians no such commensurability is required."""
    if not h.geometry.has_sublattice: raise ValueError(
            "valley exchange requires a honeycomb-like geometry with a"
            " sublattice index")
    (vx,vy,vz) = v
    h.turn_multicell()
    if vz!=0.: h.add_modified_haldane(vz/4.5)
    if vx!=0. or vy!=0.:
        (t1x,t2x,t1y,t2y) = _calibrate_taus()
        t1 = vx*t1x + vy*t1y
        t2 = vx*t2x + vy*t2y
        h.add_chiral_kekule(t1=t1,t2=t2)


def get_inplane_valley(h,angle=0.0,delta=None,**kwargs):
    """Return a callable that calculates the in-plane valley (tau_x/tau_y)
    expectation value, using a chiral Kekule coupling. angle=0 gives
    tau_x, angle=pi/2 gives tau_y, any other angle a linear combination
    (a rotation of the in-plane pseudospin axis).

    Just like the out-of-plane valley operator (get_valley), the chiral
    Kekule coupling this is built from is only periodic on a
    Kekule-commensurate geometry (e.g. h.geometry already a 3x3, or other
    multiple-of-3, supercell of the primitive honeycomb cell) for
    dimensionality>0 Hamiltonians; for 0-dimensional (finite flake)
    Hamiltonians no such commensurability is required."""
    if not h.geometry.has_sublattice: raise ValueError(
            "in-plane valley operator requires a honeycomb-like geometry"
            " with a sublattice index")
    (t1x,t2x,t1y,t2y) = _calibrate_taus()
    t1 = np.cos(angle)*t1x + np.sin(angle)*t1y
    t2 = np.cos(angle)*t2x + np.sin(angle)*t2y
    ho = h.copy() # copy Hamiltonian
    ho.turn_multicell()
    ho.clean() # set to zero
    ho.add_chiral_kekule(t1=t1,t2=t2) # add the calibrated Kekule coupling
    hkgen = ho.get_hk_gen() # get generator for the hk Hamiltonian
    sharpen = get_sharpen(delta=delta) # renormalize eigenvalues to +-1
    def fun(m=None,k=None):
        if h.dimensionality>0 and k is None: raise # requires a kpoint
        hk = hkgen(k) # evaluate Hamiltonian
        hk = sharpen(hk) # sharpen the valley
        if m is None: return hk # just return the valley operator
        else: return hk@m # return the projector
    if h.dimensionality==0: return fun() # return a matrix
    return fun # return function


def get_valley_taux(h,**kwargs):
    """Return the tau_x valley operator"""
    return get_inplane_valley(h,angle=0.0,**kwargs)


def get_valley_tauy(h,**kwargs):
    """Return the tau_y valley operator"""
    return get_inplane_valley(h,angle=np.pi/2,**kwargs)
