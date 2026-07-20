import numpy as np
from .. import algebra
from .. import kekule
from .sharpen import get_sharpen

# The in-plane valley pseudospin (tau_x, tau_y) is built from a chiral
# Kekule coupling (kekule.chiral_kekule), the real-space analogue of the
# second-neighbor modified-Haldane coupling used for the out-of-plane
# valley operator (tau_z, see operatortk/valley.py): a first-neighbor bond
# texture that mixes the K and K' Dirac points.
#
# A single, uniform chiral-Kekule texture only keeps 1/3 of all hexagons
# as "Kekule centers" (see kekule.kekule_registries): every atom ends up
# with exactly 2 of its 3 bonds "active", an intrinsically lopsided
# pattern that is only exactly C3-symmetric around the special retained
# hexagon-center points, never around a generic atomic site. Used
# directly as a measurement operator this leaves a lattice-scale
# artifact in any measured texture (e.g. it broke the C3 symmetry
# expected around a point defect in examples/2d/valley_vortex).
#
# The honeycomb lattice actually has 3 inequivalent such registries
# (translates of each other by one hexagon-lattice step,
# kekule.kekule_registries returns all 3). A C3 rotation about any atom
# permutes the 3 registries cyclically (verified numerically: rotating
# the default registry's Bloch matrix by 120 deg about an arbitrary
# atom reproduces a DIFFERENT registry's matrix exactly). That makes
# T = H(registry0) + w*H(registryA) + w^2*H(registryB), w=exp(i 2 pi/3),
# transform under that same rotation as T -> w*T (a pure phase, exact to
# machine precision) -- i.e. T is a proper complex "tau_x + i*tau_y"-like
# object, and its Hermitian real/imaginary parts
#   tau_x = (T + T^dagger)/2,  tau_y = (T - T^dagger)/(2i)
# are two EXACTLY C3-covariant, Hermitian operators, by construction,
# with no free parameter and no dependence on which registry happens to
# be labeled "the" default one (registryA/B are derived analytically,
# not by an arbitrary search).
#
# A single (t1,t2) chiral-Kekule amplitude pair still needs to be
# calibrated so that tau_x, built this way, is *purely* tau_x (not
# mixed with tau_z or with extraneous sublattice dressing) when
# restricted to the low-energy K,K' subspace -- this is the same
# calibration idea used before (projecting onto the folded K,K'
# zero-energy subspace of a Kekule-commensurate (3x3) supercell of
# graphene at Gamma), just re-solved for the new construction. Only one
# calibration constant pair is needed now (not two): tau_y is already
# the Hermitian partner of tau_x from the same (t1,t2), not an
# independently-calibrated operator.

_calibration_cache = {}


def _calibrate_tau():
    """Return the (t1,t2) chiral-Kekule amplitude pair that gives a pure
    tau_x (and, as its Hermitian partner, tau_y) valley pseudospin
    operator under the 3-registry construction"""
    if "t1t2" in _calibration_cache: return _calibration_cache["t1t2"]
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
    (cs0,csA,csB) = kekule.kekule_registries(g3)
    w = np.exp(1j*2.*np.pi/3.)
    def taux_bloch(t1,t2):
        def hk_of(cs):
            hk = h0.copy() ; hk.clean()
            fun = kekule.chiral_kekule(hk.geometry,t1=t1,t2=t2,registry=cs)
            hk.add_hopping_matrix(kekule.bond_function_to_matrix(fun))
            return hk.get_hk_gen()([0.,0.,0.])
        T = hk_of(cs0) + w*hk_of(csA) + w**2*hk_of(csB)
        return (T+T.conj().T)/2.
    sx = np.array([[0,1],[1,0]],dtype=np.complex128)
    sy = np.array([[0,-1j],[1j,0]],dtype=np.complex128)
    basis4 = [np.kron(sx,sx),np.kron(sy,sx),np.kron(sx,sy),np.kron(sy,sy)]
    def coeffs(t1,t2):
        M = Wfull.conj().T@taux_bloch(t1,t2)@Wfull
        return np.array([np.trace(M@b.conj().T)/4.0 for b in basis4])
    basis_inputs = [(1.0,0.0),(1j,0.0),(0.0,1.0),(0.0,1j)]
    Rmat = np.array([coeffs(t1,t2) for (t1,t2) in basis_inputs]).T
    Rinv = np.linalg.inv(Rmat)
    c = Rinv@np.array([1.,0.,0.,0.]) # solve for pure tau_x sigma_x
    t1,t2 = c[0]+1j*c[1], c[2]+1j*c[3]
    out = (t1,t2)
    _calibration_cache["t1t2"] = out
    return out


def _build_tau_T_gen(h,t1,t2):
    """Return hkgen_T(k), a k-dependent generator for the
    3-registry-symmetrized complex combination T = tau_x + i*tau_y (see
    module docstring). The 3 registry Hamiltonians (and their
    k-generators) are built once, up front -- not on every call --
    since hkgen_T is typically evaluated at many k-points (e.g.
    get_bands sweeping a k-path) and rebuilding the whole multicell
    Hamiltonian (re-painting every bond) per k-point would defeat the
    point of caching the registries."""
    (cs0,csA,csB) = kekule.kekule_registries(h.geometry)
    w = np.exp(1j*2.*np.pi/3.)
    def make_hkgen(cs):
        ho = h.copy() ; ho.turn_multicell() ; ho.clean()
        fun = kekule.chiral_kekule(ho.geometry,t1=t1,t2=t2,registry=cs)
        ho.add_hopping_matrix(kekule.bond_function_to_matrix(fun))
        return ho.get_hk_gen()
    (hk0,hkA,hkB) = (make_hkgen(cs0),make_hkgen(csA),make_hkgen(csB))
    def hkgen_T(k):
        return hk0(k) + w*hkA(k) + w**2*hkB(k)
    return hkgen_T


def add_valley_exchange(h,v):
    """Add a valley-space exchange term v=(vx,vy,vz).(tau_x,tau_y,tau_z)
    to the Hamiltonian, the valley-pseudospin analogue of add_exchange
    for real spin. vz is added as a modified-Haldane (second-neighbor)
    coupling; vx,vy are added as a chiral Kekule (first-neighbor)
    coupling, symmetrized over the 3 inequivalent Kekule registries
    (see module docstring) so the added term is exactly C3-covariant
    about every atom, not just about the special retained-hexagon-center
    points a single uniform Kekule texture would be tied to.

    Just like the valley operators themselves (get_operator("valley"/
    "valley_x"/"valley_y")), this is only periodic on a
    Kekule-commensurate geometry (e.g. h.geometry already a 3x3, or
    other multiple-of-3, supercell of the primitive honeycomb cell) for
    dimensionality>0 Hamiltonians; for 0-dimensional (finite flake)
    Hamiltonians no such commensurability is required."""
    if not h.geometry.has_sublattice: raise ValueError(
            "valley exchange requires a honeycomb-like geometry with a"
            " sublattice index")
    (vx,vy,vz) = v
    h.turn_multicell()
    if vz!=0.: h.add_modified_haldane(vz/4.5)
    if vx!=0. or vy!=0.:
        (t1,t2) = _calibrate_tau()
        (cs0,csA,csB) = kekule.kekule_registries(h.geometry)
        w = np.exp(1j*2.*np.pi/3.)
        def registry_multihopping(cs):
            ho = h.copy() ; ho.turn_multicell() ; ho.clean()
            fun = kekule.chiral_kekule(ho.geometry,t1=t1,t2=t2,registry=cs)
            ho.add_hopping_matrix(kekule.bond_function_to_matrix(fun))
            return ho.get_multihopping()
        mhT = registry_multihopping(cs0) + w*registry_multihopping(csA) \
                + w**2*registry_multihopping(csB)
        mhTd = mhT.get_dagger()
        mh_taux = (mhT+mhTd)*0.5
        mh_tauy = (mhT-mhTd)*(-0.5j)
        mh = vx*mh_taux + vy*mh_tauy
        h.set_multihopping(h.get_multihopping()+mh)


def get_inplane_valley(h,angle=0.0,delta=None,**kwargs):
    """Return a callable that calculates the in-plane valley (tau_x/tau_y)
    expectation value, using a chiral Kekule coupling symmetrized over
    the 3 inequivalent Kekule-hexagon registries (see module docstring)
    so the result is exactly C3-covariant about every atom. angle=0
    gives tau_x, angle=pi/2 gives tau_y, any other angle a linear
    combination (a rotation of the in-plane pseudospin axis).

    Just like the out-of-plane valley operator (get_valley), the chiral
    Kekule coupling this is built from is only periodic on a
    Kekule-commensurate geometry (e.g. h.geometry already a 3x3, or
    other multiple-of-3, supercell of the primitive honeycomb cell) for
    dimensionality>0 Hamiltonians; for 0-dimensional (finite flake)
    Hamiltonians no such commensurability is required."""
    if not h.geometry.has_sublattice: raise ValueError(
            "in-plane valley operator requires a honeycomb-like geometry"
            " with a sublattice index")
    (t1,t2) = _calibrate_tau()
    hkgen_T = _build_tau_T_gen(h,t1,t2)
    sharpen = get_sharpen(delta=delta) # renormalize eigenvalues to +-1
    phase = np.exp(-1j*angle) # tau(angle) = Re(e^{-i*angle} T), see below
    def fun(m=None,k=None):
        if h.dimensionality>0 and k is None: raise # requires a kpoint
        T = phase*hkgen_T(k) # evaluate Hamiltonian
        hk = (T+T.conj().T)/2. # cos(angle)*tau_x + sin(angle)*tau_y
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
