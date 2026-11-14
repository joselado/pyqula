import numpy as np
from .. import check
from .sharpen import get_sharpen
from .valley import valley_coupling

# The in-plane valley pseudospin (tau_x, tau_y) is a plane wave on the
# first-neighbor bonds of the honeycomb lattice,
#
#   T = tau_x + i tau_y ,   T_ij = (2/3) exp(i Q.(r_i+r_j)/2) ,   Q = K-K',
#
# for every pair of first neighbors i,j, with (r_i+r_j)/2 the midpoint of
# the bond. A term that takes an electron from K' to K has to carry the
# momentum Q, and this is the simplest one on the first neighbors: the
# Kekule bond order. Its Hermitian parts are two real symmetric hoppings,
#
#   tau_x = (2/3) cos(Q.r_mid) ,   tau_y = (2/3) sin(Q.r_mid) ,
#
# the first-neighbor counterpart of the second-neighbor modified Haldane
# coupling that gives tau_z (operatortk/valley.py). What the closed form
# guarantees:
#
# - The three bonds of a site carry phases 120 degrees apart, so they add
#   to zero: a pristine lattice has no in-plane valley, and T picks up a
#   pure phase under a C3 rotation about any site.
# - On the Dirac points tau_x, tau_y and tau_z square to one, anticommute,
#   and tau_x tau_y = i tau_z; the 2/3 is what makes the squares one, so
#   v.tau opens a gap |v|.
# - tau_x and tau_y are real, as time reversal requires for a spinless
#   real Hamiltonian: it exchanges K and K', and leaves tau_x and tau_y.
# - A finite system gets the operator of the infinite lattice restricted
#   to the bonds it has, so equivalent edges are treated alike.
#
# Two conventions fix what the closed form leaves free:
#
# - K is taken from the bond that goes from a B site to its A neighbor
#   (sublattice +1), rotated by 90 degrees. The sublattice labels fix the
#   sign of tau_z in the same way, so the three operators are right-handed
#   whatever the order of the atoms. Each bond finds K from its own
#   direction, so every layer of a multilayer gets the K of its own
#   orientation and labels.
# - The phase is measured from the site of the honeycomb lattice that is
#   closest to the origin of coordinates, whether an atom sits there or
#   not (how equally close sites are told apart is in _lattice_origins).
#   A small shift of the geometry then changes nothing, and a shift by a
#   lattice vector rotates (tau_x,tau_y) by 120 degrees: a translation is
#   a rotation of the in-plane valley. Two geometries with nearly the
#   same coordinates (a flake and the same flake with a vacancy,
#   recentered or not) share the same axes.
#
# This replaced a construction that summed a chiral Kekule coupling over
# the three Kekule registries and calibrated its two amplitudes
# numerically. In the bulk it was the same operator to 1e-3, but the
# calibration target was written in a basis whose phases came from the
# diagonalization (another LAPACK driver gave another operator), the
# bonds at the edge of a finite system missed the hexagons outside it,
# and the sign of tau_y followed the order of the atoms.


# a first-neighbor bond whose in-plane length is below this, in units of
# the bond length, is a vertical bond between layers, not a honeycomb bond
_VERTICAL = 1e-3


def _first_neighbor_pairs(rs1,rs2):
    """Return the indices (ii,jj) of the pairs rs1[i],rs2[j] at distance 1
    that have an in-plane part. A vertical bond of the same length, between
    two layers stacked on top of each other (specialgeometry.multilayer with
    its default dz, or the bilayers of a (111) diamond film), has no
    in-plane direction to give K, and the plane wave lives on the bonds of
    each layer, so it is left out, as a longer interlayer bond is"""
    from scipy.spatial import cKDTree
    near = cKDTree(rs1).query_ball_tree(cKDTree(rs2),np.sqrt(1.01))
    ii = np.repeat(np.arange(len(rs1)),[len(x) for x in near])
    jj = np.array([j for x in near for j in x],dtype=int)
    dr = rs1[ii] - rs2[jj]
    keep = np.sum(dr*dr,axis=1)>0.99 # drop the pairs closer than a bond
    keep &= np.hypot(dr[:,0],dr[:,1])>_VERTICAL # and the vertical bonds
    return ii[keep],jj[keep]


# how the lattice site that the phase is measured from is chosen, see
# _lattice_origins: in units of the bond length, and in radians
_ORIGIN_WINDOW = 0.3
_ORIGIN_DIRECTION = 0.1

_WHAT = "the in-plane valley operators (valley_x, valley_y) and the " \
        "in-plane part (vx,vy) of add_valley_exchange"


def _bond_wavevectors(r1,r2,sub1,sub2):
    """Return Q = K-K' for each of the bonds r1[i],r2[i], whose ends are
    on the sublattices sub1[i],sub2[i]. K is the in-plane direction of the
    bond from the B site to the A site (sublattice +1) rotated by 90
    degrees, of length 4 pi/(3 sqrt(3) a) for an in-plane bond length a;
    of the three bonds of a site, which give equivalent K points, the one
    with the smallest polar angle is used. Each bond finds K from its own
    direction, so that neither the order of the atoms nor the other layers
    of the geometry enter."""
    if np.any(sub1*sub2>0): raise ValueError(_WHAT+" need a honeycomb "
            "lattice, whose first neighbors are on opposite sublattices, "
            "and this geometry has first neighbors on the same one")
    dr = np.sign(sub1)[:,None]*(r1-r2) # the bond from B to A
    zs = dr[:,0] + 1j*dr[:,1]
    rot = np.exp(2j*np.pi/3.*np.arange(3))
    z0 = _smallest_polar_angle(zs[:,None]*rot[None,:]) # of the three bonds
    k = 4.*np.pi/(3.*np.sqrt(3.))*1j*z0/np.abs(z0)**2 # rotated 90 degrees
    return 2.*np.array([k.real,k.imag,0.*k.real]).T


def _smallest_polar_angle(cand):
    """Return, for each row of the complex array cand, the entry with the
    smallest polar angle in [0,2 pi), with an entry a rounding error below
    the x axis counted as on it"""
    angles = (np.angle(cand)+1e-6)%(2.*np.pi)
    return cand[np.arange(len(cand)),np.argmin(angles,axis=1)]


def _lattice_origins(r1,r2):
    """Return, for each of the bonds r1[i],r2[i], the site of the
    honeycomb lattice of that bond from which the phase is measured, as a
    complex number x+iy: the site closest to the origin of coordinates.
    The lattice is the infinite one that the bond belongs to, so the site
    does not have to be an atom of the geometry.

    The usual origins of coordinates (a hexagon center, a bond midpoint)
    are at the same distance from several sites, and a rule that took the
    closest one to the last digit would change site, and rotate the valley
    axes by 120 degrees, under the smallest shift of the geometry, as
    recentering a flake after removing an atom. So every site within
    _ORIGIN_WINDOW of the closest distance competes, and the one furthest
    along a fixed direction a little above the x axis wins. The direction
    is not one of a lattice in its usual orientations, so two competing
    sites are not level along it."""
    z1 = r1[:,0] + 1j*r1[:,1]
    dz = z1 - (r2[:,0] + 1j*r2[:,1]) # the bond
    w = np.exp(2j*np.pi/3.)
    (u1,u2) = (dz*(1.-w),dz*(1.-w*w)) # lattice vectors
    det = u1.real*u2.imag - u1.imag*u2.real
    cand = []
    for z in (z1,z1-dz): # the two sublattices
        m = np.round((-z.real*u2.imag + z.imag*u2.real)/det)
        n = np.round((-u1.real*z.imag + u1.imag*z.real)/det)
        for dm in (-1,0,1): # the cells around the closest one
            for dn in (-1,0,1): cand.append(z + (m+dm)*u1 + (n+dn)*u2)
    cand = np.array(cand).T # bond, candidate
    dist = np.abs(cand)
    out = dist>np.min(dist,axis=1)[:,None]+_ORIGIN_WINDOW # do not compete
    along = cand*np.exp(-1j*_ORIGIN_DIRECTION) # real part: along it
    score = np.where(out,-np.inf,along.real)
    level = score>np.max(score,axis=1)[:,None]-1e-6
    score = np.where(level,along.imag,-np.inf) # two level sites: the left one
    return cand[np.arange(len(cand)),np.argmax(score,axis=1)]


def _bond_phases(r1,r2,sub1,sub2):
    """Return Q.(r_mid - r_0) for each of the bonds r1[i],r2[i], with
    r_mid the midpoint of the bond and r_0 the lattice site closest to the
    origin of coordinates"""
    qs = _bond_wavevectors(r1,r2,sub1,sub2)
    z0 = _lattice_origins(r1,r2)
    rmid = (r1+r2)/2.
    return (rmid[:,0]-z0.real)*qs[:,0] + (rmid[:,1]-z0.imag)*qs[:,1]


def _check_valley_geometry(g):
    """Raise unless the in-plane valley operators can be defined on g: a
    sublattice, first neighbors on opposite sublattices in directions 120
    degrees apart, and for a periodic geometry a cell that holds a whole
    period of the plane wave"""
    check.require_sublattice(g,"the in-plane valley")
    rs = np.asarray(g.r)
    sub = np.asarray(g.sublattice)
    lattice_vectors = [g.a1,g.a2,g.a3][:g.dimensionality]
    shifts = [np.zeros(3)]
    for a in lattice_vectors: # the cell and its neighboring replicas
        shifts = [s+n*np.asarray(a) for s in shifts for n in (-1,0,1)]
    r2 = np.concatenate([rs+s for s in shifts])
    sub2 = np.tile(sub,len(shifts))
    (ii,jj) = _first_neighbor_pairs(rs,r2)
    if len(ii)==0: raise ValueError(_WHAT+" need a honeycomb lattice with "
            "bond length 1, and this geometry has no first neighbors with "
            "an in-plane part")
    qs = _bond_wavevectors(rs[ii],r2[jj],sub[ii],sub2[jj])
    # the three bonds of a site are 120 degrees apart when they all find
    # the same K; the guards are written so that a NaN fails them
    (sites,first) = np.unique(ii,return_index=True)
    qsite = np.zeros((len(rs),3))
    qsite[sites] = qs[first]
    if not np.all(np.abs(qs-qsite[ii])<=1e-3): raise ValueError(_WHAT+
            " need a honeycomb lattice: the bonds of a site have to point "
            "in three directions 120 degrees apart, and in this geometry "
            "they do not")
    # a periodic geometry has to hold a whole period of the plane wave,
    # which is the Kekule cell: Q.a must be a multiple of 2 pi
    for (i,a) in enumerate(lattice_vectors):
        n = qs@np.asarray(a)/(2.*np.pi)
        if not np.all(np.abs(n-np.round(n))<=1e-3): raise ValueError(_WHAT+
            " need a Kekule-commensurate geometry when periodic: a "
            "supercell of the primitive honeycomb cell whose size is a "
            "multiple of 3. Lattice vector a%d=%s is not commensurate "
            "with it; use e.g. g.get_supercell(3) (or any other multiple "
            "of 3) first."%(i+1,a))


def _bond_wave(sub,amplitude=1.0,angle=0.0,sparse=False):
    """Return the matrix generator fm(rs1,rs2) of the hopping
    amplitude*(2/3)*cos(Q.r_mid - angle) on the first-neighbor bonds,
    which is amplitude*(cos(angle) tau_x + sin(angle) tau_y), as sparse
    matrices if sparse=True and dense arrays otherwise. sub is the
    sublattice of the sites, the same for rs1 and rs2, which are the cell
    and one of its replicas"""
    from scipy.sparse import csc_matrix
    sub = np.asarray(sub)
    def fm(rs1,rs2):
        rs1,rs2 = np.asarray(rs1),np.asarray(rs2)
        (ii,jj) = _first_neighbor_pairs(rs1,rs2)
        phis = _bond_phases(rs1[ii],rs2[jj],sub[ii],sub[jj])
        data = amplitude*2./3.*np.cos(phis-angle) + 0j
        m = csc_matrix((data,(ii,jj)),shape=(len(rs1),len(rs2)))
        if sparse: return m
        return m.toarray()
    return fm


def _inplane_valley_hamiltonian(h,amplitude=1.0,angle=0.0):
    """Return a copy of h whose only term is
    amplitude*(cos(angle) tau_x + sin(angle) tau_y)"""
    _check_valley_geometry(h.geometry)
    ho = h.copy() ; ho.turn_multicell() ; ho.clean()
    fm = _bond_wave(h.geometry.sublattice,amplitude=amplitude,angle=angle,
            sparse=ho.is_sparse)
    # the normal state with the degrees of freedom of h, sparse if h is
    hn = h.geometry.get_hamiltonian(has_spin=h.has_spin,is_multicell=True,
            is_sparse=h.is_sparse,mgenerator=fm)
    if h.has_eh: hn.setup_nambu_spinor() # and then its Nambu form
    ho.set_multihopping(hn.get_multihopping())
    return ho


def add_valley_exchange(h,v):
    """Add a valley-space exchange term v=(vx,vy,vz).(tau_x,tau_y,tau_z)
    to the Hamiltonian, the valley-pseudospin analogue of add_exchange
    for real spin. vz is added as a modified-Haldane (second-neighbor)
    coupling; vx,vy are added as a first-neighbor hopping modulated by a
    plane wave of wavevector K-K' (see the comment at the top of this
    module), which opens a gap |v| at the Dirac points.

    Just like the valley operators themselves (get_operator("valley"/
    "valley_x"/"valley_y")), the in-plane part is only periodic on a
    Kekule-commensurate geometry (e.g. h.geometry already a 3x3, or
    other multiple-of-3, supercell of the primitive honeycomb cell) for
    dimensionality>0 Hamiltonians; for 0-dimensional (finite flake)
    Hamiltonians no such commensurability is required."""
    check.require_sublattice(h,"a valley exchange")
    if callable(v): raise TypeError("add_valley_exchange takes a uniform "
            "valley exchange v=(vx,vy,vz), three numbers, not a function "
            "of position: its in-plane part is a Kekule bond texture "
            "painted over the whole cell")
    if np.shape(v)!=(3,): raise ValueError("add_valley_exchange takes "
            "v=(vx,vy,vz), three numbers, got "+repr(v))
    (vx,vy,vz) = v
    if np.iscomplexobj(np.array([vx,vy])): raise ValueError(
            "add_valley_exchange needs real vx and vy, since a complex "
            "one would add a non-Hermitian term, got "+repr(v))
    h.turn_multicell()
    if vz!=0.: h.add_modified_haldane(vz*valley_coupling(h.geometry))
    if vx!=0. or vy!=0.:
        ho = _inplane_valley_hamiltonian(h,amplitude=np.sqrt(vx**2+vy**2),
                angle=np.arctan2(vy,vx)) # vx*tau_x + vy*tau_y
        h.set_multihopping(h.get_multihopping()+ho.get_multihopping())


def get_inplane_valley(h,angle=0.0,delta=None,**kwargs):
    """Return a callable that calculates the in-plane valley (tau_x/tau_y)
    expectation value, a first-neighbor hopping modulated by a plane wave
    of wavevector K-K' (see the comment at the top of this module).
    angle=0 gives tau_x, angle=pi/2 gives tau_y, any other angle a linear
    combination (a rotation of the in-plane pseudospin axis).

    Just like the out-of-plane valley operator (get_valley), this is
    only periodic on a Kekule-commensurate geometry (e.g. h.geometry
    already a 3x3, or other multiple-of-3, supercell of the primitive
    honeycomb cell) for dimensionality>0 Hamiltonians; for 0-dimensional
    (finite flake) Hamiltonians no such commensurability is required."""
    ho = _inplane_valley_hamiltonian(h,angle=angle)
    hkgen = ho.get_hk_gen() # cos(angle)*tau_x + sin(angle)*tau_y
    sharpen = get_sharpen(delta=delta) # renormalize eigenvalues to +-1
    if h.dimensionality==0: return sharpen(hkgen(None)) # return a matrix
    # the k-point and the matrix of the last call: an expectation value or
    # a band structure applies the operator to every state of a k-point in
    # turn, and rebuilding the Bloch sum (and with delta its
    # diagonalization) for each state cost one build per state instead of
    # one per k-point. The pair is kept and read as one object, so that two
    # threads cannot take the matrix of one k-point for another
    last = [None]
    def fun(m=None,k=None):
        if k is None: # requires a kpoint
            raise ValueError("the in-plane valley operator of a periodic "
                    "Hamiltonian has to be evaluated at a k-point, and none "
                    "was given")
        key = tuple(np.asarray(k,dtype=float).ravel())
        kept = last[0]
        if kept is None or kept[0]!=key:
            kept = (key,sharpen(hkgen(k))) # evaluate and sharpen
            last[0] = kept
        hk = kept[1]
        if m is None: return hk.copy() # a fresh matrix, as before
        else: return hk@m # return the projector
    return fun # return function


def get_valley_taux(h,**kwargs):
    """Return the tau_x valley operator"""
    return get_inplane_valley(h,angle=0.0,**kwargs)


def get_valley_tauy(h,**kwargs):
    """Return the tau_y valley operator"""
    return get_inplane_valley(h,angle=np.pi/2,**kwargs)
