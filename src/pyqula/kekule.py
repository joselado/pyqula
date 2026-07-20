import inspect
import hashlib
import numpy as np
from . import geometry


remove_duplicated = geometry.remove_duplicated_positions

### Routines dealing with Kekule order ###

def hexagon_centers(r1, r2=None):
    """
    Return the centers of an hexagon (vectorized: for honeycomb-derived
    position sets this used to be an O(N^2) pure-Python double loop, one
    of the dominant costs of building any Kekule-textured coupling)
    """
    r1 = np.asarray(r1)
    r2 = r1 if r2 is None else np.asarray(r2)
    diff = r1[:,None,:] - r2[None,:,:] # all pairwise differences
    d2 = np.sum(diff*diff,axis=-1) # all pairwise squared distances
    ii,jj = np.where((d2>3.9) & (d2<4.1)) # centers of an hexagon
    return (r1[ii] + r2[jj])/2. # midpoints


def _fast_remove_duplicated(r,tol=1e-3):
    """Deduplicate a position array by rounding to a grid, avoiding the
    O(N^2) generic geometry.remove_duplicated_positions for the (often
    large, since hexagon_centers finds every center many times over)
    hexagon-center lists built here"""
    if len(r)==0: return np.zeros((0,3))
    key = np.round(np.asarray(r)/tol).astype(np.int64)
    _,inds = np.unique(key,axis=0,return_index=True)
    return np.asarray(r)[np.sort(inds)]


def retain(centers,d=3.0,start_index=0):
    """
    Retain only the centers that are part of the same (mutually spaced
    by d) sublattice as centers[start_index], found by a breadth-first
    flood fill. Vectorized version of the original nested-loop
    implementation: each iteration checks the whole remaining point set
    against the current frontier with numpy broadcasting instead of a
    double Python loop.
    """
    centers = np.asarray(centers)
    n = len(centers)
    kept = np.zeros(n,dtype=bool)
    kept[start_index] = True
    d2lo,d2hi = d*d-0.1,d*d+0.1
    frontier = np.array([start_index])
    while len(frontier)>0:
        diff = centers[frontier][:,None,:] - centers[None,:,:]
        d2 = np.sum(diff*diff,axis=-1)
        near = np.any((d2>d2lo) & (d2<d2hi),axis=0)
        new = near & (~kept)
        if not np.any(new): break
        kept = kept | new
        frontier = np.where(new)[0]
    return centers[kept]


_registry_cache = {} # cache of (all_centers,[registry0,registryA,registryB])
                      # keyed by a hash of the geometry that produced them


def _geometry_key(g):
    parts = [np.round(g.r,6)]
    for a in ("a1","a2","a3"):
        if hasattr(g,a): parts.append(np.round(getattr(g,a),6))
    data = b"".join(p.tobytes() for p in parts)
    return hashlib.sha1(data).hexdigest()


_positions_cache = {} # cache of kekule_positions(r), keyed by a hash of r


def kekule_positions(r):
    """
    Returns the positions defining a Kekule ordering (the single,
    arbitrary registry historically picked by this function: the one
    containing the first hexagon center found). Cached by position
    content, since this is the expensive part of building any
    Kekule-textured coupling and callers (add_kekule, chiral_kekule...)
    tend to recompute it from scratch on every call for what is usually
    the same underlying geometry.
    """
    key = hashlib.sha1(np.round(np.asarray(r),6).tobytes()).hexdigest()
    if key in _positions_cache: return _positions_cache[key]
    cs = hexagon_centers(r,r) # return the centers
    cs = _fast_remove_duplicated(cs) # remove duplicated
    zs = np.unique(np.round(cs[:,2],5)) # unique z positions
    cso = [] # output list
    for z in zs:
        csi = cs[(np.abs(cs[:,2]-z)<1e-4),:] # this layer
        csi = retain(csi,d=3.0,start_index=0) # retain only centers at distance 3
        cso = cso + np.array(csi).tolist() # store
    out = np.array(cso) # return array
    _positions_cache[key] = out
    return out


def kekule_registries(g):
    """
    Return the 3 mutually-inequivalent Kekule-hexagon registries (the 3
    ways of picking 1/3 of all hexagons as "kept", each a translate of
    the others) for a honeycomb-like geometry, as a list of 3 position
    arrays [registry0, registryA, registryB]. Cached per geometry, since
    finding these is the expensive part of building any chiral-Kekule
    coupling and only depends on the atomic positions, not on the
    hopping amplitudes eventually painted onto them.

    registry0 is the same single registry historically returned by
    kekule_positions/chiral_kekule (an arbitrary but deterministic
    choice tied to atom ordering). registryA and registryB are the two
    other registries, obtained as *rigid translates* of registry0 by
    +-v and -v (v a next-nearest-neighbor vector of the honeycomb
    lattice, i.e. a primitive vector of the hexagon-center lattice) --
    not by re-running the flood fill from a different starting point,
    since two independent flood fills started near different parts of
    the necessarily-finite search domain can terminate at slightly
    different (boundary-truncated) sizes, which silently corrupted the
    registry near the region of interest. A rigid translation of the
    already-computed registry0 has no such issue: it is congruent to
    registry0 by construction.
    """
    key = _geometry_key(g)
    if key in _registry_cache and len(_registry_cache[key][1])==3:
        return _registry_cache[key][1]
    r2 = g.multireplicas(2)
    cs_all = _fast_remove_duplicated(hexagon_centers(r2,r2))
    registry0 = retain(cs_all,d=3.0,start_index=0)
    # analytic shift to an adjacent (different-registry) hexagon center:
    # a next-nearest-neighbor vector of the honeycomb lattice, derived
    # (like the bond-classification z0 in chiral_kekule) from the first
    # NN bond of the geometry
    dr0 = g.r[0]-g.r[1]
    dz = np.exp(1j*2.*np.pi/3.)
    z0 = dr0[0]+1j*dr0[1]
    shift = z0*(1.-dz) # NNN vector: one hexagon-lattice primitive step
    shift3d = np.array([shift.real,shift.imag,0.])
    registryA = registry0 + shift3d
    registryB = registry0 - shift3d
    out = [registry0,registryA,registryB]
    _registry_cache[key] = (cs_all,out)
    return out


def _registry_set(cs,tol=1e-3):
    """Hash set of a registry's centers for O(1) membership tests"""
    key = np.round(np.asarray(cs)/tol).astype(np.int64)
    return set(map(tuple,key))


def _in_registry(pos,regset,tol=1e-3):
    key = tuple(np.round(np.asarray(pos)/tol).astype(np.int64))
    return key in regset


def _bond_flanking_centers(r1,r2):
    """The (at most two) hexagon centers flanking a first-neighbor bond,
    found analytically (both are at distance 1 from r1 and from r2)
    instead of by scanning every retained hexagon center"""
    d = r2-r1
    perp = np.array([-d[1],d[0],0.])
    norm = np.sqrt(perp.dot(perp))
    if norm<1e-8: return None,None
    perp = perp/norm
    h = np.sqrt(0.75) # sqrt(1-0.5^2), since |r1-r2|=1 and mid is at 0.5
    mid = (r1+r2)/2.
    return mid+h*perp, mid-h*perp


def kekule_center(r):
    """
    Returns a function that given two positions, returns the Kekule center
    """
    cs = kekule_positions(r) # returns centers of the kekule
    regset = _registry_set(cs)
    def f(r1,r2): # function that returs the kekule center
        dr = r1-r2 ; dr=dr.dot(dr)
        if not 0.99<dr<1.01: return None # too far
        c1,c2 = _bond_flanking_centers(r1,r2)
        if c1 is None: return None
        if _in_registry(c1,regset): return c1
        if _in_registry(c2,regset): return c2
        return None
    return f # return function




def kekule_function(r,t=1.):
    """
    Returns a function that will compute Kekule hoppings
    """
    cs = kekule_positions(r) # get the centers with all the positions
    regset = _registry_set(cs)
    ## Define a function to only have hoppings in the hexagon
    ## t can be a number, a function of the bond midpoint t(rmid), or a
    ## bond-direction-aware function t(r1,r2) (e.g. chiral_kekule's output)
    if callable(t):
        nargs = len(inspect.signature(t).parameters)
        if nargs>=2: tfun = t # bond-direction-aware function
        else: tfun = lambda r1,r2: t((r1+r2)/2.) # function of the midpoint
    else: tfun = lambda r1,r2: t # t is a number
    def f(r1,r2):
        dr = r1-r2 ; dr=dr.dot(dr)
        if not 0.99<dr<1.01: return 0.0
        c1,c2 = _bond_flanking_centers(r1,r2)
        if c1 is None: return 0.0
        if _in_registry(c1,regset) or _in_registry(c2,regset):
            return tfun(r1,r2)
        return 0.0
    # now define the function
    def fm(rs1,rs2):
      m = np.zeros((len(rs1),len(rs2)),dtype=np.complex128) # initialize matrix
      for i in range(len(rs1)): # loop
        for j in range(len(rs2)): # loop
            m[i,j] = f(rs1[i],rs2[j]) # get kekule coupling
      return m # return the Kekule matrix
    return fm # return the function


def bond_function_to_matrix(fun):
    """Wrap a bond-direction-aware function fun(r1,r2) (e.g. the output
    of chiral_kekule) directly into a mgenerator-style matrix builder
    fm(rs1,rs2), without re-applying kekule_function's own (independent,
    always-registry0) mask on top of it -- needed when fun was built
    against a specific, non-default registry (see kekule_registries):
    going through add_kekule/kekule_function there would silently zero
    out every bond outside registry0, since that mask does not know
    about fun's own registry choice."""
    def fm(rs1,rs2):
        m = np.zeros((len(rs1),len(rs2)),dtype=np.complex128)
        for i in range(len(rs1)):
            for j in range(len(rs2)):
                m[i,j] = fun(rs1[i],rs2[j])
        return m
    return fm


def kekule_matrix(r1,r2=None,**kwargs):
    """
    Return a Kekule matrix for positions r, assuming
    they are from a honeycomb-like lattice
    """
    if r2 is None: r2 = r1
    f = kekule_function(r1,**kwargs)
    return f(r1,r2)


def r_in_rs(r,rs):
    """
    Check that a position is not stored
    """
    for ri in rs:
        dr = ri-r ; dr = dr.dot(dr)
        if dr<0.01: return True
    return False



def chiral_kekule(g,t1=0.0,t2=0.0,hermitian=True,registry=None):
    """Return a bond-direction-aware chiral kekule hopping function,
    input is a geometry. registry (optional) is an explicit array of
    retained hexagon centers (e.g. one of kekule_registries(g)'s three
    entries) to paint the (t1,t2) amplitudes onto; if not given, the
    single default registry (kekule_positions(g)) is used, matching the
    historical behavior of this function."""
    if registry is None: registry = kekule_positions(g.multireplicas(2))
    regset = _registry_set(registry)
    dr0 = g.r[0] - g.r[1] # one NN vector
    dz = np.exp(1j*2.*np.pi/3.) # second vector
    z0 = dr0[0] + 1j*dr0[1] # complex NN vector
    if not 0.99<dr0.dot(dr0)<1.01: raise # this has to be more robust
    # function to check first kind of bond
    def kind1(dr):
        z = dr[0] + 1j*dr[1] # complex position
        for iz in [z0,z0*dz,z0*dz*dz]: # loop
            if abs(z*np.conjugate(iz)-1.0)<0.01: return True
        return False
    # function to check if going clockwise
    clockwise = lambda r1,r2: np.cross(r1,r2)[2]>0. # if clockwise
    def fun(r1,r2):
        dr = r1-r2 # distance between sites
        if not 0.99<dr.dot(dr)<1.01: return 0. # not first neighbors
        c1,c2 = _bond_flanking_centers(r1,r2) # candidate Kekule centers
        if c1 is None: return 0.
        if _in_registry(c1,regset): rk = c1
        elif _in_registry(c2,regset): rk = c2
        else: return 0. # this bond is not part of the kept registry
        if clockwise(rk-r1,r2-r1): # clockwise hopping
          if kind1(dr): return t1 # first kind of bond
          else: return t2
        else:
          if hermitian: # conventional Hamiltonian
            if kind1(dr): return np.conjugate(t2) # first kind of bond
            else: return np.conjugate(t1)
          else: return 0.0
    return fun
