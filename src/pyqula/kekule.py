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
    # the pairs at distance 2 from a KD-tree, in the row-major order of
    # the n x n array of distances that this used to build (an 80 GB
    # array at 100,000 sites), since the centers found first are the
    # ones retain() starts from
    from scipy.spatial import cKDTree
    near = cKDTree(r1).query_ball_tree(cKDTree(r2),np.sqrt(4.1))
    ii = np.repeat(np.arange(len(r1)),[len(x) for x in near])
    jj = np.array([j for x in near for j in sorted(x)],dtype=int)
    diff = r1[ii] - r2[jj]
    d2 = np.sum(diff*diff,axis=-1) # squared distances of those pairs
    keep = (d2>3.9) & (d2<4.1) # centers of an hexagon
    return (r1[ii[keep]] + r2[jj[keep]])/2. # midpoints


def _fast_remove_duplicated(r,tol=1e-3):
    """Deduplicate a position array by rounding to a grid, avoiding the
    O(N^2) generic geometry.remove_duplicated_positions for the (often
    large, since hexagon_centers finds every center many times over)
    hexagon-center lists built here"""
    if len(r)==0: return np.zeros((0,3))
    key = np.round(np.asarray(r)/tol).astype(np.int64)
    _,inds = np.unique(key,axis=0,return_index=True)
    return np.asarray(r)[np.sort(inds)]


def _retain_mask(centers,d,start_index):
    """Boolean mask of the centers reachable from centers[start_index]
    by a breadth-first flood fill over mutual distance d (the connected
    component of the "distance d" adjacency graph containing that
    point). Vectorized: each iteration checks the whole remaining point
    set against the current frontier with numpy broadcasting instead of
    a double Python loop."""
    from scipy.spatial import cKDTree
    centers = np.asarray(centers)
    n = len(centers)
    kept = np.zeros(n,dtype=bool)
    kept[start_index] = True
    d2lo,d2hi = d*d-0.1,d*d+0.1
    tree = cKDTree(centers)
    frontier = np.array([start_index])
    while len(frontier)>0:
        # the neighbors of the frontier from the KD-tree, rather than the
        # frontier against every center, which grows as n^2
        lists = tree.query_ball_point(centers[frontier],np.sqrt(d2hi))
        jj = np.array([j for x in lists for j in x],dtype=int)
        ff = np.repeat(frontier,[len(x) for x in lists])
        diff = centers[ff] - centers[jj]
        d2 = np.sum(diff*diff,axis=-1)
        near = np.zeros(n,dtype=bool)
        near[jj[(d2>d2lo) & (d2<d2hi)]] = True
        new = near & (~kept)
        if not np.any(new): break
        kept = kept | new
        frontier = np.where(new)[0]
    return kept


def retain(centers,d=3.0,start_index=0):
    """
    Retain only the centers that are part of the same (mutually spaced
    by d) sublattice as centers[start_index], found by a breadth-first
    flood fill (the connected component of the "distance d" adjacency
    graph containing that point). Note this only returns the single
    component reachable from start_index: if centers spans several
    physically disconnected sheets/islands (different layers, separate
    flakes...), the other components are silently dropped. Unioning
    flood-filled components blindly is not a safe fix for that (it
    cannot distinguish "a different disconnected piece of the same
    coset" from "a different coset that also happens to be internally
    connected").
    """
    centers = np.asarray(centers)
    return centers[_retain_mask(centers,d,start_index)]


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
    def fm(rs1,rs2,sparse=False):
      # f vanishes beyond the first neighbors, so only the pairs at
      # distance up to 1 are evaluated, found with a KD-tree: linear in the
      # number of sites, where evaluating every pair was quadratic (216 s
      # for the kekule guess of a 10,000-site island)
      from scipy.spatial import cKDTree
      from scipy.sparse import csc_matrix
      rs1,rs2 = np.asarray(rs1),np.asarray(rs2)
      near = cKDTree(rs1).query_ball_tree(cKDTree(rs2),np.sqrt(1.01))
      ii = np.repeat(np.arange(len(rs1)),[len(x) for x in near])
      jj = np.array([j for x in near for j in x],dtype=int)
      data = np.array([f(rs1[i],rs2[j]) for (i,j) in zip(ii,jj)],
              dtype=np.complex128) # get kekule coupling
      m = csc_matrix((data,(ii,jj)),shape=(len(rs1),len(rs2)))
      m.eliminate_zeros()
      if sparse: return m
      return m.toarray() # return the Kekule matrix
    return fm # return the function


def bond_function_to_matrix(fun):
    """Wrap a bond-direction-aware function fun(r1,r2) (e.g. the output
    of chiral_kekule) directly into a mgenerator-style matrix builder
    fm(rs1,rs2), without re-applying kekule_function's own (independent,
    always-registry0) mask on top of it -- needed when fun was built
    against a specific, non-default registry:
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


def kekule_matrix(r1,r2=None,sparse=False,**kwargs):
    """
    Return a Kekule matrix for positions r, assuming
    they are from a honeycomb-like lattice, as a sparse matrix
    if sparse=True and a dense array otherwise
    """
    if r2 is None: r2 = r1
    f = kekule_function(r1,**kwargs)
    return f(r1,r2,sparse=sparse)


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
    retained hexagon centers to paint the (t1,t2) amplitudes onto; if
    not given, the
    single default registry (kekule_positions(g)) is used, matching the
    historical behavior of this function."""
    if registry is None: registry = kekule_positions(g.multireplicas(2))
    regset = _registry_set(registry)
    dr0 = g.r[0] - g.r[1] # one NN vector
    dz = np.exp(1j*2.*np.pi/3.) # second vector
    z0 = dr0[0] + 1j*dr0[1] # complex NN vector
    if not 0.99<dr0.dot(dr0)<1.01:
        raise ValueError(
            "chiral_kekule requires g.r[0] and g.r[1] to be nearest "
            "neighbors (bond length 1); got |g.r[0]-g.r[1]|=%.4f. This "
            "usually means the geometry's first two atoms are not "
            "adjacent (e.g. after removing a low-index atom)."
            % np.sqrt(dr0.dot(dr0)))
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
