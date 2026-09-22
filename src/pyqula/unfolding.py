# routines to perform unfolding of the Brillouin zone

import numpy as np
from scipy.linalg import eigh
from .algebra import dagger, inv

def unfolded_bands(hfol,hprim,kpath,inds_super=[]):
  """ Save in file unfolded band structure"""
  # the body of this stub read an unbound name in its first lines, so
  # every call raised NameError inside the k-loop instead of this
  raise NotImplementedError("unfolded_bands is not implemented; use the "
          "unfold operator (see bloch_projector) instead")




def perturb_bands(hprim,hper,kpath,inds_super=[]):
  """ Save in file perturbed band structure"""
  hkprim = hprim.get_hk_gen() # generator
  for k in kpath: # loop over kpoints
    kprim = k/3.
    hp = hkprim(k) # get matrix
    (ep,wfp) = eigh(hp) # eigenvalues and eigenvectors 
    wfp = wfp.transpose() # transpose eigenvectors
    # now calculate eigenfunctions in a supercell
    wfs = [] # wavefunction of the supercell
    for w in wfp: # loop over functions
      wlist = [] # create empty list
      for ix in range(inds_super[0]): # loop over ix
        for iy in range(inds_super[1]):  # loop over iy
          for iz in range(inds_super[2]):  # loop over iz
            phi = k.dot(np.array([ix,iy,iz])) # phase
            wj = w*np.exp(1j*np.pi*phi) # wavefunction
            wlist.append(wj) # append to the list
      wfs.append(np.concatenate(wlist)) # add this wave to the list
   # now apply perturbation theory
    vm = np.zeros((len(wfs),len(wfs)),dtype=np.complex128) # perturbation
    for iw in wfs:
      for jw in wfs:
        iw = np.array(iw)
        jw = dagger(np.array(jw))
        vaw = iw@hper@jw # matrix element
  
    raise NotImplementedError("perturb_bands is not implemented")



def bloch_projector(h,g0=None):
    """Given a certain Hamiltonian and minimal geometry, return
    a projector to the minimal BZ"""
    if g0 is None:
        if h.geometry.primal_geometry is None:
            raise ValueError("the unfolding projector needs the "
                    "primitive-cell geometry; build the supercell with "
                    "store_primal=True, or pass the primitive geometry as g0")
        else: g0 = h.geometry.primal_geometry # get the primal geometry
    natoms = len(h.geometry.r) # atoms in this Hamiltonian
    ntot = h.intra.shape[0] # orbitals in this Hamiltonian
    if ntot%natoms!=0:
        raise ValueError("unfolding: the Hamiltonian has "+str(ntot)+
            " orbitals and "+str(natoms)+" atoms, which is not a whole "
            "number of orbitals per atom")
    norb_factor = ntot//natoms # orbitals per atom (spin/electron-hole)
    n0 = len(g0.r)*norb_factor # orbitals in the primal cell
    # which primal replica and which primal atom every atom came from,
    # plus the integer matrix M with A_supercell = M@A_primal
    M,replicas,primal_indices = get_supercell_map(h.geometry,g0)
    fs = bloch_phase_matrix(n0,replicas,primal_indices,norb_factor,M)
    # The operator is O = P^dagger P, where P(k) is the matrix of Bloch
    # phases projecting a supercell state onto the Bloch states of the
    # primal cell: <psi|O|psi> is the weight of psi on the primal cell at
    # this kpoint. O is never built. P has exactly one nonzero entry per
    # column, so applying it as P^dagger(P v) costs 2N operations, whereas
    # the N x N matrix O has N/n0 nonzeros per row and would cost N^2/n0.
    # The consumers apply this operator once per band at a fixed kpoint
    # (see bandstructure.get_bands_nd), so P is cached for the kpoint in
    # hand rather than rebuilt for every eigenvector.
    cache = {"k":None,"P":None,"Pd":None}
    def get_projector(k):
        """The (n0 x N) matrix projecting onto the primal-cell Bloch
        states at this kpoint, and its adjoint"""
        key = tuple(np.asarray(k,dtype=float).ravel()) # hashable kpoint
        if cache["k"]!=key: # not the kpoint in hand, rebuild
            P = fs(k).conjugate() # the conjugate is what gets applied
            cache["k"] = key ; cache["P"] = P ; cache["Pd"] = P.conj().T
        return cache["P"],cache["Pd"]
    def fun(v,k=None):
        if k is None:
            raise ValueError("the unfolding operator is k-dependent and "
                    "needs the kpoint it is evaluated at; pass k=...")
        P,Pd = get_projector(k) # projector onto the primal Bloch states
        return Pd@(P@v) # O@v, for v a single state or a whole matrix
    from .operators import Operator
    return Operator(fun,linear=True) # return operator



def get_supercell_map(g,g0):
    """Given a supercell geometry g and the primal geometry g0 it was
    built from, return (M,replicas,primal_indices): the integer matrix
    with A_supercell = M@A_primal, and, per atom of g, the replica vector
    n in Z^3 and the index of the primal atom it is a copy of.

    Every supercell builder records this at construction time, which is
    exact and costs nothing. The record is checked against g0 before it
    is used, since it travels with the geometry through Geometry.copy()
    and a geometry can be assembled in ways that leave it behind; when it
    does not describe g in terms of *this* g0, the map is rebuilt by
    matching positions instead."""
    M = getattr(g,"supercell_matrix",None)
    replicas = getattr(g,"supercell_replica",None)
    primal_indices = getattr(g,"supercell_primal_index",None)
    if M is not None and replicas is not None and primal_indices is not None:
        M = np.array(M,dtype=int)
        replicas = np.array(replicas,dtype=int)
        primal_indices = np.array(primal_indices,dtype=int)
        if describes_supercell(g,g0,replicas,primal_indices):
            return M,replicas,primal_indices
    # no usable record, fall back to matching positions against an ideal
    # supercell of g0 (this also covers supercells with atoms removed)
    from .supercell import infer_supercell
    nsuper = infer_supercell(g,g0) # guess the diagonal supercell
    replicas,primal_indices = get_replica_map(g,g0,nsuper)
    return np.diag(nsuper),replicas,primal_indices



def describes_supercell(g,g0,replicas,primal_indices):
    """Check that every atom of g sits where (replicas,primal_indices)
    says it does, namely at r0[primal] + n@A0, up to the single rigid
    shift that the supercell builders apply when they center the cell"""
    if len(replicas)!=len(g.r) or len(primal_indices)!=len(g.r): return False
    if len(g.r)==0: return False
    if np.max(primal_indices)>=len(g0.r) or np.min(primal_indices)<0:
        return False
    d = g0.dimensionality
    if d<3 and np.any(replicas[:,d:]!=0):
        # only the first d lattice vectors are real ones; a2 and a3 of a
        # chain, and a3 of a two-dimensional lattice, are placeholders, so
        # a replica vector with a component along them is not something
        # this geometry's builders produced
        return False
    A0 = np.array([g0.a1,g0.a2,g0.a3]).real # primal lattice vectors
    expected = g0.r[primal_indices].real + replicas@A0 # where they should be
    dr = g.r.real - expected # should be one and the same shift for all
    tol = 1e-6*max(1.,np.max(np.abs(A0))) # relative to the cell size
    return bool(np.max(np.abs(dr-dr[0]))<tol)



def decompose_supercell_index(inds,nc,nsuper):
    """Decompose an atom index of an (n1,n2,n3) supercell into its replica
    vector and its primal-atom index. Every diagonal supercell builder
    fills its output in the same nesting order -- n1 outermost, then n2,
    then n3, then the atoms of the primal cell -- so this is a pure index
    decomposition with no position matching involved"""
    n1,n2,n3 = nsuper[0],nsuper[1],nsuper[2]
    inds = np.asarray(inds)
    k = inds%nc ; t = inds//nc # primal atom index
    l = t%n3 ; t = t//n3 # replica along a3
    j = t%n2 ; i = t//n2 # replica along a2 and a1
    return np.stack([i,j,l],axis=-1),k



def get_replica_map(g,g0,nsuper):
    """Given a geometry g that is a supercell of the primal geometry g0
    (possibly with some atoms removed), match every remaining atom to its
    primal replica index (n1,n2,n3) and its primal atom index, by
    comparing positions against an ideal, defect-free supercell built
    with the same g0 and nsuper. This sidesteps having to invert
    fractional coordinates (which would have to account for the arbitrary
    centering shift that get_supercell applies), since the ideal
    supercell is built with the exact same deterministic construction
    routine, so atoms that survived removal keep bit-identical
    positions."""
    from scipy.spatial import cKDTree
    nc = len(g0.r) # number of atoms in the primal cell
    g_ideal = g0.get_supercell(nsuper) # ideal, defect-free supercell
    tree = cKDTree(g_ideal.r.real)
    dists,inds = tree.query(g.r.real) # match each atom
    tol = 1e-6 # fallback tolerance, if there is no neighbor to scale it by
    if len(g_ideal.r)>1: # scale the tolerance by the nearest-neighbor distance
        dnn,_ = tree.query(g_ideal.r.real,k=2) # distances to self and to the NN
        nn_dists = dnn[:,-1] # distance to the actual nearest neighbor
        nn_dists = nn_dists[np.isfinite(nn_dists)&(nn_dists>0)]
        if len(nn_dists)>0: tol = 0.1*np.min(nn_dists) # tolerance
    if np.any(dists>tol):
        raise ValueError("unfolding: could not match "+
            str(int(np.sum(dists>tol)))+" supercell atom(s) to the "
            "ideal primal supercell (tolerance "+str(tol)+"). Check "
            "that g0 and the inferred supercell size correspond to "
            "how this supercell was built, and that atoms were "
            "removed (not displaced).")
    if len(np.unique(inds))!=len(inds):
        raise ValueError("unfolding: several supercell atoms matched "
            "the same primal-cell replica site; check for duplicated "
            "or coincident atom positions in the supercell geometry.")
    return decompose_supercell_index(inds,nc,nsuper)



def bloch_phase_matrix(n0,replicas,primal_indices,norb_factor,M):
    """Return a function of k giving the matrix of Bloch phases that
    projects a supercell state onto the Bloch states of the primal cell.

    It is built from each atom's stored replica vector n in Z^3 (an
    arbitrary integer vector, so a general, possibly non-diagonal and
    non-orthogonal supercell is no different from a diagonal one), and k
    is mapped with k_primal_reduced = Minv@k_super_reduced, where
    Minv=inv(M) and M is the integer matrix such that the supercell
    lattice vectors are A_S = M@A_0. For M=diag(n1,n2,n3) this reduces to
    dividing k by the supercell size along each axis. Nothing here cares
    about the dimensionality, and replicas/primal_indices are already
    filtered to the atoms actually present, so a supercell with atoms
    removed needs no separate treatment either.

    The matrix is sparse: every column (an orbital of the supercell) has
    exactly one nonzero entry, in the row of the primal orbital it is a
    copy of."""
    from scipy.sparse import csc_matrix
    natoms = len(primal_indices)
    ncols = natoms*norb_factor # orbitals in this supercell
    # column -> row in the primal-cell operator
    rows = np.repeat(primal_indices*norb_factor,norb_factor) + \
           np.tile(np.arange(norb_factor),natoms)
    dvecs = np.repeat(replicas,norb_factor,axis=0).astype(float) # per column
    Minv = inv(np.array(M,dtype=float))
    indptr = np.arange(ncols+1) # one nonzero entry per column
    def fun(k): # function generating the matrix
        kv = np.zeros(3) # kpoint, padded to three components
        ki = np.asarray(k,dtype=float).ravel()
        kv[0:len(ki)] = ki[0:3]
        phi = np.exp(1j*2.*np.pi*(dvecs@(Minv@kv))) # phase per column
        return csc_matrix((phi,rows,indptr),shape=(n0,ncols)) # sparse
    return fun
