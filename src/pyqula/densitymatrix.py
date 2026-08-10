from __future__ import print_function, division
import numpy as np
from numba import jit
from . import parallel

delta_dm = 1e-6 # default energy smearing for density matrix
# ds not None does not use it yet

dm_mode = "accumulate" # default mode to compute density matrix

# accumulate is the new mode, it may be worth checking
# if it yields the same results as simultaneous

def full_dm(h,T=delta_dm,dm_mode=dm_mode,**kwargs):
    """Compute the full density matrix"""
    if T==0.: T = 1e-15 # just very small 
    if dm_mode=="accumulate":
        return full_dm_accumulate(h,delta=T,**kwargs)
    elif dm_mode=="simultaneous":
        return full_dm_simultaneous(h,delta=T,**kwargs)
    else: raise NotImplementedError

# it may be worth to implement some adaptive integration with quad_vec

def full_dm_accumulate(h,nk=10,fermi=0.0,
        delta=delta_dm,
        ds=None,batch_size=16):
    """Compute the full density matrix by adding the
    contributions to the matrix kpoint by kpoint. K-points are processed
    in batches: each batch is diagonalized in parallel across numba
    threads (see htk.eigenvectors.parallel_diagonalization), then every
    kpoint's density-matrix contribution is computed in parallel too
    (see dmtk.fulldm.full_dm_batch_vectorized) and pooled with a single
    sum at the end of the batch -- no interprocess communication anywhere
    in this function, unlike parallel.pcall, so it scales with the number
    of threads instead of being dominated by IPC overhead. batch_size
    bounds how many k-points' eigenvectors are held in memory at once,
    keeping the memory footprint low regardless of how dense the k-mesh
    is."""
    from .htk.eigenvectors import parallel_diagonalization
    hk = h.get_hk_gen() # get the Hamiltonian generator
    ks = np.array(h.geometry.get_kmesh(nk=nk)) # get the mesh
    fac = 1./len(ks) # normalization
    dm = None # accumulator, one slot per batch
    for i0 in range(0,len(ks),batch_size): # loop over batches of kpoints
        kbatch = ks[i0:i0+batch_size]
        mats = np.array([hk(k) for k in kbatch]) # k-Hamiltonians in this batch
        es_batch,vs_batch = parallel_diagonalization(mats) # diagonalize in parallel
        es_batch = es_batch-fermi # substract fermi energy
        if ds is None:
            contribs = full_dm_batch_vectorized(es_batch,vs_batch,delta=delta) # one per kpoint, in parallel
            batch_total = np.sum(contribs,axis=0) # pool the batch's contributions
        else:
            n = vs_batch.shape[1]
            batch_total = np.zeros((len(ds),n,n),dtype=np.complex128)
            for idir,d in enumerate(ds): # each direction, batched over kpoints
                contribs = full_dm_batch_d_vectorized(es_batch,vs_batch,kbatch,
                        np.array(d,dtype=np.float64),delta=delta) # one per kpoint, in parallel
                batch_total[idir] = np.sum(contribs,axis=0) # pool the batch's contributions
        dm = batch_total if dm is None else dm+batch_total # pool across batches
    dm = dm*fac # renormalize
    if ds is None: return dm # return the single array
    else: # if ds were given
        outd = dict() # dictionary
        for i in range(len(ds)): outd[tuple(ds[i])] = dm[i,:,:] # as dictionary
        return outd
    


def full_dm_accumulate_sparse(h,pairs,nk=10,fermi=0.0,
        delta=delta_dm,batch_size=16,dense_fraction=0.01):
    """Sparse-position counterpart of full_dm_accumulate: for each
    direction, only computes the (row,col) entries listed in pairs[d]
    instead of the full (n,n) matrix (see dmtk.fulldm.full_dm_batch_d_sparse
    and selfconsistency.spinspin._build_sparse_pairs for why this is safe
    -- a short-range interaction matrix v[d] is mostly zero, so most of a
    full (n,n) density matrix at that direction is never read downstream).
    Still returns a dense {direction: (n,n)} dict, zero everywhere except
    at the requested pairs (or fully populated, for a direction that fell
    back to the dense kernel -- see dense_fraction), so it is a drop-in
    replacement for full_dm_accumulate(...,ds=list(pairs)) wherever only
    those entries are consumed -- used only by
    selfconsistency.spinspin._run_anisotropic_scf (Jinteraction/
    VJinteraction's shared SCF core), not by the generic (Vinteraction/
    SzSz/SxSx/SySy) path, which still gets the full matrix via
    full_dm_accumulate.

    dense_fraction: per-direction fallback threshold. full_dm_batch_d_sparse
    does asymptotically less work than the dense (n,n)@(n,n) matmul
    (full_dm_batch_d_vectorized) for a truly sparse direction, but it is a
    gather + elementwise-multiply-and-reduce, not a BLAS call, so its
    per-entry constant is much worse than a matmul's -- measured on a
    196-orbital system, the sparse kernel is ~7x *slower* than the dense
    one at 8.8% of n^2 requested entries (a common density for a bond
    direction that happens to fold entirely within one cell, e.g.
    second-neighbor bonds in a compact supercell) despite doing an order of
    magnitude fewer FLOPs, and the crossover is around 1-2% of n^2. Below
    dense_fraction*n^2 requested entries for a direction, use the sparse
    kernel and scatter its output into the (initially zero) container;
    above it, just run the dense kernel for that direction and keep its
    full result -- strictly more information than requested, but correct
    and, past the crossover, cheaper too."""
    from .htk.eigenvectors import parallel_diagonalization
    hk = h.get_hk_gen() # get the Hamiltonian generator
    ks = np.array(h.geometry.get_kmesh(nk=nk)) # get the mesh
    fac = 1./len(ks) # normalization
    n = h.intra.shape[0]
    threshold = dense_fraction*n*n
    outd = {d: np.zeros((n,n),dtype=np.complex128) for d in pairs}
    for i0 in range(0,len(ks),batch_size): # loop over batches of kpoints
        kbatch = ks[i0:i0+batch_size]
        mats = np.array([hk(k) for k in kbatch]) # k-Hamiltonians in this batch
        es_batch,vs_batch = parallel_diagonalization(mats) # diagonalize in parallel
        es_batch = es_batch-fermi # substract fermi energy
        _accumulate_dm_batch(outd,pairs,threshold,es_batch,vs_batch,kbatch,delta)
    for d in outd: outd[d] *= fac # renormalize
    return outd


def _accumulate_dm_batch(outd,pairs,threshold,es_batch,vs_batch,kbatch,delta):
    """Add one k-batch's contribution to outd (in place), choosing the
    sparse or dense kernel per direction -- the shared per-batch step of
    full_dm_accumulate_sparse and full_dm_accumulate_sparse_with_fermi."""
    for d,(rows,cols) in pairs.items():
        npairs = len(rows)
        if npairs==0: continue # nothing requested in this direction
        if npairs>threshold: # dense direction: the plain matmul wins
            contribs = full_dm_batch_d_vectorized(es_batch,vs_batch,kbatch,
                    np.array(d,dtype=np.float64),delta=delta)
            outd[d] += np.sum(contribs,axis=0) # pool the batch
        else:
            contribs = full_dm_batch_d_sparse(es_batch,vs_batch,kbatch,
                    np.array(d,dtype=np.float64),rows,cols,delta=delta)
            outd[d][rows,cols] += np.sum(contribs,axis=0) # pool the batch


def full_dm_accumulate_sparse_with_fermi(h,pairs,filling,nk=10,
        delta=delta_dm,batch_size=16,dense_fraction=0.01,max_memory_gb=2.0):
    """Like full_dm_accumulate_sparse, but also determines and returns the
    Fermi energy for `filling` from the SAME diagonalization used to build
    the density matrix, instead of paying for a second, independent
    diagonalization sweep first the way
    selfconsistency.spinspin._run_anisotropic_scf's callback_h
    (Hamiltonian.get_fermi4filling) otherwise would before calling
    get_dm/full_dm_accumulate_sparse on the (separately, again-diagonalized)
    shifted Hamiltonian. Shifting a Hamiltonian by a constant
    (H' = H - fermi*I) does not change its eigenVECTORS, only shifts the
    eigenvalues by that same constant -- so diagonalizing the UNSHIFTED h
    once, determining fermi from the pooled eigenvalues, then subtracting
    it from the already-computed eigenvalues before building the density
    matrix, is exactly equivalent to the two-diagonalization version, at
    (up to) half the diagonalization cost.

    Unlike full_dm_accumulate_sparse's own batching (which only ever needs
    one batch of eigenvectors in memory at a time, since fermi is already
    known there), this holds every batch's (es,vs,kbatch) for the whole
    k-mesh at once, since the Fermi energy needs every eigenvalue in the
    mesh before any density-matrix contribution can be computed -- unlike
    full_dm_accumulate's own batch_size, which bounds memory "regardless of
    how dense the k-mesh is" (that function's own docstring), this one does
    not, and the eigenvector memory for the whole mesh can get large for an
    unusually fine k-mesh (e.g. ~6GB for a 196-orbital system on a 100x100
    2D mesh). max_memory_gb guards against that: above it, this falls back
    to the same batch_size-bounded, memory-safe (but two-diagonalization)
    sequence full_dm_accumulate_sparse's own caller used before this
    function existed -- get_fermi4filling on `h` directly, then
    full_dm_accumulate_sparse on a shifted copy -- trading back the
    dedup for a bounded memory footprint only in that regime.

    Used only by selfconsistency.spinspin._run_anisotropic_scf for the
    normal-state (has_eh=False) case with mu=None (a Fermi-level search is
    actually needed) -- see that function's docstring for why the Nambu
    case is out of scope: BdG's own get_fermi4filling diagonalizes an
    entirely different (de-paired) Hamiltonian, not just a shifted copy of
    the one the density matrix comes from, so this trick does not apply
    there."""
    from .htk.eigenvectors import parallel_diagonalization
    from .filling import get_fermi_energy
    ks = np.array(h.geometry.get_kmesh(nk=nk)) # get the mesh
    n = h.intra.shape[0]
    if len(ks)*n*n*16 > max_memory_gb*1e9: # see max_memory_gb's docstring
        fermi = h.get_fermi4filling(filling,nk=nk)
        h_shifted = h.copy()
        h_shifted.shift_fermi(-fermi)
        dm = full_dm_accumulate_sparse(h_shifted,pairs,nk=nk,delta=delta,
                batch_size=batch_size,dense_fraction=dense_fraction)
        return dm,fermi
    hk = h.get_hk_gen() # get the Hamiltonian generator
    fac = 1./len(ks) # normalization
    threshold = dense_fraction*n*n
    batches = [] # (es_batch, vs_batch, kbatch) for the whole mesh
    all_es = []
    for i0 in range(0,len(ks),batch_size): # loop over batches of kpoints
        kbatch = ks[i0:i0+batch_size]
        mats = np.array([hk(k) for k in kbatch]) # k-Hamiltonians in this batch
        es_batch,vs_batch = parallel_diagonalization(mats) # diagonalize in parallel
        batches.append((es_batch,vs_batch,kbatch))
        all_es.append(es_batch.ravel())
    fermi = get_fermi_energy(np.concatenate(all_es),filling)
    outd = {d: np.zeros((n,n),dtype=np.complex128) for d in pairs}
    for es_batch,vs_batch,kbatch in batches:
        _accumulate_dm_batch(outd,pairs,threshold,es_batch-fermi,vs_batch,kbatch,delta)
    for d in outd: outd[d] *= fac # renormalize
    return outd,fermi


def full_dm_accumulate_sparse_local_fermi(h,pairs,filling,lam,nk=10,
        delta=delta_dm,batch_size=16,dense_fraction=0.01):
    """Sparse density matrix under a per-site local chemical potential
    (Lagrange multiplier) `lam`, for the Abrikosov-pseudofermion / hard
    local-occupation-constraint case (Savary & Balents, "Quantum Spin
    Liquids: a review", arXiv:1601.03742, Sec. 4.1): unlike the scalar
    Fermi search (Hamiltonian.get_fermi4filling /
    full_dm_accumulate_sparse_with_fermi), a per-SITE array `filling`
    cannot be enforced by sorting the pooled eigenvalues and index-picking
    one cut -- a generic eigenstate is spread over many sites, so there is
    no single energy cut that fixes every site's occupation independently.
    It has to be enforced by n_sites independent onsite potentials (one per
    site, applied identically to both spin channels via
    Hamiltonian.shift_fermi's existing per-site-array support -- see that
    function's docstring), tuned so that <n_i> = filling[i] at every site
    simultaneously.

    PERFORMANCE/CORRECTNESS TRADEOFF (why this function only takes ONE
    diagonalization, not several): a scalar Fermi shift is a RIGID shift of
    the whole spectrum -- it commutes with H, so it changes only the
    eigenVALUES, never the eigenVECTORS, which is exactly what lets
    full_dm_accumulate_sparse_with_fermi reuse a single diagonalization for
    both the Fermi search and the density matrix (see its own docstring). A
    per-site onsite potential does NOT commute with H's off-diagonal
    (hopping) part, so it generally changes the eigenVECTORS too -- there is
    no analogous "diagonalize once, read off lam from the same eigenvectors"
    trick here. Finding the n_sites lam_i's that exactly satisfy
    <n_i>=filling[i] is therefore a genuine nonlinear root-find that needs a
    fresh diagonalization per proposed lam -- filling.set_individual_filling
    already does this for a *static* (non-mean-field) Hamiltonian, via
    scipy.optimize.fsolve with a numerically-estimated Jacobian, at a cost
    of ~n_sites+1 diagonalizations for one call. That is fine there (it runs
    once), but prohibitive here: this function is called once per iteration
    of selfconsistency.spinspin._run_anisotropic_scf's own outer SCF
    fixed-point loop (already tens to hundreds of iterations), where
    n_sites+1 extra diagonalizations per iteration would multiply the
    total cost by that same factor for no asymptotic benefit -- the fixed
    point converged to is the same either way.

    So this function does not itself iterate lam to convergence. It takes
    ONE diagonalization at the caller-supplied `lam` (in practice, the
    previous outer SCF iteration's value -- warm-started, since lam should
    barely move once the mean field is close to converged, exactly the
    same reasoning that lets `mf` itself be only mixed, not re-solved, at
    every outer iteration) and returns the resulting density matrix plus
    the per-site occupation fraction, so the caller
    (_run_anisotropic_scf's array-filling branch) can take a cheap,
    Jacobian-free proportional update step
    (lam_i += step*(filling_i - occ_i)) and let lam co-converge with the
    mean field across the SAME outer loop, at the SAME one-diagonalization-
    per-iteration cost the scalar path already pays -- see
    _run_anisotropic_scf's docstring for the exact step used and why a full
    multi-dimensional Newton solve was rejected here.

    NORMALIZATION: `filling` (and the returned `occ`) use the SAME
    convention as the scalar `filling` kwarg everywhere else in this
    module -- a FRACTION of a site's own 2-orbital (spin up + spin down)
    capacity, so filling[i]=0.5 uniformly is exactly the array equivalent
    of scalar filling=0.5 (average 1 electron/site: half of the 2-orbital
    capacity, filled). filling[i]=1.0 (not 0.5) would be needed to request
    a FULLY occupied site (2 electrons) -- filling[i]=0.5 is "exactly one
    fermion per site". This is verified empirically against the scalar
    path (get_fermi4filling(0.5) -> a site's occupation from get_vev is
    1.0, i.e. 2*0.5) and is NOT the same convention
    filling.set_individual_filling/Hamiltonian.get_vev use internally (they
    compare directly against the raw electron COUNT, 0 to 2, not a 0-to-1
    fraction -- effectively off by a factor of 2 from this module's
    convention for the same physical target). That pre-existing function is
    not reused here for that reason (among others -- it is also, as of this
    writing, uncallable at all with a nonzero smearing due to an unrelated
    get_vev/densitymatrix.full_dm keyword-argument collision when it
    forwards delta=; independent of this work, not fixed here, see
    selfconsistency.spinspin's per-site-filling design notes).

    Returns (dm, occ): `dm` is full_dm_accumulate_sparse's usual
    {direction: (n,n)} dict, computed at the given (UN-updated) `lam` -- the
    caller is responsible for shifting whatever Hamiltonian this dm is
    associated with by this same lam before using it downstream, exactly
    paralleling how full_dm_accumulate_sparse_with_fermi's caller shifts by
    the fermi it returns. `occ` is the per-site occupation FRACTION, shape
    (n_sites,), read off dm[(0,0,0)]'s diagonal as (dm_uu+dm_dd)/2 for each
    site -- valid only because selfconsistency.spinspin._build_sparse_pairs
    always includes the full onsite (0,0,0) 2x2 spin block for every site
    unconditionally (see the last paragraph of its own docstring), so this
    diagonal is guaranteed populated regardless of which channels (V/J/U)
    are actually active."""
    h_shifted = h.copy()
    h_shifted.shift_fermi(-np.asarray(lam))
    dm = full_dm_accumulate_sparse(h_shifted,pairs,nk=nk,delta=delta,
            batch_size=batch_size,dense_fraction=dense_fraction)
    diag = np.real(np.diag(dm[(0,0,0)]))
    occ = (diag[0::2] + diag[1::2])/2.0 # (n_up+n_down)/2 per site, a fraction in [0,1]
    return dm,occ


def full_dm_simultaneous(h,nk=10,fermi=0.0,
        delta=delta_dm,
        ds=None):
    """Compute the full density matrix by first computing all the
    eigenvectors, and after adding all the contributions together.
    This can become memore expesive for large kmesh and moderate
    matrices"""
    if h.dimensionality == 0: fac = 1.
    elif h.dimensionality == 1: fac = 1./nk
    elif h.dimensionality == 2: fac = 1./nk**2
    elif h.dimensionality == 3: fac = 1./nk**3
    else: raise
    if ds is None: # no directions required
      es,vs = h.get_eigenvectors(nk=nk) # get eigenvectors
      es = es - fermi # shift by the Fermi energy
      return np.array(full_dm_python(es,np.array(vs),
                             delta=delta))*fac # call the function
    else: # directions required
      es,vs,ks = h.get_eigenvectors(nk=nk,kpoints=True) # get eigenvectors
      es = es - fermi # shift by the Fermi energy
      es = np.array(es,dtype=np.float64)
      vs = np.array(vs,dtype=np.complex128)
      ks = np.array(ks,dtype=np.float64) # to array
      ds_arr = np.array(ds,dtype=np.float64)
      n = h.intra.shape[0] # dimensionality
      out = full_dm_d_batch_vectorized(es,vs,ks,ds_arr,delta=delta)*fac
      outd = dict() # dictionary
      for i in range(len(ds)): outd[tuple(ds[i])] = out[i] # as dictionary
      return outd


from .dmtk.fulldm import full_dm_python
from .dmtk.fulldm import full_dm_python_d
from .dmtk.fulldm import full_dm_batch_vectorized
from .dmtk.fulldm import full_dm_batch_d_sparse
from .dmtk.fulldm import full_dm_batch_d_vectorized
from .dmtk.fulldm import full_dm_d_batch_vectorized




def restricted_dm(h,mode="KPM",pairs=[],
                   scale=10.0,npol=400,ne=None):
  """Calculate certain elements of the density matrix"""
  if h.dimensionality != 0 : raise
  if mode=="full": # full inversion and then select
    dm = full_dm(h) # Full DM
    outm = np.array([dm[j,i] for (i,j) in pairs]) # get the desired ones
    return outm # return elements
  elif mode=="KPM": # use Kernel polynomial method
    if ne is None: ne = npol*4
    from . import kpm
    xin = np.linspace(-.99*scale,0.0,ne) # input x array
    out = np.zeros(len(pairs),dtype=np.complex128)
    ii = 0
    for (i,j) in pairs: # loop over inputs
      (x,y) = kpm.dm_ij_energy(h.intra,i=i,j=j,scale=scale,npol=npol,
                      ne=ne,x=xin)
      out[ii] = np.trapezoid(y,x=x)/np.pi # pi is here so it normalizes to 0.5
      ii += 1
    return out
  else: raise
       
from . import algebra

def occupied_projector(m,delta=0.0):
    """Return a projector onto the occupied states"""
    (es,vs) = algebra.eigh(m) # diagonalize
    vs = vs.T # transpose
    return np.array(full_dm_python(es,np.array(vs)))

