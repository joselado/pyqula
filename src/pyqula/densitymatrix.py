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
    else: raise # not implemented

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
      ks = np.array(ks) # to array
      n = h.intra.shape[0] # dimensionality
      out = parallel.pcall(lambda x: full_dm_python_d(es,vs,ks,x)*fac,ds)
      outd = dict() # dictionary
      for i in range(len(ds)): outd[tuple(ds[i])] = out[i] # as dictionary
      return outd


from .dmtk.fulldm import full_dm_python
from .dmtk.fulldm import full_dm_python_d
from .dmtk.fulldm import full_dm_batch_vectorized
from .dmtk.fulldm import full_dm_batch_d_sparse
from .dmtk.fulldm import full_dm_batch_d_vectorized




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
      out[ii] = np.trapz(y,x=x)/np.pi # pi is here so it normalizes to 0.5
      ii += 1
    return out
  else: raise
       
from . import algebra

def occupied_projector(m,delta=0.0):
    """Return a projector onto the occupied states"""
    (es,vs) = algebra.eigh(m) # diagonalize
    vs = vs.T # transpose
    return np.array(full_dm_python(es,np.array(vs)))

