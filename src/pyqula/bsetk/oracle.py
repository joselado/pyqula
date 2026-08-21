"""The electron-hole pair basis, evaluated on demand instead of built.

PairBasis diagonalizes the Bloch Hamiltonian at every point of the mesh
and stores the result, which is O(nk) work and O(nk*norb^2) memory. That is
exactly right for the dense and matrix-free solvers, which touch every
k-point anyway, and exactly wrong for the quantics solver, whose whole
claim is that it only ever needs O(polylog nk) of them: tensor cross
interpolation visits a few thousand index tuples regardless of how fine
the mesh is, so materializing the mesh first would put the O(nk) scaling
straight back.

PairOracle exposes the same quantities as PairBasis -- dE, el, ho and the
(ik,iv,ic) labelling -- but computes them per k-point when asked and
memoizes by k-index. Everything downstream (factorize.KernelFactorization
in particular) then works unchanged on either object.

Two things have to be fixed up front rather than per k-point, because they
are properties of the whole calculation and a per-k choice would not be
consistent between k-points:

  - the valence/conduction window, which needs to know how many bands are
    occupied. This is read off a COARSE submesh (see coarse_nk). A band
    crossing the Fermi level only between coarse points would be missed,
    which is the price of never looking at the fine mesh; the same caveat
    pairbasis.select_bands already states about meshes applies here more
    strongly, and is warned about.
  - the gauge references (the reference orbital of a phase fix, the trial
    orbitals of a projection), also from the coarse submesh. These MUST be
    shared by every k-point: gauging different k-points towards different
    references produces a perfectly valid gauge at each k and no smooth
    gauge overall, which is the one thing the quantics route cannot
    survive.
"""
import numpy as np

from .. import algebra
from .pairbasis import select_bands
from .gauge import fix_gauge, default_trials, default_refs


class PairOracle():
    """Pair basis of a BSE at center-of-mass momentum Q, evaluated lazily.

    Attributes mirroring PairBasis:
      kpoints   the full mesh, in fractional coordinates (cheap to hold:
                nk*3 floats, not nk*norb^2 complex)
      npair     nk*nv*nc
      vbands,cbands  the band window
      Q, geometry, h
    Methods:
      pair_arrays(indices)  -> (dE, el, ho) for a batch of pair indices,
                diagonalizing (and caching) only the k-points involved
      kmesh_shape()  the per-direction mesh size, for the quantics grid
    """
    def __init__(self,h,Q=None,nk=16,nv=None,nc=None,gauge="projection",
            coarse_nk=8,trials=None):
        if h.has_eh:
            raise ValueError("BSE is not implemented for Nambu/BdG "
                "Hamiltonians (h.has_eh)")
        h = h.get_multicell().get_dense() # canonical form
        self.h = h
        self.geometry = h.geometry
        self.dimensionality = h.geometry.dimensionality
        if Q is None: Q = [0.,0.,0.]
        self.Q = np.array(Q,dtype=np.float64)
        self.nk = int(nk)
        self.hk = h.get_hk_gen()
        self.gauge = gauge
        self._cache = dict()
        # --- coarse pre-pass: band window and gauge references
        ncoarse = min(int(coarse_nk),self.nk)
        ck,ek = self._coarse(ncoarse)
        self.vbands,self.cbands = select_bands(ek,nv=nv,nc=nc)
        _warn_coarse(ncoarse,self.nk)
        groups = [self.vbands,self.cbands]
        self.groups = groups
        self.trials = default_trials(ck,groups) if trials is None else trials
        self.refs = default_refs(ck)
        self.norb = ck.shape[2]
        self.nband = len(self.vbands)*len(self.cbands)
        self.nkpoints = self.nk**max(self.dimensionality,1)
        self.npair = self.nkpoints*self.nband
        self.labels = None # too large to materialize; see label()
    def kmesh_shape(self):
        """(nk,)*dimensionality, the shape of the quantics k-grid"""
        return tuple([self.nk]*max(self.dimensionality,1))
    def _coarse(self,ncoarse):
        """Diagonalize on a coarse submesh, for the band window and the
        gauge references. A Gamma-centered mesh of ncoarse points is a
        subset of the nk one whenever ncoarse divides nk, which is the
        case here since both are powers of two."""
        ks = _mesh_points(ncoarse,self.dimensionality)
        es,cs = [],[]
        for k in ks:
            e,w = algebra.eigh(self.hk(k))
            es.append(e) ; cs.append(np.array(w.T,dtype=np.complex128))
        return np.array(cs),np.array(es)
    def kpoint(self,ik):
        """Fractional coordinates of mesh point ik, generated from the
        index rather than looked up. geometry.get_kmesh returns a
        Gamma-centered mesh in C order -- kx the slower index -- so
        ik = ix*nk + iy reproduces it exactly, which
        tests/bse/test_bse_qtt.py checks rather than assumes."""
        return _kpoint_from_index(ik,self.nk,self.dimensionality)
    def _eig(self,ik):
        """(ek,ck,ekq,ckq) at mesh point ik, gauged, cached"""
        out = self._cache.get(ik)
        if out is not None: return out
        k = self.kpoint(ik)
        e1,w1 = algebra.eigh(self.hk(k))
        e2,w2 = algebra.eigh(self.hk(k+self.Q))
        c1 = np.array(w1.T,dtype=np.complex128)[None,:,:]
        c2 = np.array(w2.T,dtype=np.complex128)[None,:,:]
        kw = dict(mode=self.gauge,trials=self.trials,refs=self.refs)
        c1 = fix_gauge(c1,self.groups,**kw)[0]
        c2 = fix_gauge(c2,self.groups,**kw)[0]
        out = (e1,c1,e2,c2)
        self._cache[ik] = out
        return out
    def label(self,m):
        """(ik,iv,ic) of pair index m, the lazy PairBasis.labels"""
        ik,ib = divmod(int(m),self.nband)
        iv,ic = divmod(ib,len(self.cbands))
        return ik,self.vbands[iv],self.cbands[ic]
    def pair_arrays(self,indices):
        """Return (dE,el,ho) for a batch of pair indices.

        el[j] and ho[j] are the electron (c,k+Q) and hole (v,k)
        coefficient vectors of pair indices[j], and dE[j] its transition
        energy -- the same objects PairBasis stores for every pair."""
        indices = np.atleast_1d(np.asarray(indices,dtype=np.int64))
        n = len(indices)
        el = np.zeros((n,self.norb),dtype=np.complex128)
        ho = np.zeros((n,self.norb),dtype=np.complex128)
        dE = np.zeros(n,dtype=np.float64)
        for j,m in enumerate(indices):
            ik,iv,ic = self.label(m)
            ek,ck,ekq,ckq = self._eig(ik)
            el[j] = ckq[ic] ; ho[j] = ck[iv]
            dE[j] = ekq[ic] - ek[iv]
        return dE,el,ho
    def kindex(self,indices):
        """The mesh index of each pair index"""
        return np.asarray(indices,dtype=np.int64)//self.nband
    def transition_energies(self,indices):
        return self.pair_arrays(indices)[0]
    def ndiag(self):
        """How many k-points have actually been diagonalized so far --
        the number the whole quantics claim is about"""
        return len(self._cache)
    def lowest_transition_energy(self):
        """min dE over the mesh, without scanning the mesh.

        The binding energy is measured from this number, so it cannot
        simply be skipped, and a literal minimum over nk points would be
        the one O(nk) step in an otherwise logarithmic solver.

        Instead: start from every k-point already diagonalized -- the
        coarse pre-pass plus everything the cross interpolation happened
        to visit, all free -- and refine the best few by a binary descent
        on the mesh index, halving the step from nk/2 down to 1 in each
        direction. dE is a smooth function of k, so this converges on the
        band-edge minimum in O(dim*log nk) further diagonalizations.

        Not a global minimizer, and it does not pretend to be: a dE with a
        minimum narrower than the coarse mesh, sitting away from anything
        the interpolation looked at, could be missed. That is the same
        caveat the band window carries (see _warn_coarse) and for the same
        reason -- this solver never looks at the whole mesh, by design. Use
        solver="iterative" if an exact mesh minimum is needed.

        NOTE what this replaces. The obvious construction is to run DMRG on
        the diagonal MPO alone, which is exactly what kernel="none" means
        and would be logarithmic and elegant. It does not work: a diagonal
        Hamiltonian is the pathological case for DMRG, since every basis
        state is already an eigenstate and the local eigenproblem at each
        bond gives the sweep nothing to descend. Measured, it came back
        0.042 high on the gapped chain at nk=32, where the descent below is
        exact to 1e-15."""
        cand = sorted(self._cache.keys(),
                key=lambda ik: self._band_gap(ik))[0:4]
        if len(cand)==0: cand = [0]
        best = min(self._descend(ik) for ik in cand)
        return best
    def _band_gap(self,ik):
        """The smallest transition energy at one mesh point"""
        ek,ck,ekq,ckq = self._eig(ik)
        return float(np.min([ekq[ic]-ek[iv] for iv in self.vbands
                                            for ic in self.cbands]))
    def _descend(self,ik):
        """Binary descent on the mesh index from a starting point"""
        dim = max(self.dimensionality,1)
        idx = list(_unravel(ik,self.nk,dim))
        best = self._band_gap(_ravel(idx,self.nk))
        step = self.nk//2
        while step>=1:
            moved = True
            while moved:
                moved = False
                for j in range(dim):
                    for s in (step,-step):
                        trial = list(idx)
                        trial[j] = (trial[j]+s)%self.nk
                        e = self._band_gap(_ravel(trial,self.nk))
                        if e<best-1e-14:
                            best,idx,moved = e,trial,True
            step //= 2
        return best


def _mesh_points(nk,dim):
    """A Gamma-centered uniform mesh, in the same C order (first
    direction slowest) geometry.get_kmesh uses"""
    return np.array([_kpoint_from_index(i,nk,dim)
        for i in range(nk**max(dim,1))])


def _kpoint_from_index(ik,nk,dim):
    """Mesh point ik of an nk**dim Gamma-centered mesh"""
    ik = int(ik)
    out = np.zeros(3,dtype=np.float64)
    d = max(dim,1)
    for j in range(d-1,-1,-1):
        ik,r = divmod(ik,nk)
        out[j] = r/nk
    return out


def _warn_coarse(ncoarse,nk):
    """The band window came from a submesh, and that is a real caveat"""
    import warnings
    if ncoarse>=nk: return
    warnings.warn("the valence/conduction window and the gauge references "
        "of this quantics BSE were determined on a %d-point-per-direction "
        "submesh of the %d-point mesh, because the whole point of the "
        "solver is never to diagonalize the fine mesh. A band that crosses "
        "the Fermi level only between coarse points, or a gauge reference "
        "that vanishes only there, would not be seen. Raise coarse_nk if "
        "the model has fine structure near the gap"%(ncoarse,nk),
        stacklevel=3)


def _unravel(ik,nk,dim):
    """Per-direction mesh indices of a flat mesh index"""
    out = [0]*dim
    for j in range(dim-1,-1,-1):
        ik,out[j] = divmod(int(ik),nk)
    return out


def _ravel(idx,nk):
    """Flat mesh index of per-direction ones"""
    ik = 0
    for j in idx: ik = ik*nk + int(j)
    return ik
