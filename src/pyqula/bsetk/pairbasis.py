import numpy as np
from .. import algebra


class PairBasis():
    """Electron-hole pair basis of a BSE calculation at a fixed
    center-of-mass momentum Q.

    Two index sets are built over the same loop (ik,iv,ic), so both reuse
    the same two diagonalizations per k-point:

      - resonant pairs, the excitations  c^dag_{c,k+Q} c_{v,k}, with
        energy  e_{c,k+Q} - e_{v,k}
      - antiresonant pairs, the de-excitations c^dag_{v,k+Q} c_{c,k},
        with energy e_{c,k} - e_{v,k+Q}

    The antiresonant set is the resonant set at -Q, relabelled by k -> k+Q;
    keeping it as a separate (but identically indexed) set is what lets the
    full non-Tamm-Dancoff BSE at finite Q be assembled without any further
    momentum bookkeeping. At Q=0 the two sets coincide."""
    def __init__(self,h,Q=None,nk=10,nv=None,nc=None):
        if h.has_eh:
            raise ValueError("BSE is not implemented for Nambu/BdG "
                "Hamiltonians (h.has_eh): a superconducting mean field "
                "needs a different two-particle structure, not the "
                "electron-hole pair basis built here")
        h = h.get_multicell().get_dense() # canonical form
        self.h = h
        self.geometry = h.geometry
        if Q is None: Q = [0.,0.,0.]
        self.Q = np.array(Q,dtype=np.float64)
        self.kpoints = np.array(h.geometry.get_kmesh(nk=nk),dtype=np.float64)
        hk = h.get_hk_gen() # Bloch Hamiltonian generator
        eks,cks,ekqs,ckqs = [],[],[],[]
        for k in self.kpoints: # loop over the mesh
            e1,w1 = algebra.eigh(hk(k))
            e2,w2 = algebra.eigh(hk(k+self.Q))
            eks.append(e1) ; cks.append(np.array(w1.T,dtype=np.complex128))
            ekqs.append(e2) ; ckqs.append(np.array(w2.T,dtype=np.complex128))
        self.ek = np.array(eks) # (nk,norb) energies at k
        self.ck = np.array(cks) # (nk,norb,norb), ck[ik][n] = C^{n,k}
        self.ekq = np.array(ekqs) # energies at k+Q
        self.ckq = np.array(ckqs) # coefficients at k+Q
        self.vbands,self.cbands = select_bands(self.ek,nv=nv,nc=nc)
        self.build()
    def build(self):
        """Build the flattened pair index and its coefficient arrays"""
        nk = len(self.kpoints)
        vb,cb = self.vbands,self.cbands
        norb = self.ck.shape[1]
        npair = nk*len(vb)*len(cb)
        self.npair = npair
        # resonant: hole (v,k), electron (c,k+Q)
        el = np.zeros((npair,norb),dtype=np.complex128)
        ho = np.zeros((npair,norb),dtype=np.complex128)
        # antiresonant: hole (v,k+Q), electron (c,k)
        elA = np.zeros((npair,norb),dtype=np.complex128)
        hoA = np.zeros((npair,norb),dtype=np.complex128)
        dE = np.zeros(npair,dtype=np.float64) # resonant energies
        dEA = np.zeros(npair,dtype=np.float64) # antiresonant energies
        kindex = np.zeros(npair,dtype=np.int64) # k-point of each pair
        labels = [] # (ik,iv,ic) of each pair, for postprocessing
        m = 0
        for ik in range(nk): # loop over k-points
            for iv in vb: # loop over valence bands
                for ic in cb: # loop over conduction bands
                    el[m] = self.ckq[ik][ic] ; ho[m] = self.ck[ik][iv]
                    elA[m] = self.ck[ik][ic] ; hoA[m] = self.ckq[ik][iv]
                    dE[m] = self.ekq[ik][ic] - self.ek[ik][iv]
                    dEA[m] = self.ek[ik][ic] - self.ekq[ik][iv]
                    kindex[m] = ik
                    labels.append((ik,iv,ic))
                    m += 1
        self.el,self.ho = el,ho
        self.elA,self.hoA = elA,hoA
        self.dE,self.dEA = dE,dEA
        self.kindex = kindex
        self.labels = labels


def select_bands(ek,nv=None,nc=None):
    """Return the valence and conduction band indices to include.

    Bands are split at the Fermi energy, which pyqula fixes at zero (the
    same convention chitk/chiAB.py's occupations use), so valence means
    e<0 and conduction e>0. The split must be the same at every k-point --
    a metal, or a filling that puts a band across zero somewhere in the
    mesh, has no well-defined electron-hole pair basis and is rejected
    here rather than silently producing pairs of ill-defined character.

    Note the check is made on the mesh, not on the continuum band
    structure: a semimetal whose nodes happen to fall between mesh points
    (graphene on a mesh that misses K, say) looks gapped here and will be
    accepted. That is the honest thing to do -- the BSE really is being
    solved on this mesh and on no other -- but it does mean a small
    apparent binding energy from such a calculation is a statement about
    the mesh rather than about the material.

    nv/nc restrict the window to the nv highest valence and nc lowest
    conduction bands; None means take all of them."""
    nocc = np.sum(ek<0.,axis=1) # number of occupied bands at each k
    if len(np.unique(nocc))!=1:
        raise ValueError("the number of occupied bands is not the same at "
            "every k-point (found %s), so this Hamiltonian has no gap at "
            "the Fermi energy (zero) on this mesh and no electron-hole "
            "pair basis can be defined. BSE needs a gapped reference "
            "state; converge the mean field into an insulating solution, "
            "or use a denser/coarser mesh that does not straddle a band "
            "crossing"%(sorted(set(nocc.tolist())),))
    nocc = int(nocc[0])
    norb = ek.shape[1]
    if nocc==0 or nocc==norb:
        raise ValueError("no occupied (or no empty) bands at the Fermi "
            "energy, so there are no electron-hole pairs to build")
    gap = np.min(ek[:,nocc]) - np.max(ek[:,nocc-1]) # indirect gap
    if gap<=0.:
        raise ValueError("the reference Hamiltonian is not gapped "
            "(indirect gap %g <= 0), BSE needs a gapped reference"%gap)
    vb = list(range(nocc)) # all valence bands
    cb = list(range(nocc,norb)) # all conduction bands
    if nv is not None: vb = vb[-nv:] # nv highest valence bands
    if nc is not None: cb = cb[:nc] # nc lowest conduction bands
    if len(vb)==0 or len(cb)==0:
        raise ValueError("empty band window (nv=%s, nc=%s)"%(nv,nc))
    return vb,cb
