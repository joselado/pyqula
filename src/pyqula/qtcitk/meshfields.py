# Quantics tensor cross interpolation (QTCI) of smooth fields on a uniform
# 2^R x 2^R k-mesh, shared by the qtci integrations that run a simplex sum
# on top of the reconstructed fields: the DOS, the LDOS and the Green's
# function of dostk/simplexdos.py and the optical conductivity of
# conductivitytk/kubosimplex.py. The fields are the band energies, the
# transition energies and the weights, which are smooth in k, so each one
# has a low rank, while the sharp resolvent is integrated afterwards.
import numpy as np


def qtci_fields(evaluate,R,nfield,tolerance=1e-6,maxbonddim=400,ncoarse=32):
    """Reconstruct the real fields evaluate(ks) -> (len(ks),nfield) on the
    2^R x 2^R mesh (row-major, as klist.kmesh) from one quantics tensor
    cross interpolation per field. Every field shares one cache of
    evaluations, so a k-point is diagonalized once whatever the number of
    fields that ask for it. A coarse ncoarse x ncoarse scan on points of
    the fine mesh seeds the pivot of each field at its largest value, and
    a field that vanishes on the whole scan is set to zero. Returns the
    fields (n*n,nfield), the number of distinct k-points evaluated and the
    largest bond dimension per field."""
    from ..qutecipytk import crossinterpolate2
    from ..qutecipytk.tensortrain.core import tensortrain
    from ..qutecipytk.tensortrain.batcheval import BatchEvaluator
    import itertools
    n = 2**R
    cache = {} # grid index (i,j) -> the nfield values there
    def ensure(ij):
        new = [q for q in dict.fromkeys(ij) if q not in cache]
        if not new: return
        ks = np.zeros((len(new),3))
        ks[:,:2] = np.array(new,dtype=np.float64)/n
        vals = evaluate(ks)
        for q,v in zip(new,vals): cache[q] = v
    weights = 2**np.arange(R-1,-1,-1)
    def decode(keys): # interleaved bits (x,y), most significant first
        q = np.array(keys,dtype=np.int64).reshape(-1,R,2)
        return list(zip((q[:,:,0]@weights).tolist(),
            (q[:,:,1]@weights).tolist()))
    def encode(i,j):
        out = []
        for s in range(R): out += [(i>>(R-1-s))&1,(j>>(R-1-s))&1]
        return out
    localdims = [2]*(2*R)
    class Field(BatchEvaluator):
        def __init__(self,f): self.f = f
        def evalmany(self,keys):
            ij = decode(keys)
            ensure(ij)
            return np.array([cache[q][self.f] for q in ij])
        def __call__(self,x): return self.evalmany([tuple(x)])[0]
        def batchevaluate(self,L,Rr,ncent):
            nl = len(L[0]) if len(L) else 0
            cd = localdims[nl:nl+ncent]
            combos = list(itertools.product(*[range(d) for d in cd]))
            keys = [tuple(l)+c+tuple(r) for l in L for c in combos for r in Rr]
            return self.evalmany(keys).reshape(len(L),*cd,len(Rr))
    # coarse scan on points of the fine grid, which seeds the pivots
    nc0 = min(n,ncoarse)
    stride = n//nc0
    ij0 = [(i*stride,j*stride) for i in range(nc0) for j in range(nc0)]
    ensure(ij0)
    f0 = np.array([cache[q] for q in ij0])
    out = np.zeros((n*n,nfield))
    xaxes = list(range(0,2*R,2)); yaxes = list(range(1,2*R,2))
    ranks = []
    for f in range(nfield):
        if np.max(np.abs(f0[:,f]))==0.:
            ranks.append(0); continue
        i0 = int(np.argmax(np.abs(f0[:,f])))
        piv = [encode(*ij0[i0])]
        tci,rk,errors = crossinterpolate2(np.float64,Field(f),localdims,
                initialpivots=piv,tolerance=tolerance,maxbonddim=maxbonddim,
                pivotsearch="full")
        ranks.append(int(max(rk)) if len(rk) else 0)
        full = tensortrain(tci).fulltensor()
        out[:,f] = np.transpose(full,xaxes+yaxes).reshape(n*n)
    return out,len(cache),ranks
