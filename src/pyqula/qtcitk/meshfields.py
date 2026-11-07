# Quantics tensor cross interpolation (QTCI) of smooth fields on a uniform
# 2^R x 2^R k-mesh, shared by the qtci integrations that run a simplex sum
# on top of the reconstructed fields: the DOS, the LDOS and the Green's
# function of dostk/simplexdos.py, the optical conductivity of
# conductivitytk/kubosimplex.py, the second-harmonic conductivity of
# conductivitytk/shgsimplex.py and the mean-field density matrix of
# qtcitk/densitymatrix_qtci.py. The fields are the band energies, the
# transition energies and the weights, which are smooth in k, so each one
# has a low rank, while the sharp resolvent is integrated afterwards.
import numpy as np

# rank cap of qtci_stacked_fields in the qtci LDOS and optical
# conductivity. The rank of one TCI of all the fields grows with their
# number: on a 2x2 supercell with 80 conductivity fields it reached 460 at
# R = 7 and 572 at R = 8, so the cap of 400 of one field stopped the
# interpolation before its tolerance
# (future_development/qtci_optical_response.md)
stacked_maxbonddim = 800


def qtci_fields(evaluate,R,nfield,tolerance=1e-6,maxbonddim=400,ncoarse=32,
        dtype=np.float64):
    """Reconstruct the fields evaluate(ks) -> (len(ks),nfield) on the
    2^R x 2^R mesh (row-major, as klist.kmesh) from one quantics tensor
    cross interpolation per field. Every field shares one cache of
    evaluations, so a k-point is diagonalized once whatever the number of
    fields that ask for it. A coarse ncoarse x ncoarse scan on points of
    the fine mesh seeds the pivot of each field at its largest value, and
    a field that vanishes on the whole scan is set to zero. Returns the
    fields (n*n,nfield), the number of distinct k-points evaluated and the
    largest bond dimension per field. The fields are real by default;
    dtype=np.complex128 interpolates complex fields, one TCI per field
    instead of one for the real part and one for the imaginary part."""
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
    out = np.zeros((n*n,nfield),dtype=dtype)
    xaxes = list(range(0,2*R,2)); yaxes = list(range(1,2*R,2))
    ranks = []
    for f in range(nfield):
        if np.max(np.abs(f0[:,f]))==0.:
            ranks.append(0); continue
        i0 = int(np.argmax(np.abs(f0[:,f])))
        piv = [encode(*ij0[i0])]
        tci,rk,errors = crossinterpolate2(dtype,Field(f),localdims,
                initialpivots=piv,tolerance=tolerance,maxbonddim=maxbonddim,
                pivotsearch="full")
        ranks.append(int(max(rk)) if len(rk) else 0)
        full = tensortrain(tci).fulltensor()
        out[:,f] = np.transpose(full,xaxes+yaxes).reshape(n*n)
    return out,len(cache),ranks


def qtci_stacked_fields(evaluate,R,nfield,tolerance=1e-6,maxbonddim=400,
        ncoarse=32,dtype=np.float64):
    """The fields of qtci_fields from one quantics tensor cross
    interpolation of all of them at once: the field index is an extra
    tensor site, placed before the 2R bits of the mesh, so every field
    shares the pivots in k and the tolerance is relative to the largest
    value of any field. A field that is zero up to rounding (a
    spin-off-diagonal projector entry of a collinear magnet) then costs
    nothing, while one TCI per field normalizes it by its own maximum and
    interpolates the rounding noise at full rank. The coarse scan seeds one
    pivot per field at its largest value. Returns the fields (n*n,nfield),
    the number of distinct k-points evaluated and the bond dimensions"""
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
    def decode(keys): # (field, interleaved bits (x,y) most significant first)
        q = np.array(keys,dtype=np.int64)
        b = q[:,1:].reshape(-1,R,2)
        return q[:,0].tolist(),list(zip((b[:,:,0]@weights).tolist(),
            (b[:,:,1]@weights).tolist()))
    def encode(f,i,j):
        out = [f]
        for s in range(R): out += [(i>>(R-1-s))&1,(j>>(R-1-s))&1]
        return out
    localdims = [nfield]+[2]*(2*R)
    class Fields(BatchEvaluator):
        def evalmany(self,keys):
            fs,ij = decode(keys)
            ensure(ij)
            return np.array([cache[q][f] for f,q in zip(fs,ij)])
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
    f0 = np.abs(np.array([cache[q] for q in ij0])) # (points,fields)
    if np.max(f0)==0.: return np.zeros((n*n,nfield),dtype=dtype),len(cache),[]
    piv = [encode(f,*ij0[int(np.argmax(f0[:,f]))]) for f in range(nfield)
            if np.max(f0[:,f])>0.]
    tci,rk,errors = crossinterpolate2(dtype,Fields(),localdims,
            initialpivots=piv,tolerance=tolerance,maxbonddim=maxbonddim,
            pivotsearch="full")
    full = tensortrain(tci).fulltensor() # (field, bits)
    axes = [0]+[1+a for a in list(range(0,2*R,2))+list(range(1,2*R,2))]
    out = np.transpose(full,axes).reshape(nfield,n*n).T
    return np.ascontiguousarray(out),len(cache),[int(x) for x in rk]
