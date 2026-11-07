# Quantics tensor cross interpolation (QTCI) of smooth fields on a uniform
# 2^R x 2^R (2D) or 2^R x 2^R x 2^R (3D) k-mesh, shared by the qtci
# integrations that run a simplex sum on top of the reconstructed fields:
# the DOS, the LDOS and the Green's function of dostk/simplexdos.py, the
# optical conductivity of conductivitytk/kubosimplex.py, the
# second-harmonic conductivity of conductivitytk/shgsimplex.py and the
# mean-field density matrix of qtcitk/densitymatrix_qtci.py. The fields are
# the band energies, the transition energies and the weights, which are
# smooth in k, so each one has a low rank, while the sharp resolvent is
# integrated afterwards.
#
# A point of the mesh is the integer vector (i_1..i_dim), each i_a in
# [0, 2^R), written in R bits, most significant first. The bits are the
# indices of the tensor train in one of two orders:
#   "interleaved": dim*R sites of dimension 2, scale by scale, the bits of
#     the coordinates of one scale next to each other (x1 y1 z1 x2 y2 ...);
#   "fused": R sites of dimension 2^dim, one per scale, the digit
#     sum_a b_a 2^(dim-1-a) holding the bits of every coordinate of that
#     scale.
# Both keep the scales in order, so the length scales of the field are
# separated along the train in the same way; they differ in the bonds
# inside one scale, which the fused order does not have.
import itertools

import numpy as np

# rank cap of qtci_stacked_fields in the qtci LDOS and optical
# conductivity. The rank of one TCI of all the fields grows with their
# number: on a 2x2 supercell with 80 conductivity fields it reached 460 at
# R = 7 and 572 at R = 8, so the cap of 400 of one field stopped the
# interpolation before its tolerance
# (future_development/qtci_optical_response.md)
stacked_maxbonddim = 800

# bit ordering of the 3D mesh, see the module comment. On the bonds between
# two scales, the only bonds the fused train has, both orders reach the
# same rank (3, 6, 6, 6, 4 on the simple cubic band at R = 6, at most 154
# against 159 on a layered two-band model). The interleaved train has
# sites of dimension 2 instead of 8, so it samples fewer points for the
# same rank: 3.5k against 13.1k diagonalizations of the 64^3 mesh of the
# cubic band, 232k against all 262k on a layered semimetal
# (future_development/qtci_dos_green.md, the section on 3D)
ordering3d = "interleaved"

_orderings = ("interleaved","fused")


class _Layout():
    """The map between the tensor indices and the points of the mesh"""
    def __init__(self,R,dim,ordering,nfield=None):
        if ordering is None: ordering = "interleaved" if dim<3 else ordering3d
        if ordering not in _orderings:
            raise ValueError("unknown bit ordering '"+str(ordering)+"' of "
                    "the quantics mesh; it must be one of "+str(_orderings))
        self.R = R; self.dim = dim; self.ordering = ordering
        self.n = 2**R
        scale = [(2**(R-1-s))*self.n**(dim-1-a) for s in range(R)
                for a in range(dim)] # contribution of each bit
        if ordering=="interleaved":
            dims = [2]*(dim*R)
            lin = [[0,c] for c in scale]
        else:
            dims = [2**dim]*R
            lin = [[sum(((c>>(dim-1-a))&1)*scale[s*dim+a]
                for a in range(dim)) for c in range(2**dim)] for s in range(R)]
        self.stacked = nfield is not None
        if self.stacked: # the field index, one more site before the mesh
            dims = [nfield]+dims
            lin = [[0]*nfield]+lin
        D = max(dims)
        self.lin = np.zeros((len(dims),D),dtype=np.int64)
        for s,row in enumerate(lin): self.lin[s,:len(row)] = row
        self.field = np.zeros((len(dims),D),dtype=np.int64)
        if self.stacked: self.field[0,:nfield] = np.arange(nfield)
        self.localdims = dims
    def encode(self,idx,f=None):
        """the tensor indices of the mesh point idx (and the field f)"""
        R = self.R; dim = self.dim
        bits = [[(idx[a]>>(R-1-s))&1 for a in range(dim)] for s in range(R)]
        if self.ordering=="interleaved": out = [b for bs in bits for b in bs]
        else: out = [sum(b<<(dim-1-a) for a,b in enumerate(bs)) for bs in bits]
        if self.stacked: out = [f]+out
        return out
    def ks(self,lin):
        """the k-points (len(lin),3) of the linear mesh indices lin"""
        idx = np.unravel_index(lin,(self.n,)*self.dim)
        ks = np.zeros((len(lin),3))
        for a in range(self.dim): ks[:,a] = idx[a]/self.n
        return ks
    def batch(self,L,Rr,ncent):
        """linear mesh index and field index of every (left, center,
        right) combination of a batch of the TCI, (len(L),ncombo,len(Rr))"""
        nl = len(L[0]) if len(L) else 0
        nr = len(Rr[0]) if len(Rr) else 0
        cd = self.localdims[nl:nl+ncent]
        La = np.array(L,dtype=np.int64).reshape(len(L),nl)
        Ra = np.array(Rr,dtype=np.int64).reshape(len(Rr),nr)
        combos = list(itertools.product(*[range(d) for d in cd]))
        Ca = np.array(combos,dtype=np.int64).reshape(len(combos),ncent)
        sl = np.arange(nl); sc = np.arange(nl,nl+ncent)
        sr = np.arange(nl+ncent,nl+ncent+nr)
        def part(tab,A,sites): return tab[sites,A].sum(axis=1)
        lin = (part(self.lin,La,sl)[:,None,None]+part(self.lin,Ca,sc)[None,:,None]
                +part(self.lin,Ra,sr)[None,None,:])
        fld = (part(self.field,La,sl)[:,None,None]
                +part(self.field,Ca,sc)[None,:,None]
                +part(self.field,Ra,sr)[None,None,:])
        return lin,fld,cd
    def to_mesh(self,full,nfield=None):
        """the full tensor of the train as (n^dim,) or (n^dim,nfield), in
        the row-major order of klist.kmesh"""
        R = self.R; dim = self.dim
        axes = [a+dim*s for a in range(dim) for s in range(R)]
        if nfield is None:
            full = full.reshape([2]*(dim*R))
            return np.transpose(full,axes).reshape(self.n**dim)
        full = full.reshape([nfield]+[2]*(dim*R))
        out = np.transpose(full,[0]+[1+a for a in axes]).reshape(nfield,-1)
        return np.ascontiguousarray(out.T)


class _MeshCache():
    """The values of evaluate at the visited points of the mesh, every
    point evaluated once, stored by linear mesh index"""
    def __init__(self,evaluate,layout):
        self.evaluate = evaluate
        self.layout = layout
        self.slot = np.full(layout.n**layout.dim,-1,dtype=np.int64)
        self.vals = None
        self.count = 0
    def ensure(self,lin):
        u = np.unique(np.asarray(lin,dtype=np.int64).ravel())
        new = u[self.slot[u]<0]
        if len(new)==0: return
        v = np.asarray(self.evaluate(self.layout.ks(new)))
        v = v.reshape(len(new),-1)
        if self.vals is None:
            self.vals = np.zeros((max(len(new),1024),v.shape[1]),dtype=v.dtype)
        if self.count+len(new)>len(self.vals): # grow the storage
            cap = max(2*len(self.vals),self.count+len(new))
            vals = np.zeros((cap,self.vals.shape[1]),
                    dtype=np.result_type(self.vals,v))
            vals[:self.count] = self.vals[:self.count]
            self.vals = vals
        self.vals[self.count:self.count+len(new)] = v
        self.slot[new] = np.arange(self.count,self.count+len(new))
        self.count += len(new)
    def get(self,lin,f):
        self.ensure(lin)
        return self.vals[self.slot[lin],f]


def _evaluator(cache,layout,field=None):
    """the BatchEvaluator of one field (field) or of all the fields"""
    from ..qutecipytk.tensortrain.batcheval import BatchEvaluator
    class Field(BatchEvaluator):
        def batchevaluate(self,L,Rr,ncent):
            lin,fld,cd = layout.batch(L,Rr,ncent)
            f = fld if field is None else field
            return cache.get(lin,f).reshape(len(L),*cd,len(Rr))
        def __call__(self,x):
            return self.batchevaluate([list(x)],[()],0).ravel()[0]
    return Field()


def _fulltensor(tci):
    """the full tensor of a tensor train as a flat array, contracted from
    both ends to the middle, so that no intermediate is larger than the
    result (the left-to-right contraction keeps a bond index on a tensor
    of half the size of the result)"""
    from ..qutecipytk.tensortrain.core import tensortrain
    ts = tensortrain(tci).sitetensors()
    ts = [t.reshape(t.shape[0],-1,t.shape[-1]) for t in ts]
    L = len(ts)
    if L==1: return ts[0].reshape(-1)
    logs = np.cumsum([np.log(t.shape[1]) for t in ts])
    mid = int(np.clip(np.searchsorted(logs,logs[-1]/2.)+1,1,L-1))
    left = ts[0].reshape(-1,ts[0].shape[2])
    for t in ts[1:mid]:
        left = (left@t.reshape(t.shape[0],-1)).reshape(-1,t.shape[2])
    right = ts[-1].reshape(ts[-1].shape[0],-1)
    for t in ts[L-2:mid-1:-1]:
        right = (t.reshape(-1,t.shape[2])@right).reshape(t.shape[0],-1)
    return (left@right).reshape(-1)


def _coarse_points(layout,ncoarse):
    """points of the fine mesh on a coarse grid of about ncoarse^2 points
    in any dimension (ncoarse per axis in 2D), in row-major order"""
    n = layout.n; dim = layout.dim
    nc0 = min(n,max(1,int(round(ncoarse**(2./dim)))))
    stride = n//nc0
    grid = np.meshgrid(*([np.arange(nc0)*stride]*dim),indexing="ij")
    idx = np.stack([g.ravel() for g in grid],axis=1)
    lin = np.ravel_multi_index(tuple(idx.T),(n,)*dim)
    return idx,lin


def qtci_fields(evaluate,R,nfield,tolerance=1e-6,maxbonddim=400,ncoarse=32,
        dtype=np.float64,dim=2,ordering=None):
    """Reconstruct the fields evaluate(ks) -> (len(ks),nfield) on the
    2^R x 2^R mesh (2^R x 2^R x 2^R with dim=3; row-major, as klist.kmesh)
    from one quantics tensor cross interpolation per field. Every field
    shares one cache of evaluations, so a k-point is diagonalized once
    whatever the number of fields that ask for it. A coarse scan of about
    ncoarse^2 points of the fine mesh (ncoarse x ncoarse in 2D) seeds the
    pivot of each field at its largest value, and a field that vanishes on
    the whole scan is set to zero. ordering is the bit ordering of the
    mesh, "interleaved" or "fused" (see the module comment; None takes
    interleaved in 2D and ordering3d in 3D). Returns the fields
    (n^dim,nfield), the number of distinct k-points evaluated and the
    largest bond dimension per field. The fields are real by default;
    dtype=np.complex128 interpolates complex fields, one TCI per field
    instead of one for the real part and one for the imaginary part."""
    from ..qutecipytk import crossinterpolate2
    layout = _Layout(R,dim,ordering)
    cache = _MeshCache(evaluate,layout)
    idx0,lin0 = _coarse_points(layout,ncoarse)
    f0 = cache.get(lin0,slice(None))
    out = np.zeros((layout.n**dim,nfield),dtype=dtype)
    ranks = []
    for f in range(nfield):
        if np.max(np.abs(f0[:,f]))==0.:
            ranks.append(0); continue
        i0 = int(np.argmax(np.abs(f0[:,f])))
        piv = [layout.encode(idx0[i0])]
        tci,rk,errors = crossinterpolate2(dtype,_evaluator(cache,layout,f),
                layout.localdims,initialpivots=piv,tolerance=tolerance,
                maxbonddim=maxbonddim,pivotsearch="full")
        ranks.append(int(max(rk)) if len(rk) else 0)
        out[:,f] = layout.to_mesh(_fulltensor(tci))
    return out,cache.count,ranks


def qtci_stacked_fields(evaluate,R,nfield,tolerance=1e-6,maxbonddim=400,
        ncoarse=32,dtype=np.float64,dim=2,ordering=None):
    """The fields of qtci_fields from one quantics tensor cross
    interpolation of all of them at once: the field index is an extra
    tensor site, placed before the bits of the mesh, so every field
    shares the pivots in k and the tolerance is relative to the largest
    value of any field. A field that is zero up to rounding (a
    spin-off-diagonal projector entry of a collinear magnet) then costs
    nothing, while one TCI per field normalizes it by its own maximum and
    interpolates the rounding noise at full rank. The coarse scan seeds one
    pivot per field at its largest value. dim and ordering are those of
    qtci_fields. Returns the fields (n^dim,nfield), the number of distinct
    k-points evaluated and the bond dimensions"""
    from ..qutecipytk import crossinterpolate2
    layout = _Layout(R,dim,ordering,nfield=nfield)
    cache = _MeshCache(evaluate,layout)
    idx0,lin0 = _coarse_points(layout,ncoarse)
    f0 = np.abs(cache.get(lin0,slice(None))) # (points,fields)
    if np.max(f0)==0.:
        return np.zeros((layout.n**dim,nfield),dtype=dtype),cache.count,[]
    piv = [layout.encode(idx0[int(np.argmax(f0[:,f]))],f)
            for f in range(nfield) if np.max(f0[:,f])>0.]
    tci,rk,errors = crossinterpolate2(dtype,_evaluator(cache,layout),
            layout.localdims,initialpivots=piv,tolerance=tolerance,
            maxbonddim=maxbonddim,pivotsearch="full")
    out = layout.to_mesh(_fulltensor(tci),nfield=nfield)
    return out,cache.count,[int(x) for x in rk]
