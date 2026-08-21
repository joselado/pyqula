"""Quantics tensor-train BSE: qutecipy builds the MPO, pyitensor's DMRG
solves it.

The idea. Binary-encode the electron-hole pair index -- one tensor-train
site per bit of each k-direction, plus one site for the (v,c) band pair --
and represent the Tamm-Dancoff BSE operator as a matrix product operator
over those sites. A smooth function of k is low-rank in that encoding, so
the MPO's bond dimension stays bounded while the mesh is refined, and the
cost of the whole calculation grows with log(nk) instead of nk.

Three things have to be true for that to work, and all three are measured
rather than assumed (see below and bsetk/gauge.py):

  1. The kernel must be cheap to evaluate at an arbitrary (m,n). It is:
     bsetk/factorize.py shows the BSE kernel is a fixed-rank object, and
     bsetk/oracle.py evaluates the pair basis one k-point at a time. Cross
     interpolation then only ever asks for O(polylog nk) k-points.
  2. The operator must be low-rank in the quantics encoding. It is, but
     ONLY in a smooth gauge -- in the gauge algebra.eigh returns, the
     kernel is exactly incompressible.
  3. The problem handed to DMRG must be Hermitian and bounded below. The
     Tamm-Dancoff (resonant) block is; the full BSE matrix is not, which
     is why this solver is Tamm-Dancoff only.

How the MPO is built. Not by decomposing a dense matrix -- that would
defeat the entire purpose -- but by cross-interpolating the matrix-element
function directly, which is what qutecipy is for and what TensorBinding
(arXiv:2607.00991) does for tight-binding Hamiltonians. Two separate
objects are interpolated:

  - the INTERACTION kernel X - D alone, as a function of the fused
    (row,column) quantics index;
  - the transition energies dE, as a function of the row index alone,
    which then become a diagonal MPO.

They are interpolated separately and summed exactly (no truncation) at the
end, and that separation is not cosmetic. The diagonal is O(gap) while
each element of the interaction is O(1/N), so on a fine mesh a single
relative-tolerance cross interpolation of A = dE + X - D would discard the
entire interaction -- and hand back the independent-particle spectrum,
looking perfectly converged. Interpolating the kernel on its own puts the
tolerance where the physics is.

Measured, maximum quantics rank of the kernel's factors at tolerance 1e-6,
as the mesh is refined 16-fold (bsetk/gauge.py has the full tables):
1D spinless chain 8 -> 8 -> 8; 2D spinless honeycomb 57 -> 62 -> 63; 2D
spinful honeycomb, projection gauge, 182 -> 248 -> 274. In the raw gauge
the same numbers grow proportionally to nk.

WHEN THIS SOLVER IS THE RIGHT ONE, measured rather than assumed. It wants
a NARROW BAND WINDOW on a FINE k-MESH, and it loses outside that.

  - 1D, nv*nc = 1: the good case. The mesh grows 256x, the work 8.7x, the
    wall time not at all (table in bsetk/qtt's roadmap entry).
  - 2D, primitive cell: slower than bsetk/iterative.py at every mesh
    reachable here -- 0.7 s against 103 s at 32x32, tolerance 1e-4. Not
    because the quantics side got worse, but because the exact solver got
    better: 4096 pairs cost it 3.1 s in 2D against 24 s in 1D. The ratio
    does improve with mesh (150x at 32^2, 76x at 64^2), so a crossover
    exists somewhere beyond what was measured.
  - A SUPERCELL is the wrong shape entirely. A 3x3 supercell has 9
    valence and 9 conduction bands, and a band label is not a smooth
    coordinate the way k is -- its digits carry no multi-scale structure,
    so the MPO rank SATURATES. Measured on that model at nk=4: kernel
    cross interpolation 450 s, MPO bond dimension 729, which is the
    maximum possible at that cut. The same physics from the primitive
    cell at 3x the mesh keeps nv*nc = 1 and compresses. Use the primitive
    cell with a fine mesh; if a supercell is unavoidable, narrow the
    window with nv/nc, or use solver="iterative", which does not care (it
    did the full window in 9.1 s against the dense solver's 96.5 s).

Benchmark. CLAUDE.md asks for a comparison against an existing open
implementation and there is not one for this construction: Xatu
(arXiv:2307.01572), whose BSE formalism bsetk/ follows, uses a dense
solver, and TensorBinding, the closest tensor-network tight-binding code,
has no exciton module. The substitutes are internal and are what
tests/bse/test_bse_qtt.py runs: agreement with the dense solver at small
nk, agreement with the matrix-free solver (bsetk/iterative.py, exact and
independent of every tensor-train choice) past the dense wall, and the
gauge invariance of the spectrum.

References:
  arXiv:1602.02646  BSE eigenproblem via low-rank kernel factorization and
                    QTT eigenvectors -- the same construction as here
  arXiv:2410.22975  quantics tensor trains for Bethe-Salpeter-type
                    equations
  arXiv:2607.00991  quantics cross interpolation of tight-binding
                    Hamiltonians into MPOs
"""
import numpy as np

from .interaction import bare_interaction, interaction_at_q
from .oracle import PairOracle


def solve_qtt(h,V=None,Q=None,nk=16,nv=None,nc=None,kernel="full",
        neig=1,gauge="projection",screening=None,nkW=None,channel="charge",
        tolerance=1e-6,maxbonddim=None,unfolding="grouped",coarse_nk=8,
        nsweep=24,maxdim=100,cutoff=1e-10,weight=None,seed=0,**kwargs):
    """Solve the Tamm-Dancoff BSE with a quantics MPO and DMRG.

    Returns (oracle,W,Wx,(energies,amplitudes)) so that BSE can present
    the same attributes it does for the other solvers.

    tolerance   cross-interpolation tolerance of the kernel MPO, RELATIVE
                to the largest kernel element sampled. 1e-6 is the value
                the rank measurements above were made at.
    maxbonddim  hard cap on the MPO bond dimension (None = only tolerance)
    maxdim      cap on the MPS bond dimension inside DMRG
    nsweep      DMRG sweeps
    unfolding   "grouped" (all kx bits, then all ky bits) or "interleaved"
                (by scale). Grouped is the default because it measured
                better on every 2D model tried here -- dE(k) on the gapped
                honeycomb at nk=128^2, tolerance 1e-6, has rank 16 grouped
                against 25 interleaved. That is the opposite of the usual
                rule of thumb for multivariate quantics functions, so it
                is a knob rather than a hard-coded choice.
    """
    if screening is not None:
        raise ValueError("solver='qtt' takes a real-space interaction. A "
            "tabulated screened interaction (screening='rpa'/'crpa') has "
            "no fixed-rank real-space form -- inverse transforming it over "
            "the mesh gives nk lattice vectors, so the kernel's rank would "
            "grow with the mesh. Build the screened interaction "
            "separately, truncate it with "
            "ScreenedInteraction.get_dict(cutoff=...), and pass the result "
            "as V=")
    nk = int(nk)
    if nk&(nk-1)!=0 or nk<2:
        raise ValueError("solver='qtt' needs nk to be a power of two "
            "(the k index is binary encoded, one tensor-train site per "
            "bit); got nk = %d, the nearest powers of two are %d and %d"
            %(nk,2**int(np.floor(np.log2(max(nk,2)))),
                 2**int(np.ceil(np.log2(max(nk,2))))))
    if neig!=1:
        raise ValueError("solver='qtt' returns the lowest exciton only "
            "(neig=1). Excited excitons would come from pyitensor's "
            "overlap-penalty dmrg_excited, and it does not converge on "
            "this problem: measured on the gapped chain at nk=32 against "
            "the dense spectrum 1.813914/1.831623/1.917174, it returned "
            "1.787139/1.829526/2.019560 at a penalty weight well above "
            "the bandwidth, and 1.659272/1.666385/1.797612 -- below the "
            "true second eigenvalue, so not even variational -- at the "
            "default weight. dmrg_excited's own docstring records the "
            "same class of stationary point on an unrelated model. Use "
            "solver='iterative' for several excitons; it is exact and "
            "still needs no dense matrix")
    pb = PairOracle(h,Q=Q,nk=nk,nv=nv,nc=nc,gauge=gauge,coarse_nk=coarse_nk)
    Wx = bare_interaction(h,V=V) # bare, for the exchange term
    W = Wx # the direct term takes the same interaction; see solve.py
    lk = LazyKernel(pb,W,Wx=Wx,kernel=kernel)
    grid = quantics_grid(pb,unfolding=unfolding)
    dcores = _diagonal_cores(lk,grid,tolerance=tolerance)
    kcores = _kernel_cores(lk,grid,tolerance=tolerance,
            maxbonddim=maxbonddim)
    mpo = _mpo_from_cores(grid,[dcores,kcores])
    es,ws = run_dmrg(mpo,grid,neig=neig,nsweep=nsweep,maxdim=maxdim,
            cutoff=cutoff,weight=weight,seed=seed)
    # The lowest independent-particle transition, which binding energies
    # are measured from. A minimum over the mesh would be the one O(nk)
    # step in an otherwise logarithmic solver, so it is found by a binary
    # descent seeded from the k-points already diagonalized -- see
    # PairOracle.lowest_transition_energy, including why running DMRG on
    # the diagonal MPO (the elegant construction) does not work.
    pb.lowest_transition = pb.lowest_transition_energy()
    return pb,W,Wx,(np.array(es,dtype=np.complex128),ws)


class LazyKernel():
    """The interaction part of the resonant BSE block, X - D, evaluated at
    an arbitrary (m,n) without the matrix and without the mesh.

    This is build_blocks' resonant block minus its diagonal, written for
    one element at a time:

      X[m,n] = (1/N) sum_ab F[m,a] Wx_ab(Q) conj(F[n,b])
      D[m,n] = (1/N) sum_ab conj(el[m,a]) el[n,a] W_ab(k_m-k_n)
                            conj(ho[n,b]) ho[m,b]

    with F[m,a] = conj(el[m,a]) ho[m,a]. Everything it needs comes from
    PairOracle, which diagonalizes only the k-points actually asked for."""
    def __init__(self,pb,W,Wx=None,kernel="full"):
        if kernel not in ("full","direct","exchange","none"):
            raise ValueError("kernel must be one of 'full', 'direct', "
                    "'exchange', 'none', got %r"%(kernel,))
        self.pairs = pb
        self.kernel = kernel
        self.W = W
        self.Wx = W if Wx is None else Wx
        self.norm = 1.0/pb.nkpoints
        self.geometry = pb.geometry
        self.WQ = interaction_at_q(self.Wx,self.geometry,pb.Q)
        self._wq = dict() # W(k_m-k_n), memoized by the k-index difference
        self._pair = dict() # (F,el,ho) per pair index, memoized
    def _pair_data(self,m):
        """(F,el,ho) of one pair, memoized.

        Cross interpolation revisits the same pair index many times over
        -- once per column it pairs it with, and again on every pivot
        search that passes through it -- so this cache, not the k-point
        cache underneath it, is what keeps the oracle cheap. F is the
        density form factor conj(el)*ho, precomputed because the exchange
        term needs it on every single element."""
        out = self._pair.get(m)
        if out is None:
            dE,el,ho = self.pairs.pair_arrays([m])
            el,ho = el[0],ho[0]
            out = (np.conj(el)*ho,el,ho)
            self._pair[m] = out
        return out
    def _direct_W(self,ikm,ikn):
        """W at the difference of two mesh points, memoized.

        The key is the per-direction index difference modulo nk, not the
        pair of k-indices: W(q) is periodic, so the mesh has only nk^dim
        distinct differences rather than nk^(2 dim) --
        kernel.qdifference_map makes the same observation for the dense
        build. Cross interpolation visits few enough points that this
        dictionary stays far smaller than that anyway."""
        nk = self.pairs.nk
        dim = max(self.pairs.dimensionality,1)
        a,b = int(ikm),int(ikn)
        key = []
        for _ in range(dim):
            a,ra = divmod(a,nk) ; b,rb = divmod(b,nk)
            key.append((ra-rb)%nk)
        key = tuple(key)
        out = self._wq.get(key)
        if out is None:
            q = self.pairs.kpoint(ikm)-self.pairs.kpoint(ikn)
            out = interaction_at_q(self.W,self.geometry,q)
            self._wq[key] = out
        return out
    def element(self,m,n):
        """One matrix element of X - D"""
        if self.kernel=="none": return 0.0+0.0j
        Fm,elm,hom = self._pair_data(int(m))
        Fn,eln,hon = self._pair_data(int(n))
        out = 0.0+0.0j
        if self.kernel in ("full","exchange"):
            out = out + self.norm*(Fm@self.WQ@np.conj(Fn))
        if self.kernel in ("full","direct"):
            ikm = int(m)//self.pairs.nband
            ikn = int(n)//self.pairs.nband
            Wq = self._direct_W(ikm,ikn)
            # sum_ab conj(el_m,a) el_n,a W_ab conj(ho_n,b) ho_m,b
            out = out - self.norm*((np.conj(elm)*eln)@Wq@
                                   (np.conj(hon)*hom))
        return out
    def diagonal_energy(self,m):
        """The independent-particle transition energy of pair m"""
        return float(self.pairs.pair_arrays([m])[0][0])


def quantics_grid(pb,unfolding="grouped"):
    """The quantics grid of the pair index.

    One variable per k-direction with `nbit` binary sites, and then the
    valence and conduction indices as variables of their own -- each
    FACTORIZED across as many sites as its prime factorization allows,
    rather than sitting on one site of dimension nv*nc.

    That factorization is not cosmetic. A tensor-train site of local
    dimension d becomes a site of d^2 in the MPO (row and column indices
    are fused there), and cross interpolation has to sample that many
    states at that site for every pivot. Keeping all band pairs on one
    site is fine for the small windows a primitive cell gives -- nv*nc = 1
    or 4, fused to 1 or 16 -- and becomes the binding constraint the
    moment the cell grows: a 3x3 supercell of the honeycomb has 9 valence
    and 9 conduction bands, so a single band site would be 81 wide and
    6561 wide in the MPO. Split into separate v and c variables and each
    factorized as 9 = 3*3, the same window is four sites of dimension 3,
    i.e. 9 in the MPO. The problem is unchanged; only its encoding is.

    Any nv is handled, since the prime factorization of any integer
    exists: the largest site is the largest prime factor (a window of 7
    bands is one site of 7, of 8 is three sites of 2). No padding is
    involved, so cross interpolation never samples an index that does not
    correspond to a real band pair.

    Variables are declared k-directions first, then valence, then
    conduction, and grid_to_pair decodes that back to the flat pair index
    PairBasis and PairOracle use. tests/bse/test_bse_qtt.py checks the
    correspondence against PairBasis directly rather than trusting it."""
    from ..qutecipytk.quantics.grid import InherentDiscreteGrid
    dim = max(pb.dimensionality,1)
    nbit = int(round(np.log2(pb.nk)))
    names = ["k%d"%j for j in range(dim)]
    Rs = [nbit]*dim
    base = [2]*dim
    nv,nc = len(pb.vbands),len(pb.cbands)
    vf,cf = prime_factors(nv),prime_factors(nc)
    for i,(b,r) in enumerate(vf):
        names.append("v%d"%i) ; Rs.append(r) ; base.append(b)
    for i,(b,r) in enumerate(cf):
        names.append("c%d"%i) ; Rs.append(r) ; base.append(b)
    grid = InherentDiscreteGrid.from_resolutions(Rs,base=base,
            unfoldingscheme=unfolding,variablenames=names)
    grid.pyqula_nk = pb.nk
    grid.pyqula_dim = dim
    grid.pyqula_nband = pb.nband
    grid.pyqula_ncb = nc
    # the size each v/c variable spans, for the mixed-radix decode
    grid.pyqula_vsizes = [b**r for b,r in vf]
    grid.pyqula_csizes = [b**r for b,r in cf]
    return grid


def prime_factors(n):
    """[(base,multiplicity)] of n, so that prod(base**mult) == n.

    n=1 gives [], i.e. no site at all -- a one-band window needs no index.
    """
    out,m,p = [],int(n),2
    while p*p<=m:
        r = 0
        while m%p==0: m //= p ; r += 1
        if r: out.append((p,r))
        p += 1
    if m>1: out.append((m,1))
    return out


def grid_to_pair(grid,gi):
    """Flat pair index of a grid index (ix,...,v digits...,c digits...).

    The pair index PairOracle.label decodes is
    m = ik*nband + iv*ncb + ic, so this is the same mixed-radix
    composition read the other way."""
    ik = 0
    for j in range(grid.pyqula_dim): ik = ik*grid.pyqula_nk + int(gi[j])
    pos = grid.pyqula_dim
    iv = 0
    for s in grid.pyqula_vsizes: iv = iv*s + int(gi[pos]) ; pos += 1
    ic = 0
    for s in grid.pyqula_csizes: ic = ic*s + int(gi[pos]) ; pos += 1
    return (ik*(grid.pyqula_nband//grid.pyqula_ncb) + iv)*grid.pyqula_ncb + ic


def build_mpo(lk,grid,tolerance=1e-6,maxbonddim=None):
    """Cross-interpolate the kernel and the diagonal, and return their sum
    as a pyitensor MPO."""
    dcores = _diagonal_cores(lk,grid,tolerance=tolerance)
    kcores = _kernel_cores(lk,grid,tolerance=tolerance,maxbonddim=maxbonddim)
    return _mpo_from_cores(grid,[dcores,kcores])


def _diagonal_cores(lk,grid,tolerance=1e-6):
    """MPO cores of diag(dE), from a cross interpolation of dE(m).

    dE is a smooth, gauge-independent function of k, so its tensor train
    is small (rank 8-16 on the models measured here) and it costs one
    interpolation over the ROW index only, not the fused one."""
    from ..qutecipytk import crossinterpolate2
    from ..qutecipytk.tensortrain.core import tensortrain
    from ..qutecipytk.tensortrain.cachedfunction import CachedFunction
    localdims = list(grid.localdimensions())
    def f(idx):
        gi = grid.quantics_to_grididx(list(idx))
        return complex(lk.diagonal_energy(grid_to_pair(grid,gi)))
    fc = CachedFunction(np.complex128,f,localdims)
    tci,ranks,errors = crossinterpolate2(np.complex128,fc,localdims,
            tolerance=tolerance)
    tt = tensortrain(tci)
    out = []
    for T in tt.sitetensors():
        # a diagonal operator: W[al,s,s',be] = T[al,s,be] delta_{s,s'}
        al,d,be = T.shape
        core = np.zeros((al,d,d,be),dtype=np.complex128)
        for s in range(d): core[:,s,s,:] = T[:,s,:]
        out.append(core)
    return out


def _kernel_cores(lk,grid,tolerance=1e-6,maxbonddim=None):
    """MPO cores of the interaction part X - D, from a cross
    interpolation over the FUSED (row,column) index.

    Site i carries the pair (row bit, column bit) as a single index
    s = s_row*d + s_col, which is what turns a matrix into a tensor train
    with the same site structure as the vectors it acts on."""
    from ..qutecipytk import crossinterpolate2
    from ..qutecipytk.tensortrain.core import tensortrain, TensorTrain
    from ..qutecipytk.tensortrain.cachedfunction import CachedFunction
    dims = list(grid.localdimensions())
    fused = [d*d for d in dims]
    if lk.kernel=="none": # nothing to interpolate: the zero operator
        return [np.zeros((1,d,d,1),dtype=np.complex128) for d in dims]
    def f(idx):
        srow = [int(s)//d for s,d in zip(idx,dims)]
        scol = [int(s)%d for s,d in zip(idx,dims)]
        m = grid_to_pair(grid,grid.quantics_to_grididx(srow))
        n = grid_to_pair(grid,grid.quantics_to_grididx(scol))
        return complex(lk.element(m,n))
    fc = CachedFunction(np.complex128,f,fused)
    # The default first pivot -- the all-zero index -- decodes to
    # srow = scol = 0, i.e. the element (0,0), which is ON THE DIAGONAL
    # where the kernel is largest. That is what is wanted here (TCI's
    # first pivot sets the scale every later error is measured against)
    # and is why no initialpivots argument appears; it is worth knowing
    # rather than rediscovering if the index encoding is ever changed.
    tci,ranks,errors = crossinterpolate2(np.complex128,fc,fused,
            tolerance=tolerance,maxbonddim=maxbonddim)
    tt = TensorTrain.reshaped(tensortrain(tci),[(d,d) for d in dims],
            dtype=np.complex128)
    # (row,column) -> the engine's (in,out) = (column,row) storage order
    return [np.swapaxes(T,1,2) for T in tt.sitetensors()]


def _mpo_from_cores(grid,corelists):
    """Assemble pyitensor MPOs from lists of dense cores and add them.

    The sum is EXACT: sum_many concatenates the link spaces and then makes
    one sweep at cutoff zero, so nothing is truncated. That matters here
    more than usual -- the diagonal is O(gap) and the kernel is O(1/N), so
    a relative-tolerance truncation of their sum would throw the
    interaction away entirely and return the independent-particle
    spectrum."""
    from dmrgpy.pyitensor.index import Index
    from dmrgpy.pyitensor.tensor import ITensor
    from dmrgpy.pyitensor.mpscontainer import MPO
    from dmrgpy.pyitensor.mpsalgebra import sum_many
    dims = list(grid.localdimensions())
    n = len(dims)
    sites = [Index(d,tags="Site,n=%d"%(i+1)) for i,d in enumerate(dims)]
    grid.pyqula_sites = sites
    mpos = []
    for cores in corelists:
        tensors = []
        links = [Index(cores[i].shape[3],tags="Link,l=%d"%(i+1))
                 for i in range(n-1)]
        for i in range(n):
            s = sites[i]
            arr = cores[i]
            inds,shape = [],[]
            if i>0: inds.append(links[i-1]) ; shape.append(arr.shape[0])
            inds += [s,s.prime(1)] ; shape += [arr.shape[1],arr.shape[2]]
            if i<n-1: inds.append(links[i]) ; shape.append(arr.shape[3])
            # the boundary cores carry a trivial dimension-one link that
            # this engine does not represent, unlike ITensor itself
            a = arr
            if i==0: a = a[0]
            if i==n-1: a = a[...,0]
            tensors.append(ITensor(tuple(inds),
                np.array(a,dtype=np.complex128).reshape(tuple(shape))))
        m = MPO(tensors)
        m.center = 1
        mpos.append(m)
    if len(mpos)==1: return mpos[0]
    return sum_many(mpos,cutoff=0.0)


class _Sites():
    """The minimal SiteSet interface pyitensor's randomMPS needs: a
    length, a physical Index per site and its dimension. The stock
    sites.SiteX only knows spin/fermion/boson type codes, and the sites
    here are quantics bits and a band index, neither of which is one."""
    def __init__(self,indices):
        self._i = list(indices)
    def length(self): return len(self._i)
    N = length
    def si(self,i): return self._i[i-1]
    def dim(self,i): return self._i[i-1].dim


def run_dmrg(mpo,grid,neig=1,nsweep=24,maxdim=100,cutoff=1e-10,weight=None,
        seed=0):
    """Ground (and optionally excited) states of the MPO, by DMRG.

    Returns (energies,amplitudes) with amplitudes[i] the FULL exciton
    amplitude vector of state i, reconstructed from its MPS. That
    reconstruction is O(npair) and so is the one step of this solver that
    is not logarithmic in the mesh: it is done because every other solver
    returns amplitudes and the downstream code expects them, and it is
    skipped automatically once it would not fit.

    neig>1 uses pyitensor's overlap-penalty dmrg_excited and is NOT
    reachable from solve_qtt, which refuses it -- on this problem it
    converges to stationary points up to 0.4 off, sometimes below the true
    eigenvalue, at any penalty weight tried. The branch is kept because
    the failure is a property of the penalized objective rather than of
    this code, and a better excited-state driver (a proper block DMRG, or
    a shift-invert on the MPO) would slot straight in here. Until then,
    several excitons come from bsetk/iterative.py."""
    from dmrgpy.pyitensor.mpsalgebra import randomMPS, inner
    from dmrgpy.pyitensor.dmrg import dmrg, dmrg_excited
    from dmrgpy.pyitensor.sweeps import Sweeps
    sites = _Sites(grid.pyqula_sites)
    sweeps = Sweeps(nsweep)
    sweeps.maxdim = maxdim
    sweeps.cutoff = cutoff
    sweeps.niter = 4
    np.random.seed(seed)
    states,energies = [],[]
    for i in range(neig):
        psi = randomMPS(sites,min(maxdim,8))
        if i==0:
            dmrg(psi,mpo,sweeps)
        else:
            w = weight
            if w is None:
                # the penalty must sit well above the spectral range being
                # searched, per dmrg_excited's own docstring; the first
                # energy is the natural scale here
                w = 10.*max(1.,abs(energies[0]))
            dmrg_excited(psi,mpo,states,w,sweeps)
        # the Rayleigh quotient, not <psi|H|psi>: dmrg_excited does not
        # leave psi normalized, and an unnormalized state reports an
        # energy scaled by <psi|psi> -- which comes out BELOW the true
        # excited energy and so does not even look wrong. Measured on the
        # gapped chain at nk=16, the first three excited excitons came
        # back 1.8206/1.8334/1.9510 against 1.8712/1.9028/2.2253 before
        # this division
        nrm = float(np.real(inner(psi,psi)))
        energies.append(float(np.real(inner(psi,mpo,psi)))/nrm)
        psi.normalize()
        states.append(psi)
    ws = _amplitudes(states,grid)
    order = np.argsort(energies)
    return np.array(energies)[order],ws[order]


def _amplitudes(states,grid,max_npair=4_000_000):
    """Reconstruct the full amplitude vector of each MPS, in PAIR-INDEX
    order.

    Contracting the MPS gives a vector indexed in SITE order, which is not
    the pair index unless every variable's digits happen to sit
    contiguously and in order -- true for unfolding="grouped", false for
    "interleaved", where the k-directions' digits are interleaved by
    scale. Since each site carries exactly one digit of one variable, the
    fix is a pure axis permutation of the reshaped tensor: sort the sites
    by (variable, digit significance) and a C-order reshape is then the
    mixed-radix pair index by construction. No index arithmetic per
    element, and it is exact for every unfolding.

    O(npair) in memory, which is the one non-logarithmic step of this
    solver. It is done because every other solver returns amplitudes and
    the downstream code expects them; above max_npair it returns an empty
    array instead, so a calculation whose whole point was to avoid an
    npair-sized allocation does not end by making one."""
    dims = list(grid.localdimensions())
    npair = int(np.prod(dims))
    if npair>max_npair:
        return np.zeros((len(states),0),dtype=np.complex128)
    perm = _site_to_variable_order(grid)
    out = []
    for psi in states:
        v = np.ones((1,1),dtype=np.complex128) # (accumulated, left bond)
        for i in range(1,psi.length()+1):
            a = _mps_core(psi,i) # (left,phys,right)
            dl,d,dr = a.shape
            v = (v@a.reshape(dl,d*dr)).reshape(-1,dr)
        t = v.reshape(dims)
        out.append(np.transpose(t,perm).reshape(-1))
    return np.array(out,dtype=np.complex128)


def _site_to_variable_order(grid):
    """Axis permutation taking the site order to (variable, digit) order.

    grid.indextable[i] is the list of (variablename, digitnumber) a site
    carries -- exactly one pair for the unfolding schemes used here -- and
    digit 0 is the most significant, so sorting by (variable position,
    digit) reproduces the mixed-radix layout grid_to_pair assumes."""
    # InherentDiscreteGrid keeps these as plain attributes; the
    # grid_variablenames()/grid_indextable() accessors belong to
    # DiscretizedGrid, which is a different class
    names = list(grid.variablenames)
    key = []
    for i,site in enumerate(grid.indextable):
        if len(site)!=1:
            raise ValueError("this MPS reconstruction needs one variable "
                "digit per tensor-train site; the 'fused' unfolding puts "
                "several on one site and is not supported here")
        name,digit = site[0]
        key.append((names.index(name),digit,i))
    return [i for _,_,i in sorted(key)]


def _mps_core(psi,i):
    """Site tensor i of an MPS as a plain (left,physical,right) array.

    pyitensor's boundary sites carry no trivial dimension-one link (unlike
    ITensor itself), so those are put back here."""
    T = psi.A(i)
    phys = [ind for ind in T.inds if ind.hastags("Site")][0]
    left = _link(psi,i,i-1)
    right = _link(psi,i,i+1)
    order = [ind for ind in (left,phys,right) if ind is not None]
    a = T.transpose_to(order)
    if left is None: a = a[None,...]
    if right is None: a = a[...,None]
    return np.array(a,dtype=np.complex128)


def _link(chain,i,j):
    from dmrgpy.pyitensor.tensor import commonIndex
    if j<1 or j>chain.length(): return None
    return commonIndex(chain.A(i),chain.A(j))
