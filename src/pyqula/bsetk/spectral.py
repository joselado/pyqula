"""Exciton spectral functions at finite center-of-mass momentum Q, scanned
along a q-path, from the Tamm-Dancoff BSE.

exciton_bands (bands.py) diagonalizes one dense BSE per q-point, which
limits the pair basis to a few thousand pairs. The quantity a q-path map
needs is not every eigenvector but the spectral function of the block,

    A(Q,w) = -(1/pi N) Im Tr[O (w + i delta - A_Q)^-1] ,

with A_Q the resonant block at Q and N the number of k-points of the mesh,
so it counts excitons per unit cell and energy, and integrates to nv*nc
without an operator. That is a resolvent, and the Chebyshev expansion of
chebyshev.py gives it from matrix-vector products alone, with the block
applied as a convolution on the mesh (convolution.py), which is Q-agnostic.
The trace is stochastic, over nrand random-phase vectors, and exact over
the unit vectors once nrand reaches the number of pairs, or with the
dense route.

O is an electronic operator (anything h.get_operator takes: a name such as
"sublatticeA", a matrix, an Operator) acting on the electron or on the
hole of the pair, restricted to the band window,

    electron:  (O x)[k,v,c] = sum_c' <c,k+Q|O|c',k+Q> x[k,v,c'] ,
    hole:      (O x)[k,v,c] = sum_v' <v',k|O|v,k> x[k,v',c] ,

which is the change of <O> that the exciton carries on each particle, the
hole one with the sign of a missing electron removed, so that a projector
on a set of atoms gives the probability that the electron (or the hole)
sits there. The projectors on every atom add up to the unprojected result.

exciton_structure_factor replaces the trace by the density fluctuation at
momentum Q, D[k,v,c] = sum_a exp(2 pi i Q.f_a) <c,k+Q|a><a|v,k>, with f_a
the fractional position of orbital a, which is the excitonic part of the
dynamical structure factor S(Q,w) that momentum-resolved electron energy
loss and inelastic x-ray scattering measure. The phase on the orbital
position is what makes it a property of the crystal rather than of the
unit cell chosen, which tests/bse/test_bse_spectral.py checks on a
supercell. Dark excitons do not appear in it.

method="dense" computes the same two quantities from the eigenvectors of
the dense BSE, as the reference of the Chebyshev route.
"""
import numpy as np


# the BSE options the spectral function forwards
_keys = ("V","nv","nc","kernel","screening","nkW","channel")


def get_particle_names():
    """Every value particle= accepts"""
    return sorted(_particles)


def get_method_names():
    """Every value method= accepts"""
    return sorted(_methods)


def _check_name(value,names,what):
    if value not in names: raise ValueError("%s must be one of %s, got %r"%(
        what,", ".join(repr(n) for n in sorted(names)),value))


def _band_matrix(op,C,ks):
    """<n,k|O|m,k> for the eigenvectors C (nk,nb,norb), rows the states,
    returning (nk,nb,nb)"""
    nk,nb,norb = C.shape
    out = np.zeros((nk,nb,nb),dtype=np.complex128)
    for ik in range(nk): # loop over the mesh
        OC = np.array([op(C[ik,m],k=ks[ik]) for m in range(nb)]) # O|m>
        out[ik] = np.conj(C[ik])@OC.T
    return out


class _PairOperator():
    """An electronic operator acting on the electron (at k+Q) or on the
    hole (at k) of the pairs of a pair basis, inside its band window"""
    def __init__(self,h,pb,operator,particle):
        op = h.get_operator(operator)
        nk = len(pb.kpoints)
        self.shape = (nk,len(pb.vbands),len(pb.cbands))
        self.particle = particle
        if particle=="electron": # <c,k+Q|O|c',k+Q>
            C = pb.ckq[:,pb.cbands,:]
            self.M = _band_matrix(op,C,pb.kpoints+pb.Q[None,:])
        else: # <v',k|O|v,k>, the transpose of the band matrix
            C = pb.ck[:,pb.vbands,:]
            self.M = np.swapaxes(_band_matrix(op,C,pb.kpoints),1,2)
    def apply(self,X):
        """O X for X of shape (npair,ncol)"""
        nk,nv,nc = self.shape
        Xk = X.reshape(nk,nv,nc,-1)
        if self.particle=="electron":
            out = np.einsum("kcd,kvdn->kvcn",self.M,Xk,optimize=True)
        else:
            out = np.einsum("kuv,kvcn->kucn",self.M,Xk,optimize=True)
        return out.reshape(X.shape)


def _fractional_positions(h,norb):
    """Fractional coordinates of every orbital, (norb,3), the components
    beyond the dimensionality set to zero"""
    from ..geometrytk.fractional import get_fractional_function
    g = h.geometry
    f = get_fractional_function(g)
    fr = np.array([f(r) for r in g.r],dtype=np.float64)
    fr[:,g.dimensionality:] = 0.
    m = norb//len(g.r) # orbitals per site, spin and the like
    return np.repeat(fr,m,axis=0)


def density_vertex(h,pb):
    """The density fluctuation at momentum pb.Q in the pair basis, (npair,1),
    D[k,v,c] = sum_a exp(2 pi i Q.f_a) conj(C_c(k+Q)_a) C_v(k)_a"""
    fr = _fractional_positions(h,pb.el.shape[1])
    ph = np.exp(2j*np.pi*(fr@pb.Q)) # (norb,)
    return (np.conj(pb.el)*pb.ho)@ph[:,None]


def _random_vectors(npair,nrand,seed,chunk=64):
    """Yield the vectors of the trace in blocks of columns: nrand
    random-phase vectors, or the unit vectors once nrand reaches npair,
    where the stochastic trace has nothing left to estimate. The unit
    vectors come chunk at a time, so that the recursion never holds an
    npair x npair block"""
    if nrand>=npair:
        for i in range(0,npair,chunk):
            R = np.zeros((npair,min(chunk,npair-i)),dtype=np.complex128)
            R[np.arange(i,i+R.shape[1]),np.arange(R.shape[1])] = 1.
            yield R
        return
    rng = np.random.default_rng(seed)
    yield np.exp(2j*np.pi*rng.random((npair,nrand)))


def _lorentzian(es,E,w,delta):
    """sum_n w_n delta/pi/((e - E_n)^2 + delta^2), at every e of es"""
    den = (es[:,None]-E[None,:])**2 + delta**2
    return (delta/np.pi/den)@w


class _Dense():
    """The spectral function from the eigenvectors of the dense
    Tamm-Dancoff BSE at Q, whose trace is always exact"""
    exact = True
    def __init__(self,h,q,nk,W,kw):
        from .solve import BSE
        b = BSE(h,Q=q,nk=nk,tda=True,solver="dense",screening=W,**kw)
        self.pairs = b.pairs
        self.E = np.real(b.energies)
        self.U = np.array(b.amplitudes.T) # (npair,nexc)
    def bounds(self):
        return np.min(self.E),np.max(self.E)
    def trace(self,P,R,es,delta):
        """sum_i <P_i|A(w)|R_i>, A(w) = -Im G(w)/pi, with the weights of
        every exciton taken from its eigenvector"""
        w = np.real(np.sum(np.conj(np.conj(self.U.T)@P)*(np.conj(self.U.T)@R),
            axis=1))
        return _lorentzian(es,self.E,w,delta)
    def braket(self,D,es,delta):
        """<D|A(w)|D>"""
        w = np.abs(np.conj(self.U.T)@D[:,0])**2
        return _lorentzian(es,self.E,w,delta)


class _Chebyshev():
    """The spectral function from the Chebyshev resolvent of the block
    applied as a convolution on the mesh, never formed"""
    exact = False
    def __init__(self,h,q,nk,W,kw):
        from .pairbasis import PairBasis
        from .interaction import bare_interaction
        from .convolution import ConvolutionKernel
        from . import chebyshev
        self.pairs = PairBasis(h,Q=q,nk=nk,nv=kw.get("nv"),nc=kw.get("nc"))
        Wx = bare_interaction(h,V=kw.get("V"))
        if W is None: W = Wx
        self.f = ConvolutionKernel(self.pairs,W,Wx=Wx,
                kernel=kw.get("kernel","full"))
        self.cheb = chebyshev
        self.c,self.s = chebyshev.spectral_bounds(self.f)
    def bounds(self):
        return self.c-self.s,self.c+self.s
    def trace(self,P,R,es,delta):
        """sum_i <P_i|A(w)|R_i>, from one Chebyshev recursion on R, the
        moments <P_i|T_n|R_i> taken on the vectors it already makes"""
        zs = es + 1j*delta
        if self.f.nterm==0: # kernel="none", the block is diagonal
            G = 1./(zs[None,:]-self.f.diag[:,None]) # (npair,nz)
            t = np.einsum("pi,pi,pz->z",np.conj(P),R,G,optimize=True)
            return -np.imag(t)/np.pi
        cheb = self.cheb
        r,qs = cheb._coefficients(zs,self.c,self.s)
        N = cheb.number_of_moments(zs,self.c,self.s)
        t = np.zeros(len(zs),dtype=np.complex128)
        R = np.ascontiguousarray(R,dtype=np.complex128)
        for (n0,V) in cheb._recursion(self.f.matvec,R,self.c,self.s,N):
            mus = np.einsum("pi,npi->n",np.conj(P),V,optimize=True)
            ns = np.arange(n0,n0+len(V))
            g = 2.*qs[None,:]**ns[:,None]/r[None,:] # (nb,nz)
            if n0==0: g[0] = g[0]/2.
            t += mus@g
        return -np.imag(t/self.s)/np.pi
    def braket(self,D,es,delta):
        """<D|A(w)|D>"""
        zs = es + 1j*delta
        if self.f.nterm==0:
            G = 1./(zs[None,:]-self.f.diag[:,None])
            return -np.imag(np.abs(D[:,0])**2@G)/np.pi
        out = self.cheb.resolvent_braket(self.f.matvec,D,zs,self.c,self.s)
        return -np.imag(out[:,0,0])/np.pi


_particles = ("electron","hole")
# how the resolvent is evaluated, method=
_methods = {"kpm":_Chebyshev,"dense":_Dense}


def exciton_kdos_bands(h,operator=None,particle="electron",nrand=20,seed=0,
        **kwargs):
    """Return the exciton density of states A(Q,w) along a q-path, as
    three flat arrays (qs,es,ds): the index of the q-point along the path,
    the energy, and the number of excitons per unit cell and energy, in
    the convention of get_kdos_bands, so that the bright lines of the map
    are the exciton bands.

    operator projects it, acting on the electron or on the hole of every
    pair (particle=), so that a projector on a set of atoms gives the
    weight of the exciton with its electron (or its hole) there. The
    trace runs over nrand random-phase vectors drawn with seed, and is
    exact once nrand reaches the number of pairs, at the cost of one
    recursion per pair, so that is for small pair bases, where
    method="dense" does the same. Every other keyword is the one of
    exciton_structure_factor"""
    _check_name(particle,_particles,"particle")
    if int(nrand)<1: raise ValueError("nrand must be at least 1, got %r"
            %(nrand,))
    def probe(b,es,delta):
        pb = b.pairs
        n = pb.npair if b.exact else int(nrand) # every eigenvector is
        # there already on the dense route
        O = None if operator is None else _PairOperator(h,pb,operator,
                particle)
        t = 0.
        for R in _random_vectors(pb.npair,n,seed): # blocks of columns
            P = R if O is None else O.apply(R)
            t = t + b.trace(P,R,es,delta)
        # summed over the unit vectors this is the trace, and for random
        # vectors with entries of unit modulus the mean of <r|O G|r> is
        # Tr[O G], so only those are divided by their number
        ncol = 1 if n>=pb.npair else n
        return t/ncol/len(pb.kpoints)
    return _qpath_map(h,probe,"exciton_kdos_bands",**kwargs)


def exciton_structure_factor(h,**kwargs):
    """Return the excitonic dynamical structure factor S(Q,w) along a
    q-path, as three flat arrays (qs,es,ss), per unit cell, in the
    convention of get_kdos_bands.

    qpath/nq select the path as in get_exciton_bands. energies is the
    energy grid, by default 200 points over the spectrum of the BSE
    along the path, which builds the BSE at every q-point twice, once for
    the window and once for the result. delta is the Lorentzian broadening, and the number of
    Chebyshev moments grows as the spectral width over delta, about
    20 s/delta with s the half width. method="kpm" is the Chebyshev
    resolvent, method="dense" the eigenvectors of the dense BSE, both
    Tamm-Dancoff. Every other keyword (V, nv, nc, kernel, screening, nkW,
    channel) is passed on to the BSE"""
    def probe(b,es,delta):
        pb = b.pairs
        return b.braket(density_vertex(h,pb),es,delta)/len(pb.kpoints)
    return _qpath_map(h,probe,"exciton_structure_factor",**kwargs)


def _qpath_map(h,probe,name,qpath=None,nq=None,energies=None,delta=0.05,
        nk=10,method="kpm",**kwargs):
    """probe(b,es,delta) of the BSE b at every q-point of the path, as the
    three flat columns (qs,es,ds)"""
    from ..klist import get_qpath
    from .solve import get_direct_interaction
    from .pairbasis import PairBasis
    _check_name(method,_methods,"method")
    bad = sorted(k for k in kwargs if k not in _keys)
    if bad: raise TypeError("%s got unexpected keyword argument(s) %s; "
            "besides its own it accepts qpath, nq, energies, delta, nk, "
            "method, %s"%(name,", ".join(bad),", ".join(_keys)))
    if h.dimensionality<1: raise ValueError("%s needs a periodic "
            "Hamiltonian, a 0d system has no center-of-mass momentum"
            %(name,))
    if not delta>0.: raise ValueError("the broadening delta must be "
            "positive, got %r"%(delta,))
    qpath = get_qpath(h.geometry,qpath,nq) # generate the q-path
    kw = dict(kwargs)
    screening,nkW = kw.pop("screening",None),kw.pop("nkW",None)
    W = None
    if screening is not None:
        # the screened interaction does not depend on Q, built once on
        # the mesh of the first q-point and reused along the path
        pb = PairBasis(h,Q=qpath[0],nk=nk,nv=kw.get("nv"),nc=kw.get("nc"))
        W = get_direct_interaction(h,pb,V=kw.get("V"),nk=nk,
                screening=screening,nkW=nkW,kernel=kw.get("kernel","full"),
                channel=kw.get("channel","charge"))
    if method=="kpm": kw.pop("channel",None) # only the screening uses it
    solver = _methods[method]
    if energies is None: # the window of the spectrum along the path
        bs = np.array([solver(h,q,nk,W,kw).bounds() for q in qpath])
        energies = np.linspace(np.min(bs[:,0])-5*delta,
                np.max(bs[:,1])+5*delta,200)
    es = np.asarray(energies,dtype=np.float64)
    # one BSE per q-point, serially, since the matvec is the parallel part
    ds = [probe(solver(h,q,nk,W,kw),es,delta) for q in qpath]
    qs = np.repeat(np.arange(len(qpath)),len(es))
    return qs,np.tile(es,len(qpath)),np.concatenate(ds)
