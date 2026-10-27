# Excitonic linear and second-harmonic optical response of a periodic
# tight-binding insulator, from the Tamm-Dancoff Bethe-Salpeter equation.
# The physics, the units and the conventions are documented in the public
# entry points of conductivity.py (excitonic_optical_conductivity and
# excitonic_shg_conductivity); this module holds the machinery.
import numpy as np
import numba

from . import kubo
from . import shg
from .. import parallel


class _DenseResolvent():
    """Tamm-Dancoff BSE at Q=0 on the nk mesh, dense, so that every
    exciton is available: energies E[n] and amplitudes U[p,n] over the
    pairs p = (k,v,c), the exciton n being sum_p U[p,n] c+_ck c_vk |0>.
    The resolvent (z - A)^-1 = U diag(1/(z - E)) U^+"""
    def __init__(self,h,nk,bse_kwargs):
        from ..bsetk.solve import BSE
        b = BSE(h,nk=nk,tda=True,solver="dense",**bse_kwargs)
        self.pairs = b.pairs
        self.E,self.U = np.real(b.energies),np.array(b.amplitudes.T)
    def apply(self,D,zs):
        """(z - A)^-1 D, D of shape (npair,ncol). Returns (npair,ncol,nz)"""
        Ud = np.conj(self.U.T)@D # (nexc,ncol)
        den = 1./(zs[None,:]-self.E[:,None]) # (nexc,nz)
        return np.array([self.U@(Ud[:,a,None]*den)
            for a in range(D.shape[1])]).transpose(1,0,2)
    def braket(self,D,zs):
        """D_a^+ (z - A)^-1 D_b. Returns (nz,ncol,ncol)"""
        Ud = np.conj(self.U.T)@D
        den = 1./(zs[None,:]-self.E[:,None])
        return np.einsum("na,nb,nz->zab",np.conj(Ud),Ud,den,optimize=True)


class _ChebyshevResolvent():
    """The same resolvent through a Chebyshev expansion of the matrix-free
    BSE block (bsetk/chebyshev.py), never forming nor diagonalizing it, so
    the cost is linear in the number of pairs. The block is applied as a
    convolution on the mesh (bsetk/convolution.py), and with the GPU
    switched on (pyqula/gpu.py) the recursion runs on the device
    (bsetk/chebyshevjax.py), in the precision kpm_prec"""
    _keys = ("V","nv","nc","kernel","screening","nkW","channel","gauge",
            "kpm_prec")
    def __init__(self,h,nk,bse_kwargs):
        from ..bsetk.pairbasis import PairBasis
        from ..bsetk.interaction import bare_interaction
        from ..bsetk.solve import get_direct_interaction
        from ..bsetk.convolution import ConvolutionKernel
        from ..bsetk import chebyshev
        bad = sorted(k for k in bse_kwargs if k not in self._keys)
        if bad: raise TypeError("method='kpm' got unexpected keyword "
                "argument(s) %s; it accepts %s"%(", ".join(bad),
                    ", ".join(self._keys)))
        kw = dict(bse_kwargs)
        V,kernel = kw.get("V"),kw.get("kernel","full")
        gauge = kw.get("gauge","auto")
        if gauge=="auto": gauge = None # as in BSE, the result is exact in any gauge
        self.pairs = PairBasis(h,nk=nk,nv=kw.get("nv"),nc=kw.get("nc"),
                gauge=gauge)
        Wx = bare_interaction(h,V=V)
        W = get_direct_interaction(h,self.pairs,V=V,nk=nk,
                screening=kw.get("screening"),nkW=kw.get("nkW"),
                kernel=kernel,channel=kw.get("channel","charge"))
        self.f = ConvolutionKernel(self.pairs,W,Wx=Wx,kernel=kernel)
        self.cheb = chebyshev
        from .. import gpu
        self.prec = kw.get("kpm_prec","double")
        chebyshev.check_precision(self.prec) # before the device question
        self.device = gpu.get_gpu()
        lanczos = None
        if self.device:
            from ..bsetk import chebyshevjax
            lanczos = chebyshevjax.lanczos(self.f,self.prec)
            self.cheb = chebyshevjax
        elif self.prec!="double": raise ValueError("kpm_prec=%r needs the "
                "GPU path, pyqula.gpu.set_gpu(True); the CPU expansion "
                "runs in double precision"%(self.prec,))
        self.c,self.s = chebyshev.spectral_bounds(self.f,lanczos=lanczos)
    def apply(self,D,zs):
        if self.f.nterm==0: # kernel="none", A is diagonal, so exactly
            return D[:,:,None]/(zs[None,None,:]-self.f.diag[:,None,None])
        if self.device: # the device kernels take the block itself
            return self.cheb.resolvent_apply(self.f,D,zs,self.c,self.s,
                    kpm_prec=self.prec)
        return self.cheb.resolvent_apply(self.f.matvec,D,zs,self.c,self.s)
    def braket(self,D,zs):
        if self.f.nterm==0:
            den = 1./(zs[None,:]-self.f.diag[:,None]) # (npair,nz)
            return np.einsum("pa,pb,pz->zab",np.conj(D),D,den,optimize=True)
        if self.device:
            return self.cheb.resolvent_braket(self.f,D,zs,self.c,self.s,
                    kpm_prec=self.prec)
        return self.cheb.resolvent_braket(self.f.matvec,D,zs,self.c,self.s)


# how the resolvent of the BSE is evaluated, selected by method=
_methods = {"dense":_DenseResolvent,"kpm":_ChebyshevResolvent}


def get_methods():
    """The accepted values of method="""
    return sorted(_methods)


def _bse(h,nk,method,bse_kwargs):
    """The resolvent of the Tamm-Dancoff BSE at Q=0, with every exciton"""
    if h.dimensionality not in (1,2): raise NotImplementedError(
            "the excitonic optical response is only implemented for "
            "dimensionality 1 or 2, got "+str(h.dimensionality))
    for key in ["Q","tda","solver"]:
        if key in bse_kwargs: raise ValueError("the excitonic response "
                "fixes "+key+" (Q=0, tda=True, and the solver through "
                "method=), it cannot be passed")
    if method not in _methods: raise ValueError("method must be one of "
            "%s, got %r"%(", ".join(repr(m) for m in get_methods()),method))
    if method!="kpm" and "kpm_prec" in bse_kwargs: raise TypeError(
            "kpm_prec is the precision of the Chebyshev expansion of "
            "method='kpm'; method=%r diagonalizes in double precision"
            %(method,))
    return _methods[method](h,nk,bse_kwargs)


def _pair_dipoles(h,pb):
    """Interband dipoles d^a_p = <c k|r_a|v k> = v^a_cv/(i(E_c - E_v)) of
    every pair, with the velocity of the Peierls-coupled model (see
    kubo._velocities) rotated with the very eigenvectors the BSE was built
    from, so that the phases of d and of the amplitudes U agree"""
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    ph = kubo._bloch_phases(dirs,pb.kpoints)
    vb,cb = pb.vbands,pb.cbands
    nk = len(pb.kpoints)
    out = np.zeros((3,nk,len(vb),len(cb)),dtype=np.complex128)
    for a in range(3):
        va = shg._cartesian_derivative(tms,d,ph,[a]) # (nk,n,n), orbitals
        for ik in range(nk):
            C = pb.ck[ik] # rows are the eigenvectors
            m = np.conj(C[cb])@va[ik]@C[vb].T # <c|v_a|v>, (nc,nv)
            dE = pb.ek[ik][cb][:,None] - pb.ek[ik][vb][None,:]
            out[a,ik] = (m/(1j*dE)).T # (nv,nc), the pair order (k,v,c)
    return out.reshape(3,-1),cellvol


class _PositionOperator():
    """Matrix elements of the many-body position operator between two
    excitations, X^a = <n|R_a|m> - delta_nm <0|R_a|0>, applied to a pair
    vector g. The pair vector is written as a matrix in orbital space,

      Phi_k = sum_{vc} g_{kvc} |u_ck><u_vk| ,

    which does not depend on the phases of the Bloch eigenvectors and, in
    pyqula's lattice gauge, is periodic in the reduced momentum. Its
    Fourier transform on the mesh, phi_ij(R), is the amplitude of an
    electron on orbital i of cell R and a hole on orbital j of cell 0, and
    the position operator multiplies it by their relative coordinate,

      X^a phi_ij(R) = -(R + tau_j - tau_i)_a phi_ij(R) ,

    which is i dPhi/dK + [tau,Phi] written in real space. On the periodic
    mesh R is only defined modulo the supercell, and it is taken as the
    minimal image of the bond, the R that makes |R + tau_j - tau_i|
    shortest (ties averaged, which for a one-site cell is the spectral
    derivative without its Nyquist mode). The bond is a geometric object,
    so the operator transforms as a vector under every point-group
    operation of the lattice, which a derivative in reduced coordinates
    does not do once the orbitals sit away from the origin of the cell
    (the C3 of a stacked multilayer). No smooth gauge of the eigenvectors
    is needed, which is the step that ab initio implementations do with a
    gauge-fixing construction."""
    def __init__(self,h,pb,nk):
        self.dim = h.dimensionality
        self.nks = [nk]*self.dim
        self.vb,self.cb = pb.vbands,pb.cbands
        self.C = pb.ck # (nk,norb,norb), rows the eigenvectors
        n = self.C.shape[1]
        Cc = np.array([c[self.cb] for c in self.C]) # (nk,nc,norb)
        Cv = np.array([c[self.vb] for c in self.C]) # (nk,nv,norb)
        # the arrays the gather and scatter of bsetk/convolution.py read,
        # with every orbital channel (i,j)
        self.CcT = np.ascontiguousarray(np.swapaxes(Cc,1,2))
        self.CvcT = np.ascontiguousarray(np.swapaxes(np.conj(Cv),1,2))
        self.Ccc,self.Cv = np.ascontiguousarray(np.conj(Cc)),Cv
        self.ia,self.ib = np.repeat(np.arange(n),n),np.tile(np.arange(n),n)
        g = h.geometry
        r = np.array(g.r)
        avecs = np.array([g.a1,g.a2][:self.dim])
        F = _bond_vectors(r,avecs,nk) # (3,nk..,nsites,nsites)
        m = n//len(r) # orbitals per site
        F = np.repeat(np.repeat(F,m,axis=-1),m,axis=-2)
        # one channel axis, broadcast over the columns
        self.F = F.reshape((3,)+tuple(self.nks)+(1,n*n))
    def apply(self,a,g):
        """X^a g for a batch of pair vectors g (npair,nb): the pair vectors
        as orbital matrices Phi_k[i,j] = sum_vc g_kvc C_c(k)_i conj(C_v(k)_j),
        multiplied by the bond vector in real space, and projected back"""
        return self.backward(a,self.forward(g))
    def forward(self,g):
        """The pair vectors g (npair,nb) as orbital matrices in real space,
        phi_ij(R), which do not depend on the direction of X and so serve
        all three"""
        from ..bsetk.convolution import _gather
        import scipy.fft
        nk,nv,nc = len(self.C),len(self.vb),len(self.cb)
        ncol = g.shape[1]
        G = np.ascontiguousarray(g,dtype=np.complex128).reshape(nk,nv,nc,ncol)
        Phi = np.empty((nk,ncol,len(self.ia)),dtype=np.complex128)
        parallel.set_num_threads()
        _gather(self.CcT,self.CvcT,G,self.ia,self.ib,Phi)
        axes = tuple(range(self.dim))
        return scipy.fft.fftn(Phi.reshape(tuple(self.nks)+Phi.shape[1:]),
                axes=axes,workers=numba.get_num_threads(),overwrite_x=True)
    def backward(self,a,P):
        """X^a applied to the output of forward, as pair vectors"""
        from ..bsetk.convolution import _scatter
        import scipy.fft
        nk,nv,nc = len(self.C),len(self.vb),len(self.cb)
        ncol = P.shape[-2]
        axes = tuple(range(self.dim))
        # times the bond vector of every R and (i,j), into a new array, so
        # P serves the other directions too
        M = scipy.fft.ifftn(P*self.F[a],axes=axes,
                workers=numba.get_num_threads(),overwrite_x=True)
        out = np.zeros((nk,nv,nc,ncol),dtype=np.complex128)
        parallel.set_num_threads()
        _scatter(self.Ccc,self.Cv,M.reshape(nk,ncol,-1),self.ia,self.ib,out)
        return out.reshape(-1,ncol)


def _bond_vectors(r,avecs,nk):
    """F[a,m...,i,j] = -(R + tau_j - tau_i)_a for the mesh frequency m
    (one index per lattice direction), with R = sum_l (m_l + nk p_l) a_l
    the minimal image of the bond from site i to site j on the nk-periodic
    supercell, averaged over the images of equal length"""
    dim,ns = len(avecs),len(r)
    grid = np.meshgrid(*[np.arange(nk)]*dim,indexing="ij")
    shifts = np.meshgrid(*[np.arange(-2,3)]*dim,indexing="ij")
    shifts = [s.reshape((-1,)+(1,)*dim) for s in shifts]
    # every candidate lattice vector, (ncand,nk..,3)
    Rs = sum((grid[l][None]+nk*shifts[l])[...,None]*avecs[l]
            for l in range(dim))
    F = np.zeros((3,)+(nk,)*dim+(ns,ns))
    for i in range(ns):
        dv = Rs[...,None,:] + (r[None,:,:] - r[i]) # (ncand,nk..,ns,3)
        d2 = np.sum(dv*dv,axis=-1)
        sel = d2<=d2.min(axis=0)[None]*(1.+1e-8) + 1e-10
        dm = np.sum(dv*sel[...,None],axis=0)/np.sum(sel,axis=0)[...,None]
        F[...,i,:] = -np.moveaxis(dm,-1,0)
    return F


def _check_delta(delta):
    if not delta>0.: raise ValueError("the broadening delta must be "
            "positive, the exciton sums have poles on the real axis, got "
            "delta="+str(delta))


def excitonic_optical_conductivity(h,energies=None,nk=20,delta=0.1,
        component=None,method="dense",**bse_kwargs):
    """Excitonic optical conductivity, see
    conductivity.excitonic_optical_conductivity"""
    if energies is None: energies = np.linspace(0.,4.,100)
    _check_delta(delta)
    w = np.array(energies,dtype=np.float64) + 1j*delta
    res = _bse(h,nk,method,bse_kwargs)
    d,cellvol = _pair_dipoles(h,res.pairs)
    # G[z,a,c] = d_a^+ (z - A)^-1 d_c, at z = w and z = -w
    G = res.braket(d.T,np.concatenate([w,-w]))
    Gp,Gm = G[:len(w)],G[len(w):]
    sigma = np.zeros((len(w),3,3),dtype=np.complex128)
    for a in range(3):
        for c in range(3):
            # sum_n R^a_0n R^c_n0/(E_n - w) + R^c_0n R^a_n0/(E_n + w)
            chi = -Gp[:,a,c] - Gm[:,c,a]
            # J = dP/dt misses the frequency-independent antisymmetric
            # (Berry curvature) part of the conductivity, which for an
            # insulator is its Chern number and is not changed by the
            # excitons; it is added at the independent-particle level,
            # -i q^2 sum_vck (r^a_vc r^c_cv - r^a_cv r^c_vc), which makes
            # the kernel="none" limit the Kubo result for every component
            hall = -1j*np.sum(np.conj(d[a])*d[c] - d[a]*np.conj(d[c]))
            sigma[:,a,c] = (-1j*w*chi + hall)*shg._charge**2
    sigma = sigma/(len(res.pairs.kpoints)*cellvol)
    if component is not None:
        return np.real(w),sigma[:,kubo._index(component[0]),
                kubo._index(component[1])]
    return np.real(w),sigma


batch_memory = 2. # GB, the frequencies of one batch of the SHG


def _frequency_batch(pb,nfreq):
    """How many frequencies of the SHG fit in batch_memory: per frequency,
    the resolvents at four energies on three dipoles, and about three
    copies of the orbital matrices of the position operator. Each batch
    runs the Chebyshev recursion again, so the larger the better"""
    nk,norb = len(pb.kpoints),pb.el.shape[1]
    per = 16.*(12*pb.npair + 3*nk*norb**2) # bytes per frequency
    return int(max(1,min(nfreq,batch_memory*1e9//per)))


def excitonic_shg_conductivity(h,energies=None,nk=20,delta=0.1,
        components=None,wbatch=None,method="dense",**bse_kwargs):
    """Excitonic second-harmonic conductivity sigma_abc(2w;w,w), see
    conductivity.excitonic_shg_conductivity"""
    if energies is None: energies = np.linspace(0.,4.,100)
    energies = np.array(energies,dtype=np.float64)
    _check_delta(delta)
    single = isinstance(components,str)
    if components is None: components = shg._all_components
    if single: components = [components]
    trip,full,triplets,pairs,ipair = shg._triplets(components)
    res = _bse(h,nk,method,bse_kwargs)
    pb = res.pairs
    d,cellvol = _pair_dipoles(h,pb)
    X = _PositionOperator(h,pb,nk)
    if wbatch is None: wbatch = _frequency_batch(pb,len(energies))
    elif not (int(wbatch)==wbatch and wbatch>=1): raise ValueError("wbatch "
            "is the number of frequencies per batch, a positive integer, "
            "got %r"%(wbatch,))
    wbatch = int(wbatch)
    def dag(x,y): return np.sum(np.conj(x)*y,axis=0) # x^+ y, per column
    chi = np.zeros((len(energies),len(full)),dtype=np.complex128)
    for i0 in range(0,len(energies),wbatch): # batches of frequencies
        w = energies[i0:i0+wbatch] + 1j*delta
        nw = len(w)
        # the four resolvents applied to the three dipoles, one column per
        # frequency: (w - A)^-1 d, (-2w - A)^-1 d, and the two whose
        # Hermitian conjugates appear on the left, at conj(2w) and -conj(w)
        F = res.apply(d.T,np.concatenate([w,-2.*w,np.conj(2.*w),
            -np.conj(w)]))
        Fm1 = [F[:,a,0:nw] for a in range(3)] # (w - A)^-1 d^a
        Fp2 = [F[:,a,nw:2*nw] for a in range(3)] # (-2w - A)^-1 d^a
        L2 = [F[:,a,2*nw:3*nw] for a in range(3)] # ((2w - A)^-1)^+ d^a
        Lm = [-F[:,a,3*nw:] for a in range(3)] # ((w + A)^-1)^+ d^a
        # for every triplet t = (a,b,c), the sum of the three terms
        #   1:  d^a+ (2w - A)^-1 X^b (w - A)^-1 d^c
        #   2:  d^b+ (w + A)^-1 X^c (2w + A)^-1 d^a
        #   3: -d^c+ (w + A)^-1 X^a (w - A)^-1 d^b
        # Each X product depends on a direction and a source vector only,
        # so the products are grouped by source: one forward transform per
        # source, one backward transform per direction it is needed in,
        # used by every term and triplet that needs it before the next one
        # is made. That is at most 6 forward and 18 backward transforms
        # per batch, where one per term and triplet was 81 of each for
        # every component
        uses = {} # (source,direction) -> [(triplet,left vector,sign)]
        for (t,(a,bb,c)) in enumerate(full):
            uses.setdefault((("m1",c),bb),[]).append((t,L2[a],1.))
            uses.setdefault((("p2",a),c),[]).append((t,Lm[bb],1.))
            uses.setdefault((("m1",bb),a),[]).append((t,Lm[c],-1.))
        sources = {("m1",c):Fm1[c] for c in range(3)}
        sources.update({("p2",a):-Fp2[a] for a in range(3)})
        for src in sorted({s for (s,_) in uses}):
            P = X.forward(sources[src])
            for (s,b) in sorted(uses):
                if s!=src: continue
                XF = X.backward(b,P)
                for (t,L,sign) in uses[(s,b)]:
                    chi[i0:i0+wbatch,t] += sign*dag(L,XF)
    chi = chi*shg._charge**3/(2.*len(pb.kpoints)*cellvol)
    out = {}
    for (comp,(a,bb,c)) in zip(components,trip):
        x = chi[:,full.index((a,bb,c))] + chi[:,full.index((a,c,bb))]
        out[comp] = -2j*(energies+1j*delta)*x # sigma = -2 i w chi
    if single: return energies,out[components[0]]
    return energies,out


def excitonic_shg_susceptibility(h,length_unit=1.0,energy_unit=1.0,**kwargs):
    """Excitonic second-harmonic susceptibility in SI units, see
    conductivity.excitonic_shg_susceptibility"""
    import inspect # the default delta is the conductivity's own
    delta = kwargs.get("delta",inspect.signature(
        excitonic_shg_conductivity).parameters["delta"].default)
    energies,sigma = excitonic_shg_conductivity(h,**kwargs)
    return shg.to_susceptibility(h,energies,sigma,delta,length_unit,
            energy_unit)
