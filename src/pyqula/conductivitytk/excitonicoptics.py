# Excitonic linear and second-harmonic optical response of a periodic
# tight-binding insulator, from the Tamm-Dancoff Bethe-Salpeter equation.
# The physics, the units and the conventions are documented in the public
# entry points of conductivity.py (excitonic_optical_conductivity and
# excitonic_shg_conductivity); this module holds the machinery.
import numpy as np

from . import kubo
from . import shg


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
    the cost is linear in the number of pairs"""
    _keys = ("V","nv","nc","kernel","screening","nkW","channel","gauge")
    def __init__(self,h,nk,bse_kwargs):
        from ..bsetk.pairbasis import PairBasis
        from ..bsetk.interaction import bare_interaction
        from ..bsetk.solve import get_direct_interaction
        from ..bsetk.factorize import KernelFactorization
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
        self.f = KernelFactorization(self.pairs,W,Wx=Wx,kernel=kernel,
                block="A")
        self.c,self.s = chebyshev.spectral_bounds(self.f)
        self.cheb = chebyshev
    def apply(self,D,zs):
        return self.cheb.resolvent_apply(self.f.matvec,D,zs,self.c,self.s)
    def braket(self,D,zs):
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
        self.Cc = np.array([c[self.cb] for c in self.C]) # (nk,nc,norb)
        self.Cv = np.array([c[self.vb] for c in self.C]) # (nk,nv,norb)
        g = h.geometry
        r = np.array(g.r)
        avecs = np.array([g.a1,g.a2][:self.dim])
        F = _bond_vectors(r,avecs,nk) # (3,nk..,nsites,nsites)
        m = n//len(r) # orbitals per site
        self.F = np.repeat(np.repeat(F,m,axis=-1),m,axis=-2)
    def to_orbital(self,g):
        """Phi_k for a batch of pair vectors g (npair,nb)"""
        nk,nv,nc = len(self.C),len(self.vb),len(self.cb)
        G = g.reshape(nk,nv,nc,-1)
        # Phi[k,i,j,b] = sum_{vc} Cc[k,c,i] G[k,v,c,b] conj(Cv[k,v,j])
        return np.einsum("kci,kvcb,kvj->kijb",self.Cc,G,np.conj(self.Cv),
                optimize=True)
    def to_pairs(self,M):
        """Projection <u_ck|M_k|u_vk> back onto the pairs"""
        out = np.einsum("kci,kijb,kvj->kvcb",np.conj(self.Cc),M,self.Cv,
                optimize=True)
        return out.reshape(-1,M.shape[-1])
    def apply(self,a,g):
        """X^a g for a batch of pair vectors g (npair,nb)"""
        Phi = self.to_orbital(g)
        shape = Phi.shape
        axes = tuple(range(self.dim))
        P = np.fft.fftn(Phi.reshape(tuple(self.nks)+shape[1:]),axes=axes)
        P = P*self.F[a][...,None] # the bond vector of every R and (i,j)
        M = np.fft.ifftn(P,axes=axes).reshape(shape)
        return self.to_pairs(M)


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


def excitonic_shg_conductivity(h,energies=None,nk=20,delta=0.1,
        components=None,wbatch=16,method="dense",**bse_kwargs):
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
        for (t,(a,bb,c)) in enumerate(full):
            # term 1: d^a+ (2w - A)^-1 X^b (w - A)^-1 d^c
            T1 = dag(L2[a],X.apply(bb,Fm1[c]))
            # term 2: d^b+ (w + A)^-1 X^c (2w + A)^-1 d^a
            T2 = dag(Lm[bb],X.apply(c,-Fp2[a]))
            # term 3: -d^c+ (w + A)^-1 X^a (w - A)^-1 d^b
            T3 = -dag(Lm[c],X.apply(a,Fm1[bb]))
            chi[i0:i0+wbatch,t] = T1 + T2 + T3
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
    delta = kwargs.get("delta",0.1)
    energies,sigma = excitonic_shg_conductivity(h,**kwargs)
    return shg.to_susceptibility(h,energies,sigma,delta,length_unit,
            energy_unit)
