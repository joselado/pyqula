# Excitonic linear and second-harmonic optical response of a periodic
# tight-binding insulator, from the Tamm-Dancoff Bethe-Salpeter equation.
# The physics, the units and the conventions are documented in the public
# entry points of conductivity.py (excitonic_optical_conductivity and
# excitonic_shg_conductivity); this module holds the machinery.
import numpy as np

from . import kubo
from . import shg


def _bse(h,nk,bse_kwargs):
    """Tamm-Dancoff BSE at Q=0 on the nk mesh, dense, so that every
    exciton is available: energies E[n] and amplitudes U[p,n] over the
    pairs p = (k,v,c), the exciton n being sum_p U[p,n] c+_ck c_vk |0>"""
    from ..bsetk.solve import BSE
    if h.dimensionality not in (1,2): raise NotImplementedError(
            "the excitonic optical response is only implemented for "
            "dimensionality 1 or 2, got "+str(h.dimensionality))
    for key in ["Q","tda","solver"]:
        if key in bse_kwargs: raise ValueError("the excitonic response "
                "fixes "+key+" (Q=0, tda=True, solver='dense'), it cannot "
                "be passed")
    b = BSE(h,nk=nk,tda=True,solver="dense",**bse_kwargs)
    return b,np.real(b.energies),np.array(b.amplitudes.T)


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
    pyqula's lattice gauge, is periodic in the reduced momentum. The
    position operator, diagonal in the orbitals with positions R + tau,
    acts on it as

      X^a Phi = i dPhi/dK_a + [tau_a, Phi] ,

    and the result is projected back onto the pairs. The k-derivative is
    spectral on the uniform periodic mesh, so no smooth gauge of the
    eigenvectors is needed, which is the step that ab initio
    implementations do with a gauge-fixing construction."""
    def __init__(self,h,pb,nk):
        self.dim = h.dimensionality
        self.nks = [nk]*self.dim
        self.vb,self.cb = pb.vbands,pb.cbands
        self.C = pb.ck # (nk,norb,norb), rows the eigenvectors
        n = self.C.shape[1]
        self.Cc = np.array([c[self.cb] for c in self.C]) # (nk,nc,norb)
        self.Cv = np.array([c[self.vb] for c in self.C]) # (nk,nv,norb)
        r = np.array(h.geometry.r)
        self.tau = np.repeat(r,n//len(r),axis=0) # orbital positions
        hm,orders,hkgen,jac,dr,cellvol,scale = kubo._setup(h)
        self.jac = jac # jac[i,a] = a_i[a]/(2 pi)
        # 2 pi i times the integer frequency of each reduced direction
        self.freqs = [2j*np.pi*np.fft.fftfreq(m)*m for m in self.nks]
        for (f,m) in zip(self.freqs,self.nks):
            if m%2==0: f[m//2] = 0. # drop the Nyquist mode, it has no
                                    # well-defined derivative
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
    def derivatives(self,Phi):
        """d Phi/dk_i along every reduced direction, spectrally"""
        shape = Phi.shape
        P = Phi.reshape(tuple(self.nks)+shape[1:])
        Pf = np.fft.fftn(P,axes=tuple(range(self.dim)))
        out = []
        for i in range(self.dim):
            f = self.freqs[i].reshape([-1 if j==i else 1
                    for j in range(self.dim)]+[1]*(len(shape)-1))
            out.append(np.fft.ifftn(Pf*f,axes=tuple(range(self.dim)))
                    .reshape(shape))
        return out
    def apply(self,a,g,dks=None,Phi=None):
        """X^a g for a batch of pair vectors g (npair,nb)"""
        if Phi is None: Phi = self.to_orbital(g)
        if dks is None: dks = self.derivatives(Phi)
        M = 1j*sum(self.jac[i,a]*dks[i] for i in range(self.dim))
        t = self.tau[:,a]
        M = M + (t[None,:,None,None] - t[None,None,:,None])*Phi
        return self.to_pairs(M)


def _check_delta(delta):
    if not delta>0.: raise ValueError("the broadening delta must be "
            "positive, the exciton sums have poles on the real axis, got "
            "delta="+str(delta))


def _resolvent(E,U,Ud,z):
    """(z - A)^{-1} d for a batch of complex z, with A = U diag(E) U^+ and
    Ud = U^+ d precomputed. Returns (npair,nz)"""
    return U@(Ud[:,None]/(z[None,:]-E[:,None]))


def excitonic_optical_conductivity(h,energies=None,nk=20,delta=0.1,
        component=None,**bse_kwargs):
    """Excitonic optical conductivity, see
    conductivity.excitonic_optical_conductivity"""
    if energies is None: energies = np.linspace(0.,4.,100)
    _check_delta(delta)
    w = np.array(energies,dtype=np.float64) + 1j*delta
    b,E,U = _bse(h,nk,bse_kwargs)
    d,cellvol = _pair_dipoles(h,b.pairs)
    R0 = U.T@np.conj(d).T # R0[n,a] = <0|R_a|n> = sum_p U[p,n] conj(d_p)
    sigma = np.zeros((len(w),3,3),dtype=np.complex128)
    for a in range(3):
        for c in range(3):
            num1 = R0[:,a]*np.conj(R0[:,c]) # R^a_0n R^c_n0
            num2 = R0[:,c]*np.conj(R0[:,a]) # R^c_0n R^a_n0
            chi = (num1[None,:]/(E[None,:]-w[:,None])
                   + num2[None,:]/(E[None,:]+w[:,None])).sum(axis=1)
            # J = dP/dt misses the frequency-independent antisymmetric
            # (Berry curvature) part of the conductivity, which for an
            # insulator is its Chern number and is not changed by the
            # excitons; it is added at the independent-particle level,
            # -i q^2 sum_vck (r^a_vc r^c_cv - r^a_cv r^c_vc), which makes
            # the kernel="none" limit the Kubo result for every component
            hall = -1j*np.sum(np.conj(d[a])*d[c] - d[a]*np.conj(d[c]))
            sigma[:,a,c] = (-1j*w*chi + hall)*shg._charge**2
    sigma = sigma/(len(b.pairs.kpoints)*cellvol)
    if component is not None:
        return np.real(w),sigma[:,kubo._index(component[0]),
                kubo._index(component[1])]
    return np.real(w),sigma


def excitonic_shg_conductivity(h,energies=None,nk=20,delta=0.1,
        components=None,wbatch=16,**bse_kwargs):
    """Excitonic second-harmonic conductivity sigma_abc(2w;w,w), see
    conductivity.excitonic_shg_conductivity"""
    if energies is None: energies = np.linspace(0.,4.,100)
    energies = np.array(energies,dtype=np.float64)
    _check_delta(delta)
    single = isinstance(components,str)
    if components is None: components = shg._all_components
    if single: components = [components]
    trip,full,triplets,pairs,ipair = shg._triplets(components)
    b,E,U = _bse(h,nk,bse_kwargs)
    pb = b.pairs
    d,cellvol = _pair_dipoles(h,pb)
    Ud = np.conj(U.T)@d.T # U^+ d^a for the three directions, (nexc,3)
    X = _PositionOperator(h,pb,nk)
    def dag(x,y): return np.sum(np.conj(x)*y,axis=0) # x^+ y, per column
    chi = np.zeros((len(energies),len(full)),dtype=np.complex128)
    for i0 in range(0,len(energies),wbatch): # batches of frequencies
        w = energies[i0:i0+wbatch] + 1j*delta
        # resolvents applied to the dipoles, one column per frequency
        Fm1 = [_resolvent(E,U,Ud[:,a],w) for a in range(3)] # (w - A)^-1 d
        Fp2 = [_resolvent(E,U,Ud[:,a],-2.*w) for a in range(3)] # (-2w-A)^-1
        for (t,(a,bb,c)) in enumerate(full):
            # term 1: d^a+ (2w - A)^-1 X^b (w - A)^-1 d^c
            y = X.apply(bb,Fm1[c])
            l = _resolvent(E,U,Ud[:,a],np.conj(2.*w)) # ((2w-A)^-1)^+ d^a
            T1 = dag(l,y)
            # term 2: d^b+ (w + A)^-1 X^c (2w + A)^-1 d^a
            y = X.apply(c,-Fp2[a]) # (2w + A)^-1 d^a = -(-2w - A)^-1 d^a
            l = -_resolvent(E,U,Ud[:,bb],-np.conj(w)) # ((w+A)^-1)^+ d^b
            T2 = dag(l,y)
            # term 3: -d^c+ (w + A)^-1 X^a (w - A)^-1 d^b
            y = X.apply(a,Fm1[bb])
            l = -_resolvent(E,U,Ud[:,c],-np.conj(w)) # ((w+A)^-1)^+ d^c
            T3 = -dag(l,y)
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
