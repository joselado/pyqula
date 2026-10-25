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
