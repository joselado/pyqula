"""Observables of the ultracell: the unfolded spectral function, the density
of states, the local density of states and expectation values per site, and
the harmonics of a quantity defined on the sites.

Every Lorentzian sum goes through dostk.eigtodos.calculate_dos with the 1/pi
of the rest of the package, so that each routine gives the number its
counterpart on the unit cell or on the supercell gives: the unfolded
spectral function at zero modulation is h.get_kdos_bands() of the unit cell,
the density of states is per unit cell, and the local density of states and
the expectation values are those of the supercell, site by site.
"""

import numpy as np

from .. import filewrite
from ..dostk.eigtodos import calculate_dos
from .basis import pad3
from .matrix import eigenstates, eigenvalues, real_space_states, state_chunks


def check_nk(nk):
    """nk is one positive integer, the same in every direction, or one per
    direction, as klist.kmesh takes it"""
    ns = np.atleast_1d(np.asarray(nk)).ravel()
    if len(ns)==0 or np.any(ns<1) or np.any(ns!=np.round(ns)):
        raise ValueError("nk must be a positive integer or one per "
                "direction, got "+repr(nk))
    return int(ns[0]) if np.ndim(nk)==0 else [int(n) for n in ns]


def check_delta(delta):
    from ..utilities import check_delta as check
    check(delta)
    return delta


def kdos_bands(u,kpath=None,energies=np.linspace(-3.0,3.0,200),delta=0.01,
        nk=100,write=None):
    """Unfolded spectral function along a path of the unit cell's Brillouin
    zone, from the weight of each ultracell state on the folded momentum
    equal to the point of the path, sum_m |c_{q=0,m}|^2, which is the weight
    per unit cell. Returns the three columns of h.get_kdos_bands(): position
    along the path, energy and spectral function. The states of the unit
    cell above emax are not in the basis, so energies above emax carry no
    weight"""
    from ..kdos import write_kdos_bands
    write = filewrite.resolve(write,True) # the call, else the global switch
    check_delta(delta)
    kpath = u.h.geometry.get_kpath(kpath,nk=nk,write=write) # unit-cell path
    out = []
    for p in kpath:
        kS = pad3(p)*u.basis.n # so that the folded momentum q=0 is p itself
        es,vs,U,valid = eigenstates(u.basis,u.hk(kS),kS)
        b0 = np.sum(valid[0]) # the kept states of q=0 come first
        w = np.sum(np.abs(vs[0:b0])**2,axis=0) # weight on q=0
        ys = calculate_dos(es,energies,delta,w=w)/np.pi
        out.append((energies,ys))
    return write_kdos_bands(kpath,out,write=write)


def ultracell_kmesh(u,nk):
    from ..klist import kmesh
    return kmesh(u.basis.dimensionality,nk=check_nk(nk))


def dos(u,energies=np.linspace(-4.0,4.0,400),delta=None,nk=10,write=None):
    """Density of states per unit cell, on an nk mesh of the ultracell's
    Brillouin zone. delta defaults to the width h.get_dos() would use on the
    mesh of the unit cell that the folded momenta form. Energies above emax
    carry no weight, since those states are not in the basis"""
    from ..dos import write_dos
    write = filewrite.resolve(write,True) # the call, else the global switch
    ks = ultracell_kmesh(u,nk)
    if delta is None:
        d = u.basis.dimensionality
        nks = np.zeros(d) + np.atleast_1d(check_nk(nk))[0:d] # per direction
        delta = 5./min(nks*u.basis.n[0:d]) # as dos_kmesh, on the folded mesh
    check_delta(delta)
    es = np.concatenate([eigenvalues(u.basis,u.hk(k),k) for k in ks])
    ys = calculate_dos(es,energies,delta)/(np.pi*len(ks)*u.basis.N)
    if write: write_dos(energies,ys)
    return (energies,ys)


def site_sum(u,kS,weight,operator=None):
    """sum_s weight(e_s) |Phi_s(a)|^2 on every orbital a of the supercell,
    or with Re[conj(Phi_s(a)) (O Phi_s)(a)] for an operator matrix O, over
    the eigenstates at kS whose weight is not zero, a chunk of states at a
    time"""
    es,vs,U,valid = eigenstates(u.basis,u.hk(kS),kS)
    w = weight(es)
    keep = w!=0.
    w,vs = w[keep],vs[:,keep]
    out = np.zeros(u.hs.intra.shape[0])
    for sl in state_chunks(u.basis,len(w)):
        Phi = real_space_states(u.basis,kS,vs[:,sl],U,valid)
        if operator is None: w2 = np.abs(Phi)**2
        else: w2 = np.real(np.conjugate(Phi)*np.asarray(operator@Phi))
        out += w2@w[sl]
    return out


def ldos(u,e=0.0,delta=0.001,nk=10,write=None):
    """Local density of states per site of the supercell at energy e,
    averaged over an nk mesh of the ultracell's Brillouin zone, with the
    normalization of hs.get_ldos(mode="diagonalization"). Returns x, y and
    the value on each site. Energies above emax carry no weight, since those
    states are not in the basis"""
    from ..ldos import write_ldos
    from ..increase_hilbert import full2profile
    write = filewrite.resolve(write,True) # the call, else the global switch
    check_delta(delta)
    ks = ultracell_kmesh(u,nk)
    d = 0.
    def lor(es): return delta/((e-es)**2 + delta**2)/np.pi # per state
    for k in ks: d = d + site_sum(u,k,lor)
    d = full2profile(u.hs,d/len(ks)) # sum the orbitals of each site
    g = u.hs.geometry
    if write: write_ldos(g.x,g.y,d)
    return (np.array(g.x),np.array(g.y),np.array(d))


def vev(u,operator=None,nk=10):
    """Expectation value per site of the supercell in the ground state with
    the Fermi level at zero energy, every state below it occupied, averaged
    over an nk mesh of the ultracell's Brillouin zone. Without an operator
    it is the occupation of each site, as hs.get_vev(). It needs every
    occupied state in the basis, so emax must be above zero"""
    from ..increase_hilbert import full2profile
    if u.hs.has_eh:
        raise NotImplementedError("expectation values of the ultracell are "
                "not implemented for a Hamiltonian with the electron-hole "
                "(Nambu) degree of freedom")
    if u.emax is not None and u.emax<=0.:
        raise ValueError("the expectation values occupy every state below "
                "zero energy, and with emax="+str(u.emax)+" the states of "
                "the unit cell between emax and zero are not in the basis; "
                "build the ultracell with an emax above zero")
    op = None
    if operator is not None:
        op = u.hs.get_operator(operator).get_matrix(required=False)
        if op is None:
            raise NotImplementedError("the ultracell takes an operator with "
                    "a matrix on the supercell, not one defined only by its "
                    "action on a state or depending on k")
    ks = ultracell_kmesh(u,nk)
    d = 0.
    def occupied(es): return (es<0.).astype(float)
    for k in ks: d = d + site_sum(u,k,occupied,operator=op)
    return full2profile(u.hs,d/len(ks)).real


def harmonics(u,values):
    """Fourier components over the cells of a quantity given on the sites of
    the supercell (the order of hs.geometry.r):
    f_Q(s) = (1/N) sum_R exp(-2 pi i Q.R) f(R,s), with s a site of the unit
    cell. Returns the Q vectors, (N,3) in reduced coordinates of the unit
    cell, and the components, (N,number of unit-cell sites). R is the cell
    index the supercell builder records, and the builders center the
    positions, so the phase of f_Q is measured from that cell and not from
    the origin of the positions; the modulus does not depend on it"""
    b = u.basis
    values = np.asarray(values).ravel()
    if len(values)!=len(b.site_cell):
        raise ValueError("harmonics takes one value per site of the "
                "supercell, "+str(len(b.site_cell))+", got "+str(len(values)))
    F = np.zeros((b.N,b.nsites),dtype=np.complex128)
    F[b.site_cell,b.site_primal] = values
    F = F.reshape(tuple(b.n)+(b.nsites,))
    F = np.fft.fftn(F,axes=(0,1,2)).reshape(b.N,b.nsites)/b.N
    return b.q/b.n[None,:],F
