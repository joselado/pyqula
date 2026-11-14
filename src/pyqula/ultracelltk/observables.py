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
from .matrix import eigenstates, real_space_states


def check_nk(nk):
    if int(nk)!=nk or nk<1:
        raise ValueError("nk must be a positive integer, got "+repr(nk))
    return int(nk)


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
        es,C,U = eigenstates(u.basis,u.hk(kS),kS)
        w = np.sum(np.abs(C[0])**2,axis=0) # weight on q=0
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
        delta = 5./(check_nk(nk)*min(u.basis.n[0:d])) # as dos_kmesh, folded
    check_delta(delta)
    es = np.concatenate([eigenstates(u.basis,u.hk(k),k)[0] for k in ks])
    ys = calculate_dos(es,energies,delta)/(np.pi*len(ks)*u.basis.N)
    if write: write_dos(energies,ys)
    return (energies,ys)


def site_weights(u,kS,operator=None):
    """Eigenvalues at kS and the weight of every eigenstate on every orbital
    of the supercell, |Phi(a)|^2, or Re[conj(Phi(a)) (O Phi)(a)] with an
    operator matrix O, (norbs,nstates)"""
    es,C,U = eigenstates(u.basis,u.hk(kS),kS)
    Phi = real_space_states(u.basis,kS,C,U)
    if operator is None: return es,np.abs(Phi)**2
    return es,np.real(np.conjugate(Phi)*np.asarray(operator@Phi))


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
    for k in ks:
        es,w2 = site_weights(u,k)
        lor = delta/((e-es)**2 + delta**2)/np.pi # one Lorentzian per state
        d = d + w2@lor
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
    for k in ks:
        es,w2 = site_weights(u,k,operator=op)
        d = d + np.sum(w2[:,es<0.],axis=1)
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
