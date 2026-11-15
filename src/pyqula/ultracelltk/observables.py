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
    along the path, energy and spectral function. Only the states inside
    the window (emin,emax) are solved, so energies outside it carry no
    weight: the states above emax are not in the basis, and the frozen
    states below emin are not drawn"""
    from ..kdos import write_kdos_bands
    write = filewrite.resolve(write,True) # the call, else the global switch
    check_delta(delta)
    kpath = u.h.geometry.get_kpath(kpath,nk=nk,write=write) # unit-cell path
    out = []
    for p in kpath:
        kS = u.basis.M@pad3(p) # so that the folded momentum q=0 is p itself
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
    Brillouin zone. delta defaults to the width h.get_dos() would use on a
    mesh of the unit cell with as many momenta as the folded mesh, N times
    the kpoints of the ultracell, the same number in every direction. For
    the sqrt3 x sqrt3 cell of graphene and nk=10 this is 300 momenta, about
    17 per direction. Energies outside the window (emin,emax) carry no
    weight: the states above emax are not in the basis, and the frozen
    states below emin are not drawn"""
    from ..dos import write_dos
    write = filewrite.resolve(write,True) # the call, else the global switch
    ks = ultracell_kmesh(u,nk)
    if delta is None:
        d = u.basis.dimensionality
        nks = np.zeros(d) + np.atleast_1d(check_nk(nk))[0:d] # per direction
        # the folded momenta per direction of the unit-cell zone; for a
        # supercell matrix that is not diagonal they are not a rectangular
        # mesh of the unit cell, so only their number measures them
        delta = 5./(u.basis.N*np.prod(nks))**(1./d) # as dos_kmesh
    check_delta(delta)
    es = np.concatenate([eigenvalues(u.basis,u.hk(k),k) for k in ks])
    ys = calculate_dos(es,energies,delta)/(np.pi*len(ks)*u.basis.N)
    if write: write_dos(energies,ys)
    return (energies,ys)


def site_sum(u,kS,weight,operator=None,frozen=False):
    """sum_s weight(e_s) |Phi_s(a)|^2 on every orbital a of the supercell,
    or with Re[conj(Phi_s(a)) (O Phi_s)(a)] for an operator matrix O, over
    the eigenstates at kS whose weight is not zero, a chunk of states at a
    time. Returns it and the eigenvalues at kS, and with frozen=True also
    the frozen states (Ef,Uf,validf) of the same basis states"""
    es,vs,U,valid,*fz = eigenstates(u.basis,u.hk(kS),kS,frozen=frozen)
    w = weight(es)
    # the states with a weight; the eigenvectors are local here, so the
    # copy of the kept columns replaces them instead of living beside them
    keep = w!=0.
    w,vs = w[keep],vs[:,keep]
    out = np.zeros(u.hs.intra.shape[0])
    for sl in state_chunks(u.basis,len(w)):
        Phi = real_space_states(u.basis,kS,vs[:,sl],U,valid)
        if operator is None: w2 = np.abs(Phi)**2
        else: w2 = np.real(np.conjugate(Phi)*np.asarray(operator@Phi))
        out += w2@w[sl]
    if frozen: return out,es,tuple(fz)
    return out,es


def ldos(u,e=0.0,delta=0.001,nk=10,write=None):
    """Local density of states per site of the supercell at energy e,
    averaged over an nk mesh of the ultracell's Brillouin zone, with the
    normalization of hs.get_ldos(mode="diagonalization"). Returns x, y and
    the value on each site. Energies outside the window (emin,emax) carry
    no weight: the states above emax are not in the basis, and the frozen
    states below emin are not drawn"""
    from ..ldos import write_ldos
    from ..increase_hilbert import full2profile
    write = filewrite.resolve(write,True) # the call, else the global switch
    check_delta(delta)
    ks = ultracell_kmesh(u,nk)
    d = 0.
    def lor(es): return delta/((e-es)**2 + delta**2)/np.pi # per state
    for k in ks: d = d + site_sum(u,k,lor)[0]
    d = full2profile(u.hs,d/len(ks)) # sum the orbitals of each site
    g = u.hs.geometry
    if write: write_ldos(g.x,g.y,d)
    return (np.array(g.x),np.array(g.y),np.array(d))


def vev(u,operator=None,nk=10):
    """Expectation value per site of the supercell with the Fermi level at
    zero energy, averaged over an nk mesh of the ultracell's Brillouin
    zone, with the Fermi-Dirac occupations of hs.get_vev(): at the
    temperature of the mean field that gave hs, else at the smearing
    delta_dm of densitymatrix, so that a level at zero energy is half
    occupied. Without an operator it is the occupation of each site, as
    hs.get_vev(). It needs every occupied state in the basis, so emax must
    be above zero, and with a cut zero energy must lie below the highest
    kept state of hs, as meanfield.fermi_level requires of a filling. The
    frozen states below emin are full and add their part, the frozen
    density matrix of meanfield.frozen_entries at the entries of the
    operator; every one of them must lie below zero energy by
    meanfield.empty_ratio smearings in hs"""
    from scipy.special import expit
    from ..densitymatrix import default_smearing, nonzero_smearing
    from ..increase_hilbert import full2profile
    from .meanfield import (empty_ratio, require_kept_above,
            require_frozen_below, periodic_part, frozen_levels)
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
    # the smearing hs.get_vev() reads, so that a level at the Fermi energy
    # (the two degenerate states of a metal the loop left there, or a
    # level at zero up to rounding) shares its electron as it does there,
    # instead of being counted by the sign of its rounding
    delta = nonzero_smearing(default_smearing(u.hs))
    def occupied(es): # empty from empty_ratio smearings above zero
        return np.where(es<=empty_ratio*delta,expit(-es/delta),0.)
    hp = periodic_part(u.basis,u.hs) # the energies of the frozen states
    d,esk,efk,frozen = 0.,[],[],[]
    for k in ks:
        dk,es,(Ef,Uf,validf) = site_sum(u,k,occupied,operator=op,
                frozen=True)
        d = d + dk ; esk.append(es)
        frozen.append(Uf) ; efk.append(frozen_levels(u.basis,k,Ef,Uf,validf,hp))
    nfrozen = sum(len(e) for e in efk)
    # a lattice-periodic shift in hs can occupy states the cut left out,
    # even with emax above zero, or empty frozen ones
    what = "zero energy, the Fermi level of the expectation values,"
    require_kept_above(esk,u.hs.intra.shape[0],delta,u.emax,what,
            nfrozen=nfrozen)
    require_frozen_below(efk,delta,u.emin,what)
    d = d/len(ks)
    if nfrozen>0: d = d + frozen_site_sum(u,ks,frozen,op)
    return full2profile(u.hs,d).real


def frozen_site_sum(u,ks,frozen,operator=None):
    """The part of the frozen states in site_sum averaged over the mesh
    ks, on every orbital a: Re sum_b O[a,b] dmf[(0,0,0)][a,b] with dmf the
    frozen density matrix (meanfield.frozen_entries), elementwise because
    dm is the transpose of rho, or Re dmf[(0,0,0)][a,a] without an
    operator. frozen holds the frozen vectors Uf of every kpoint"""
    from scipy.sparse import coo_matrix
    from .meanfield import frozen_entries
    n = u.hs.intra.shape[0]
    if operator is None:
        r = np.arange(n) ; c = r ; o = np.ones(n)
    else:
        m = coo_matrix(operator)
        r,c,o = m.row,m.col,m.data
    v = frozen_entries(u.basis,ks,frozen,{(0,0,0): (r,c)})[(0,0,0)]
    return np.bincount(r,weights=np.real(o*v),minlength=n)


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
    return b.folded_momenta(np.zeros(3)),F
