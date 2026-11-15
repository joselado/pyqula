"""Hellmann-Feynman for the ultracell mean field.

At a fixed chemical potential, with the interaction scaled by lambda, the
self-consistent grand potential Omega(lambda) is the minimum of the
mean-field functional over the states the ultracell can describe, so its
slope at lambda=1 is the interaction energy, the band energy of the
converged Hamiltonian minus Omega. This holds exactly only when that set
of states does not move with lambda: the clean basis, whose kept states
and frozen density are computed once, with a window that cuts, and every
state for a Nambu supercell. A wrong band energy of the frozen states or a
wrong double counting of the pairing gives a slope of first order in the
error, while a central difference leaves an eps^2 residue.
"""

import numpy as np

from pyqula import geometry
from pyqula.ultracell import Ultracell
from pyqula.ultracelltk.meanfield import SparseBackend, MeshStates
from pyqula.superconductivity import get_eh_sector


def fractional(g):
    B = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)
    return lambda r: B@np.array(r)


def slope_and_interaction(u,kw,eps,**extra):
    """The central difference of Omega(lambda) at lambda=1, the
    interaction energy band - Omega of the converged state, and that
    state, with U and V1 of kw scaled by lambda"""
    def run(lam):
        k = dict(kw)
        for name in ("U","V1"):
            if name in k: k[name] = lam*k[name]
        return u.get_mean_field_hamiltonian(return_total_energy=True,
                rebuild_basis=False,maxerror=1e-12,verbose=0,write=False,
                **extra,**k)
    u1,omega = run(1.)
    hc = u1.hs.copy()
    hc.ultracell_basis = u1.basis
    band = SparseBackend(nk=kw["nk"]).band_energy(hc)
    slope = (run(1.+eps)[1]-run(1.-eps)[1])/(2.*eps)
    return slope,band-omega,hc


def test_a_window_on_the_clean_basis_obeys_hellmann_feynman():
    """A modulated honeycomb with V1, a window that freezes the lowest
    states and drops the highest, at fixed mu: the slope of Omega is the
    interaction energy"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    hs = h.get_supercell([3,2])
    x = fractional(hs.geometry)
    hs.add_onsite(lambda r: 0.4*np.cos(2*np.pi*x(r)[0]+0.2)
            + 0.2*np.sin(2*np.pi*(x(r)[0]+x(r)[1])))
    u = Ultracell(h,hs,window=(-2.5,2.))
    kw = dict(V1=1.0,mu=0.3,nk=2)
    slope,eint,hc = slope_and_interaction(u,kw,3e-4)
    states = MeshStates(hc,nk=2)
    n = hc.intra.shape[0]
    assert states.nfrozen>0 # the window freezes states
    kept = sum(len(e) for e in states.es)
    assert kept+states.nfrozen<n*len(states.ks) # and drops others
    assert abs(eint)>1. # a sizable interaction energy
    assert abs(slope-eint)<1e-8


def test_a_paired_nambu_ultracell_obeys_hellmann_feynman():
    """A modulated chain with an attractive U, every state kept, at fixed
    mu: the pairing is alive and the slope of Omega is the interaction
    energy, pairing included"""
    h = geometry.chain().get_hamiltonian()
    hs = h.get_supercell(4)
    hs.add_onsite(lambda r: 0.3*np.cos(2*np.pi*r[0]/4+0.4))
    h.setup_nambu_spinor()
    hs.setup_nambu_spinor()
    u = Ultracell(h,hs)
    kw = dict(U=-2.0,mu=-0.8,nk=6)
    slope,eint,hc = slope_and_interaction(u,kw,3e-4,mf="swave",mix=0.5,
            maxite=10000)
    delta = np.abs(get_eh_sector(np.array(hc.intra.todense()),i=0,j=1)).max()
    assert delta>0.1 # the premise: a paired state
    assert abs(slope-eint)<1e-6
