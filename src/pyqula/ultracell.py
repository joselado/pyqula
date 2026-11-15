"""The ultracell, or ultra long-range method: a supercell whose potential,
exchange field, hoppings or pairing change slowly from cell to cell, solved
in the Bloch states of its unit cell.

At a Bloch momentum k_S of the supercell, the Bloch states of the unit cell
at the folded momenta p_q = (k_S+q)/n diagonalize the clean supercell. The
ultracell keeps those with energy up to emax and diagonalizes the modulated
supercell in that basis, a matrix of the band energies on the diagonal plus
the modulation projected on the states. With every state kept (emax=None)
the result is the supercell exactly; with a cut, the one approximation is
that the states above emax do not take part in the modulated states.

u.get_mean_field_hamiltonian() runs the mean field of scftk on the
supercell with integration="ultracell" (ultracelltk/meanfield.py), so the
density matrix, the Fermi level and the band energy of every iteration come
from the ultracell states.

The method is that of T. Mueller, S. Sharma, E. K. U. Gross and
J. K. Dewhurst, Phys. Rev. Lett. 125, 256402 (2020), arXiv:2008.12573,
written for a tight-binding basis, where the unit-cell states at every
folded momentum are one small diagonalization each. The roadmap is
future_development/ultra_long_range.md.
"""

import numpy as np

from .ultracelltk import basis as basistk
from .ultracelltk import matrix as matrixtk
from .ultracelltk import observables


class Ultracell():
    """A supercell hs of the unit cell h, solved in the Bloch states of h
    with energy up to emax. Built by h.get_ultracell(hs,emax=...).

    rebuild_basis=True takes, at each folded momentum, the states of the
    lattice-periodic part of hs instead of those of h, as many as the cut
    keeps in h. This is the default of the mean field, whose
    lattice-periodic part (a uniform moment) can be large; with every
    state kept the two bases span the same space"""
    def __init__(self,h,hs,emax=None,rebuild_basis=False):
        if h.non_hermitian or hs.non_hermitian:
            raise NotImplementedError("the ultracell is only implemented for "
                    "Hermitian Hamiltonians")
        self.h = h.copy() # the unit cell, which gives the basis
        self.hs = hs.copy() # the modulated supercell
        self.hs.turn_sparse() # it is only applied to vectors
        if hasattr(self.hs,"ultracell_basis"): del self.hs.ultracell_basis
        self.basis = basistk.FoldedBasis(self.h,self.hs,emax=emax)
        self.rebuild_basis = bool(rebuild_basis)
        if self.rebuild_basis: # the states of the periodic part of hs
            self.basis = self.basis.rebuilt(self.hs)
        self.basis.rebuild = self.rebuild_basis
        self.emax = self.basis.emax
        self._hk = self.hs.get_hk_gen()
    def hk(self,kS):
        """Bloch Hamiltonian of the supercell at kS"""
        return self._hk(basistk.pad3(kS))
    def get_hk_gen(self):
        """Function of the Bloch momentum of the ultracell (reduced
        coordinates of the supercell) returning the ultracell matrix, the
        supercell Hamiltonian in the kept folded Bloch states"""
        def f(kS):
            hk = self.hk(kS)
            E,U,valid = self.basis.states(kS)
            return matrixtk.ultracell_matrix(self.basis,hk,kS,E,U,valid)
        return f
    def get_kdos_bands(self,**kwargs):
        return observables.kdos_bands(self,**kwargs)
    def get_dos(self,**kwargs):
        return observables.dos(self,**kwargs)
    def get_ldos(self,**kwargs):
        return observables.ldos(self,**kwargs)
    def get_vev(self,operator=None,**kwargs):
        return observables.vev(self,operator=operator,**kwargs)
    def get_harmonics(self,values):
        return observables.harmonics(self,values)
    def get_mean_field_hamiltonian(self,return_total_energy=False,
            rebuild_basis=True,**kwargs):
        """Self-consistent mean field of the supercell, with the density
        matrix, the Fermi level and the band energy of every iteration
        computed in the ultracell states (integration="ultracell"). It takes
        the keywords of h.get_mean_field_hamiltonian() for a normal-state
        Hamiltonian: U, V1, V2, V3, Vr, and for a spinful one the spin-spin
        exchange J1, J2, J3, Jr (J1x, J1y, J1z), filling or mu, mf, nk (the
        mesh of the ultracell, 1 by default), T, mix, maxerror, maxite, ...
        Returns the Ultracell of the converged mean-field supercell, with
        the same basis, and its total energy with return_total_energy=True.

        A spinful supercell runs on the sparse engine of the spin-spin mean
        field (scftk.spinspin), whose interaction, density matrix and mean
        field are sparse matrices, so the loop needs no dense matrix of the
        supercell; what grows with the size is the ultracell matrix of the
        kept states. A spinless one runs on the dense loop of
        scftk.densitydensity.

        rebuild_basis=True (the default) rebuilds the basis at every
        iteration from the lattice-periodic part of the current
        Hamiltonian, with the same number of states per folded momentum.
        With False the basis stays the one of the clean unit cell, which
        with a cut misses a uniform order: for the antiferromagnet of
        graphene at U=3t, cut at emax=0.5, the moment came out 0.41 off
        with the clean basis and 0.004 off with the rebuilt one"""
        if "integration" in kwargs:
            raise ValueError("the ultracell sets integration=\"ultracell\" "
                    "itself; do not pass integration")
        if self.hs.has_eh:
            raise NotImplementedError("the mean field of the ultracell is "
                    "not implemented for a Hamiltonian with the "
                    "electron-hole (Nambu) degree of freedom")
        exchange = [k for k in ("J1","J2","J3","Jr","J1x","J1y","J1z")
                if kwargs.get(k) is not None and np.any(kwargs.get(k)!=0.)]
        if len(exchange)>0 and not self.hs.has_spin:
            from . import check
            check.require_spin(self.hs,"the spin-spin exchange "
                    +str(exchange))
        hs = self.hs.copy()
        hs.ultracell_basis = basistk.FoldedBasis(self.h,hs,emax=self.emax)
        hs.ultracell_basis.rebuild = bool(rebuild_basis)
        kwargs.setdefault("nk",1)
        out = hs.get_mean_field_hamiltonian(integration="ultracell",
                return_total_energy=return_total_energy,**kwargs)
        h2,etot = out if return_total_energy else (out,None)
        u2 = Ultracell(self.h,h2,emax=self.emax,rebuild_basis=rebuild_basis)
        if return_total_energy: return u2,etot
        return u2


get_ultracell = Ultracell

# the docstrings of the delegators are those of the routines they call
for _name,_f in [("get_kdos_bands",observables.kdos_bands),
        ("get_dos",observables.dos),("get_ldos",observables.ldos),
        ("get_vev",observables.vev),("get_harmonics",observables.harmonics)]:
    getattr(Ultracell,_name).__doc__ = _f.__doc__
