import numpy as np
import style
from pyqula import geometry
g = geometry.honeycomb_lattice() # honeycomb lattice, x is the armchair direction
h = g.get_hamiltonian(has_spin=True) # spinful, so both spins are counted
h.add_sublattice_imbalance(3.9/2.33) # onsite +-3.9 eV with t = 2.33 eV, as hBN

# sheet susceptibility chi_xxx in nm^2/V, bond length 1.45 A and t = 2.33 eV
(ws,chi) = h.get_shg_susceptibility(energies=np.linspace(0.,4.,401),
                                    nk=200,T=0.,delta=0.02,components="xxx",
                                    length_unit=1.45,energy_unit=2.33)
print("static chi_xxx in nm^2/V:",chi[0].real)

gap = h.get_gap() # the band gap, 2 times the onsite energy

def draw(ax, c):
    ax.plot(ws, chi.real*1e3, color=c.blue, label=r"Re $\chi_{xxx}$")
    ax.plot(ws, chi.imag*1e3, color=c.orange, label=r"Im $\chi_{xxx}$")
    ax.axvline(gap/2, color=c.ink2, lw=1.0, ls="--")
    ax.text(gap/2-0.03, 0.97, "gap/2", color=c.ink2, va="top", ha="right",
            transform=ax.get_xaxis_transform())
    ax.axvline(gap, color=c.ink2, lw=1.0, ls="--")
    ax.text(gap-0.03, 0.97, "gap", color=c.ink2, va="top", ha="right",
            transform=ax.get_xaxis_transform())
    ax.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    ax.set_xlabel(r"frequency $\hbar\omega$ [$t$]")
    ax.set_ylabel(r"sheet $\chi^{(2)}$ [$10^{-3}$ nm$^2$/V]")
    ax.legend(loc="lower left")

style.save(draw, "optical_shg")
