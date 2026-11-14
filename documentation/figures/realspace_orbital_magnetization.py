import numpy as np
import style
from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=10,nedges=6) # 600 sites
h = g.get_hamiltonian(has_spin=False,is_sparse=True)
h.add_haldane(0.2) # Chern insulator, C=1, bulk gap of 1.8
efs = np.linspace(-1.5,1.5,121) # Fermi levels
m = h.get_orbital_magnetization(fermi=efs,T=0.02,trace="exact") # total moment
area = len(g.r)*3*np.sqrt(3)/4 # area of the flake
gap = np.abs(efs)<0.3 # Fermi levels inside the bulk gap
slope = np.polyfit(efs[gap],m[gap],1)[0] # dM/dE_F
print("2 pi (dM/dE_F)/A in the gap:",round(2*np.pi*slope/area,3)) # 0.91

def draw(ax, c):
    ax.axvspan(-0.9, 0.9, color=c.ink2, alpha=0.12, lw=0) # bulk gap
    ax.plot(efs, m/area, color=c.blue, label=r"$M_z/A$")
    ax.plot(efs[np.abs(efs)<0.9], efs[np.abs(efs)<0.9]/(2*np.pi),
            color=c.orange, ls="--", label=r"$C E_F/2\pi$")
    ax.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    ax.set_xlabel(r"Fermi level $E_F$ [$t$]")
    ax.set_ylabel(r"magnetization $M_z/A$ [$et/\hbar$]")
    ax.legend(loc="upper left")

style.save(draw, "realspace_orbital_magnetization")
