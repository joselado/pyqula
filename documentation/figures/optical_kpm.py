import numpy as np
import style
from pyqula import geometry
np.random.seed(1)
g = geometry.honeycomb_lattice().get_supercell(30) # 1800 sites
h = g.get_hamiltonian(has_spin=False,is_sparse=True)
h.add_sublattice_imbalance(0.5) # gap of 1
h.add_onsite(lambda r: 0.6*(np.random.random()-0.5)) # onsite disorder
ws = np.linspace(0.,4.,100) # frequencies
(ws,s) = h.get_optical_conductivity(energies=ws,integration="kpm",
                                    delta=0.05,ntries=4,seed=1)
i = np.argmax(s[:,0,0].real)
print("maximum of Re sigma_xx:",round(s[i,0,0].real,3),"at w =",round(ws[i],2))

def draw(ax, c):
    ax.plot(ws, s[:,0,0].real, color=c.blue, label=r"Re $\sigma_{xx}$")
    ax.plot(ws, s[:,0,0].imag, color=c.orange, label=r"Im $\sigma_{xx}$")
    ax.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    ax.set_xlabel(r"frequency $\hbar\omega$ [$t$]")
    ax.set_ylabel(r"conductivity [$e^2/\hbar$]")
    ax.legend(loc="upper right")

style.save(draw, "optical_kpm")
