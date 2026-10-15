import style
from pyqula import geometry
import numpy as np
g = geometry.chain()
h = g.get_hamiltonian(has_spin=True)
sz = h.get_operator("sz") # any of the operators from the "Operators" section
(es,chi) = h.get_chi(A=sz,B=sz,energies=np.linspace(-2.,2.,100),nk=40,delta=0.1)
(es,chic) = h.get_chi(energies=np.linspace(-2.,2.,100),nk=40,delta=0.1) # charge, as a reference

def draw(ax, c):
    ax.plot(es, chi.imag, color=c.blue, label=r"Im $\chi_{S_zS_z}$")
    ax.plot(es, chi.real, color=c.orange, label=r"Re $\chi_{S_zS_z}$")
    ax.plot(es, chic.imag, "--", color=c.ink, lw=1.0, label="charge")
    ax.plot(es, chic.real, "--", color=c.ink, lw=1.0)
    ax.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    ax.set_xlabel(r"frequency $\omega$ [$t$]")
    ax.set_ylabel(r"local response $\chi(\omega)$")
    ax.legend(loc="upper left")

style.save(draw, "charge_operator")
